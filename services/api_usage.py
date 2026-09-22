"""API metering without recording prompts, answers, credentials, or user IDs.

Every event goes to application logs immediately and to TiDB in a background
batch. Prices are estimates at public standard rates, not a provider invoice.
"""
import atexit
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from functools import wraps
import inspect
import json
import logging
import queue
import re
import ssl
import threading
import time
import uuid

import certifi
import pymysql

from config import (API_USAGE_ENABLED, API_USAGE_PERSIST, DB_HOST, DB_NAME,
                    DB_PASSWORD, DB_PORT, DB_SSL, DB_USER)

logger = logging.getLogger(__name__)
PRICING_VERSION = '2026-09-22'
# USD per million tokens. Sources and limitations: docs/api-usage.md.
ANTHROPIC_PRICES = {
    'claude-sonnet-5': ('2', '10'),
    'claude-opus-4-8': ('5', '25'),
    'claude-haiku-4-5': ('1', '5'),
    'claude-sonnet-4-5': ('3', '15'),
    'claude-sonnet-4-6': ('3', '15'),
    'claude-opus-4-5': ('5', '25'),
    'claude-opus-4-6': ('5', '25'),
    'claude-opus-4-7': ('5', '25'),
}
VOYAGE_PRICES = {'voyage-3-large': '0.18', 'rerank-2-lite': '0.02'}
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'cache_read_input_tokens',
                'cache_creation_input_tokens', 'cache_write_5m_tokens',
                'cache_write_1h_tokens')
_context = ContextVar('api_usage_context', default=None)

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS api_usage_events (
    event_id CHAR(32) PRIMARY KEY,
    occurred_at DATETIME(6) NOT NULL,
    `event` JSON NOT NULL,
    INDEX idx_api_usage_time (occurred_at)
)
"""
INSERT_SQL = """
INSERT INTO api_usage_events (event_id, occurred_at, `event`) VALUES (%s, %s, %s)
ON DUPLICATE KEY UPDATE event_id = VALUES(event_id)
"""


@contextmanager
def usage_scope(source):
    """Group related calls; ContextVars propagate through asyncio.to_thread."""
    token = _context.set({'source': source, 'request_id': uuid.uuid4().hex})
    try:
        yield
    finally:
        _context.reset(token)


def usage_source(source):
    """Attach a source and request ID to a web handler, bot command, or job."""
    def decorate(function):
        if inspect.iscoroutinefunction(function):
            @wraps(function)
            async def asynchronous(*args, **kwargs):
                with usage_scope(source):
                    return await function(*args, **kwargs)
            return asynchronous

        @wraps(function)
        def synchronous(*args, **kwargs):
            with usage_scope(source):
                return function(*args, **kwargs)
        return synchronous
    return decorate


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def response_tokens(provider, response):
    if provider == 'voyage':
        total = _count(_get(response, 'total_tokens'))
        return dict.fromkeys(TOKEN_FIELDS, 0) | {'input_tokens': total}
    usage = _get(response, 'usage')
    counts = {key: _count(_get(usage, key, 0)) for key in TOKEN_FIELDS[:4]}
    # Missing usage is unknown, never a free call.
    counts['input_tokens'] = _count(_get(usage, 'input_tokens'))
    counts['output_tokens'] = _count(_get(usage, 'output_tokens'))
    creation = _get(usage, 'cache_creation')
    counts['cache_write_1h_tokens'] = _count(_get(creation, 'ephemeral_1h_input_tokens', 0))
    counts['cache_write_5m_tokens'] = _count(_get(
        creation, 'ephemeral_5m_input_tokens', counts['cache_creation_input_tokens']))
    # Older SDKs expose only the aggregate; this app uses the default 5m TTL.
    return counts


def estimate_cost(provider, model, counts):
    """Return a Decimal USD cost, or None when usage/pricing is unknown."""
    if any(counts.get(key) is None for key in TOKEN_FIELDS):
        return None
    if provider == 'voyage':
        rate = VOYAGE_PRICES.get(model)
        return Decimal(rate) * counts['input_tokens'] / 1_000_000 if rate else None
    # Only strip an exact dated snapshot suffix, not arbitrary model names.
    rates = ANTHROPIC_PRICES.get(re.sub(r'-\d{8}$', '', model)) if provider == 'anthropic' else None
    if rates is None:
        return None
    if counts['cache_write_5m_tokens'] + counts['cache_write_1h_tokens'] != counts['cache_creation_input_tokens']:
        return None
    input_rate, output_rate = map(Decimal, rates)
    return (input_rate * (
        counts['input_tokens']
        + Decimal('0.1') * counts['cache_read_input_tokens']
        + Decimal('1.25') * counts['cache_write_5m_tokens']
        + Decimal('2') * counts['cache_write_1h_tokens']
    ) + output_rate * counts['output_tokens']) / 1_000_000


class APICall:
    def __init__(self, operation, provider, model):
        self.operation, self.provider, self.model = operation, provider, model
        self.response = None

    def capture(self, response):
        self.response = response
        return response

    def __enter__(self):
        self.started = time.monotonic()
        self.occurred_at = datetime.now(timezone.utc).isoformat()
        return self

    def __exit__(self, exc_type, exc, traceback):
        if not API_USAGE_ENABLED:
            return False
        try:
            counts = response_tokens(self.provider, self.response)
            model = _get(self.response, 'model') or self.model
            cost = estimate_cost(self.provider, model, counts)
            # Pricing covers the standard synchronous API calls used here.
            tier = _get(_get(self.response, 'usage'), 'service_tier')
            if tier not in (None, 'standard'):
                cost = None
            _emit({
                **_base_event(self.operation, self.provider, model), **counts,
                'occurred_at': self.occurred_at,
                'requested_model': self.model,
                'provider_request_id': _get(self.response, '_request_id'),
                'status': 'error' if exc_type else 'success',
                'error_type': exc_type.__name__ if exc_type else None,
                'duration_ms': round((time.monotonic() - self.started) * 1000),
                'estimated_cost_usd': str(cost) if cost is not None else None,
                'service_tier': tier,
            })
        except Exception as error:
            # Metering must not turn a successful answer into a failure or mask
            # the provider exception. Never log the exception's sensitive text.
            logger.warning('API usage capture failed (%s)', type(error).__name__)
        return False


def _base_event(operation, provider, model):
    return {
        'type': 'api_usage', 'version': 1, 'event_id': uuid.uuid4().hex,
        'occurred_at': datetime.now(timezone.utc).isoformat(),
        **(_context.get() or {'source': 'unattributed', 'request_id': uuid.uuid4().hex}),
        'operation': operation, 'provider': provider, 'model': model,
        'pricing_version': PRICING_VERSION,
    }


def record_cache_hit(operation, model):
    if API_USAGE_ENABLED:
        try:
            _emit({**_base_event(operation, 'anthropic', model),
                   **dict.fromkeys(TOKEN_FIELDS, 0), 'status': 'cache_hit',
                   'duration_ms': 0, 'estimated_cost_usd': '0'})
        except Exception as error:
            logger.warning('API cache usage capture failed (%s)', type(error).__name__)


def connect_usage_db():
    kwargs = {'ssl': ssl.create_default_context(cafile=certifi.where())} if DB_SSL else {}
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset='utf8mb4', autocommit=True,
        connect_timeout=5, read_timeout=10, write_timeout=10, **kwargs)


class UsageWriter:
    """Bounded, lazy background writer; no DB connection on the request path."""
    def __init__(self):
        self.queue = queue.Queue(maxsize=2000)
        self.lock = threading.Lock()
        self.thread = None
        self.connection = None
        self.stopping = threading.Event()

    def submit(self, event):
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='api-usage', daemon=True)
                self.thread.start()
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            logger.warning('API usage queue full; event %s remains in application logs', event['event_id'])

    def _write_batch(self, events):
        if self.connection is None:
            self.connection = connect_usage_db()
            with self.connection.cursor() as cursor:
                cursor.execute(CREATE_TABLE_SQL)
        rows = [(e['event_id'], datetime.fromisoformat(e['occurred_at']).astimezone(
            timezone.utc).replace(tzinfo=None), json.dumps(e)) for e in events]
        with self.connection.cursor() as cursor:
            cursor.executemany(INSERT_SQL, rows)

    def _run(self):
        while not self.stopping.is_set() or not self.queue.empty():
            try:
                first = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            events = [first]
            # Collect a short batch, amortizing DB writes under normal traffic.
            self.stopping.wait(0.5)
            while len(events) < 100:
                try:
                    events.append(self.queue.get_nowait())
                except queue.Empty:
                    break
            for attempt in range(2):
                try:
                    self._write_batch(events)
                    break
                except Exception as error:
                    if self.connection is not None:
                        try:
                            self.connection.close()
                        except Exception:
                            pass
                        self.connection = None
                    if attempt == 1:
                        logger.warning('API usage persistence failed (%s); %d events remain in application logs',
                                       type(error).__name__, len(events))
                    else:
                        self.stopping.wait(1)
            for _ in events:
                self.queue.task_done()
        if self.connection is not None:
            self.connection.close()

    def close(self):
        self.stopping.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            if self.thread.is_alive():
                logger.warning('API usage shutdown timed out; pending events remain in application logs')


_writer = UsageWriter()
atexit.register(_writer.close)


def _emit(event):
    logger.info('api_usage %s', json.dumps(event, separators=(',', ':')))
    if API_USAGE_PERSIST:
        _writer.submit(event)
