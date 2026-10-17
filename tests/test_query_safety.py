"""Guards on the main query feature, where the SQL comes from a model a user can steer.

Treat the model as untrusted: anything a user can talk it into writing must be
stopped by the validator, bounded by the time and row limits, or throttled by
the rate limits. Fully offline — no model, database, or network calls.

    python -m pytest tests/test_query_safety.py -v
"""
import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import limits.storage.memory
import pytest
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent.parent))

import extensions
from config import MAX_GUEST_QUERY_QUESTIONS, QUERY_TIMEOUT_SECONDS
from services import wca_api
from services.nl_to_sql import NLToSQLService, _execution_error


@pytest.fixture(scope='module')
def service():
    return NLToSQLService()


# ---------------------------------------------------------------------------
# One SELECT, no comments
# ---------------------------------------------------------------------------

def _prompt_examples(service):
    """The SQL examples the system prompt shows the model, one per block."""
    examples = service.schema_context.split('EXAMPLE QUERIES:')[1]
    blocks = [b.strip() for b in examples.split('\n\n') if b.strip()]
    return ['\n'.join(b.splitlines()[1:]) for b in blocks]  # drop the caption


def test_every_prompt_example_passes(service):
    """The validator must accept the shapes the model is taught to write."""
    examples = _prompt_examples(service)
    assert len(examples) >= 10
    for sql in examples:
        assert service.validate_sql(sql), sql


@pytest.mark.parametrize('sql', [
    'SELECT 1; DROP TABLE persons',
    # MySQL runs the body of a versioned comment, and there is no word
    # boundary between '0' and 'D' for the keyword check to catch.
    'SELECT 1; /*!50000DROP TABLE persons*/',
    'SELECT 1 /*!50000INTO OUTFILE "/tmp/x"*/',
    'SELECT 1; RENAME TABLE persons TO persons_old',
    "SELECT 1; SET PASSWORD = 'x'",
    'SELECT 1; LOCK TABLES results WRITE',
    "SELECT * FROM persons INTO\nOUTFILE '/tmp/x'",
    'SELECT 1 -- comment',
    'SELECT 1 # comment',
    'SELECT 1;',
])
def test_rejects_extra_statements_comments_and_file_writes(service, sql):
    assert not service.validate_sql(sql)


def _model_reply(sql):
    text = json.dumps({'sql': sql, 'error': None})
    return SimpleNamespace(stop_reason='end_turn',
                           content=[SimpleNamespace(type='text', text=text)])


def test_one_trailing_semicolon_is_stripped_not_rejected(service, monkeypatch):
    monkeypatch.setattr(service, '_create_message',
                        AsyncMock(return_value=_model_reply('SELECT 1;  ')))
    assert asyncio.run(service._generate_sql('q')) == 'SELECT 1'


def test_stripping_the_trailing_semicolon_cannot_unlock_a_second_statement(service, monkeypatch):
    monkeypatch.setattr(service, '_create_message',
                        AsyncMock(return_value=_model_reply('SELECT 1; DROP TABLE persons;')))
    assert asyncio.run(service._generate_sql('q')) is None


# ---------------------------------------------------------------------------
# Time and row limits
# ---------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, rows, delay=0.0):
        self.rows, self.delay, self.closed = rows, delay, False

    async def execute(self, sql):
        await asyncio.sleep(self.delay)

    async def fetchmany(self, size):
        return self.rows[:size]

    async def close(self):
        self.closed = True


class FakeConn:
    def __init__(self, cursor):
        self._cursor, self.closed = cursor, False

    async def cursor(self, cursor_class):
        return self._cursor

    def close(self):
        self.closed = True


class FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        return self

    async def __aenter__(self):
        return self.conn

    async def __aexit__(self, *exc):
        return False


def _service_with(cursor):
    service = wca_api.WCAService()
    conn = FakeConn(cursor)
    service.pool = FakePool(conn)
    return service, conn


def test_pool_sets_a_server_side_time_limit(monkeypatch):
    create_pool = AsyncMock(return_value=object())
    monkeypatch.setattr(wca_api.aiomysql, 'create_pool', create_pool)
    asyncio.run(wca_api.WCAService()._get_pool())
    assert create_pool.await_args.kwargs['init_command'] == (
        f'SET SESSION max_execution_time = {QUERY_TIMEOUT_SECONDS * 1000}')


def test_small_results_keep_the_connection():
    cursor = FakeCursor([{'n': i} for i in range(3)])
    service, conn = _service_with(cursor)
    assert len(asyncio.run(service.execute_query('SELECT n'))) == 3
    assert cursor.closed and not conn.closed


def test_rows_past_the_cap_are_dropped_with_the_connection():
    """Draining millions of unread rows would cost more than reconnecting."""
    cursor = FakeCursor([{'n': i} for i in range(wca_api.MAX_FETCH_ROWS + 5)])
    service, conn = _service_with(cursor)
    assert len(asyncio.run(service.execute_query('SELECT n'))) == wca_api.MAX_FETCH_ROWS
    assert conn.closed and not cursor.closed


def test_slow_query_times_out_and_drops_the_connection(monkeypatch):
    monkeypatch.setattr(wca_api, 'CLIENT_TIMEOUT_SECONDS', 0.05)
    service, conn = _service_with(FakeCursor([], delay=5))
    results = asyncio.run(service.execute_query('SELECT SLEEP(600)'))
    # A non-empty message, so the repair loop and summarizer see an error
    # rather than a one-row result.
    assert 'timed out' in _execution_error(results)
    assert conn.closed


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------

def _ip_for(monkeypatch, on_fly, headers):
    monkeypatch.setattr(extensions, '_ON_FLY', on_fly)
    with Flask(__name__).test_request_context(
            headers=headers, environ_base={'REMOTE_ADDR': '10.0.0.1'}):
        return extensions.client_ip()


def test_on_fly_the_limit_keys_on_the_visitor_not_the_proxy(monkeypatch):
    assert _ip_for(monkeypatch, True, {'Fly-Client-IP': '203.0.113.7'}) == '203.0.113.7'
    assert _ip_for(monkeypatch, True, {}) == '10.0.0.1'


def test_off_fly_the_header_is_ignored(monkeypatch):
    """Off Fly no proxy sets the header, so any client could send its own."""
    assert _ip_for(monkeypatch, False, {'Fly-Client-IP': '203.0.113.7'}) == '10.0.0.1'


@pytest.fixture
def query_client(monkeypatch):
    """The real /query route and limiter, with the model and database stubbed.

    The limiter's clock is replaced so a test can step past the per-minute
    window and reach the daily cap behind it.
    """
    import app as flask_app
    monkeypatch.setattr(flask_app.nl_to_sql_service, 'answer_question',
                        AsyncMock(return_value=('SELECT 1', [{'n': 1}])))
    monkeypatch.setattr(flask_app.nl_to_sql_service, 'summarize_results',
                        AsyncMock(return_value=None))
    monkeypatch.setattr(flask_app, 'get_current_user', lambda: (None, None))
    clock = [1_000_000.0]
    monkeypatch.setattr(limits.storage.memory, 'time',
                        SimpleNamespace(time=lambda: clock[0]))
    flask_app.limiter.reset()
    yield flask_app, flask_app.app.test_client(), clock
    flask_app.limiter.reset()


def _ask_spread_over_minutes(client, clock, count):
    """Ask `count` questions, ten a minute, to stay under the per-minute limit."""
    responses = []
    for i in range(count):
        if i and i % 10 == 0:
            clock[0] += 61
        responses.append(client.post('/query', json={'question': 'q'}))
    return responses


def test_per_minute_limit_answers_in_json(query_client):
    _, client, _ = query_client
    responses = [client.post('/query', json={'question': 'q'}) for _ in range(11)]
    assert [r.status_code for r in responses[:10]] == [200] * 10
    assert responses[10].status_code == 429
    assert 'too quickly' in responses[10].json['error']


def test_guests_hit_a_server_side_daily_cap(query_client):
    _, client, clock = query_client
    responses = _ask_spread_over_minutes(client, clock, MAX_GUEST_QUERY_QUESTIONS + 1)
    assert [r.status_code for r in responses[:-1]] == [200] * MAX_GUEST_QUERY_QUESTIONS
    assert responses[-1].status_code == 429
    assert 'Sign in' in responses[-1].json['error']


def test_signed_in_users_are_exempt_from_the_daily_cap(query_client, monkeypatch):
    flask_app, client, clock = query_client
    monkeypatch.setattr(flask_app, 'get_current_user', lambda: (object(), 'token'))
    responses = _ask_spread_over_minutes(client, clock, MAX_GUEST_QUERY_QUESTIONS + 1)
    assert all(r.status_code == 200 for r in responses)
