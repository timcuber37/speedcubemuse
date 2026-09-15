"""Free-text question -> Predicate. The only place this feature calls a model.

A player in Mode 1 or 3 types "are they European?" and the game needs to know
that means `continent_id == '_Europe'`. That is a small classification problem
over a closed vocabulary, not a database query — so it runs on Haiku with a
tight output schema, and the answer is cached.

Cost, at Haiku 4.5 ($1/$5 per MTok, cache reads ~0.1x): the vocabulary system
prompt is ~1,200 tokens and identical on every request, so it is cached and
read back at roughly a tenth of list price; the question and the JSON reply are
a few dozen tokens each. That lands near $0.0005 a question before the
predicate cache, and the cache absorbs most repeats — "are they American?" gets
typed in a great many games.

Sync client, like services/rag.py: this path is web-only, so there is no event
loop to block, and the Discord bot never reaches it.
"""
import hashlib
import json
import logging
import re
import ssl
import threading
from collections import OrderedDict

import certifi
import pymysql
from anthropic import Anthropic

from config import (
    ANTHROPIC_API_KEY, DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL,
    GAME_MODEL,
)

from .attributes import (ATTRIBUTES, BY_KEY, EVENT_GROUPS, EVENT_NAMES,
                         Predicate)

_EVENT_IDS = ', '.join(EVENT_NAMES)

logger = logging.getLogger(__name__)

MAX_QUESTION_LENGTH = 200

# Per-process cache of parsed questions. A few thousand entries is a trivial
# amount of memory against the 512 MB machine and covers the long tail of
# phrasings players actually reuse.
LOCAL_CACHE_SIZE = 2000

CACHE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `game_question_cache` (
  `question_hash` CHAR(64) NOT NULL PRIMARY KEY,
  `normalized`    VARCHAR(255) NOT NULL,
  `predicate`     JSON NOT NULL,
  `hits`          INT NOT NULL DEFAULT 0,
  `created_at`    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_OUTPUT_FORMAT = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "attribute": {
                "type": ["string", "null"],
                "description": "The attribute key this question is about, or null if none fits",
            },
            # No `enum` here: the API rejects an enum whose values don't all
            # match a nullable union type, and `op` has to be nullable so the
            # model can decline. _to_predicate validates it against the
            # attribute's kind anyway, which is the stronger check.
            "op": {
                "type": ["string", "null"],
                "description": "One of: 'is' for a boolean attribute, 'eq' for categorical, 'gte' for numeric",
            },
            "value": {
                "type": ["string", "null"],
                "description": "For 'is': true or false. For 'eq': the exact category value. For 'gte': the integer threshold.",
            },
            "error": {
                "type": ["string", "null"],
                "description": "Brief reason the question cannot be mapped, when attribute is null",
            },
        },
        "required": ["attribute", "op", "value", "error"],
        "additionalProperties": False,
    },
}

_SYSTEM_TMPL = """You map a player's yes/no question about a speedcuber onto exactly one \
precomputed attribute.

This is a guessing game. The player is trying to identify a competitor from the World Cube \
Association database by asking yes/no questions. Your only job is to decide which attribute \
the question is asking about, and with what comparison.

Rules:
- Choose exactly one attribute from the list below. Never invent an attribute key.
- 'is' for boolean attributes; value is "true" or "false".
- 'eq' for categorical attributes; value must be one of the listed values, spelled exactly.
- 'gte' for numeric attributes; value is an integer threshold.
- Phrase the comparison so that a "yes" answer matches the player's intent. "Are they \
retired?" maps to is_active is false, not is_active is true.
- If the question cannot be answered from these attributes, set attribute to null and give \
a one-line error. Do this rather than forcing a poor match — a wrong mapping makes the game \
unwinnable and the player cannot tell why.

== ATTRIBUTES ==
{vocabulary}
"""


def _normalize(question: str) -> str:
    """Collapse a question to its cacheable form.

    Deliberately aggressive: 'Are they American?' and 'are they american'
    should share a cache entry. Punctuation and case carry no meaning here.
    """
    text = (question or '').strip().casefold()
    text = re.sub(r"[^\w\s]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


# Display name -> canonical WCA event id, so "Megaminx" and "megaminx" both
# resolve to 'minx'. Without this a display name would pass validation on
# `main_event` (which has no declared value list) and then match no candidate at
# all — every answer silently wrong rather than visibly refused.
_EVENT_BY_LABEL = {label.casefold(): eid for eid, label in EVENT_NAMES.items()}
_EVENT_ALIASES = {
    '3x3x3': '333', '2x2x2': '222', '4x4x4': '444', '5x5x5': '555',
    '6x6x6': '666', '7x7x7': '777', 'oh': '333oh', 'one handed': '333oh',
    'fmc': '333fm', 'fewest moves': '333fm', 'bld': '333bf',
    'blindfolded': '333bf', 'mbld': '333mbf', 'multiblind': '333mbf',
    'multi-blind': '333mbf', 'square one': 'sq1', 'square-1': 'sq1',
}


def _coerce_event_id(value: str) -> str | None:
    """Resolve a loosely-written event to its canonical WCA id, or None."""
    raw = (value or '').strip()
    if raw in EVENT_NAMES:
        return raw
    folded = raw.casefold()
    return (_EVENT_BY_LABEL.get(folded)
            or _EVENT_ALIASES.get(folded)
            or (folded if folded in EVENT_NAMES else None))


def _build_vocabulary() -> str:
    """Render the attribute list for the system prompt.

    Built once at import and never interpolated with per-request data, so the
    cached prefix stays byte-identical — the same discipline nl_to_sql.py
    applies to its schema context.
    """
    lines = []
    for attr in ATTRIBUTES:
        if attr.kind == 'bool':
            spec = 'bool (op: is, value: true|false)'
        elif attr.kind == 'numeric':
            spec = f'numeric (op: gte, thresholds: {", ".join(map(str, attr.thresholds))})'
        elif attr.kind == 'multi':
            spec = (f'set of event ids (op: has, value: one event id from {_EVENT_IDS}) '
                    f'OR (op: has_group, value: one of {", ".join(EVENT_GROUPS)})')
        else:
            if attr.values:
                spec = f'categorical (op: eq, values: {", ".join(attr.values)})'
            elif attr.key == 'country_id':
                spec = ('categorical (op: eq, value: the country name as the WCA spells '
                        'it, e.g. USA, China, Japan, Poland, United Kingdom, Republic of Korea)')
            else:
                spec = f'categorical (op: eq, value: one event id from {_EVENT_IDS})'
        aliases = f"  [also: {', '.join(attr.aliases)}]" if attr.aliases else ''
        lines.append(f'- {attr.key}: {spec}\n  "{attr.question}"{aliases}')
    return '\n'.join(lines)


_VOCABULARY = _build_vocabulary()

# Every cache entry is stamped with a fingerprint of the attribute schema, and
# entries stamped with a different one are ignored.
#
# This matters most for cached *declines*. A question the schema couldn't answer
# yesterday ("top 100 in megaminx") becomes answerable the moment an attribute
# is added, but a cached "I can't answer that" would keep refusing it forever —
# and the failure is invisible, because the endpoint still returns a tidy
# message. Deriving the stamp from the schema means no one has to remember to
# purge anything.
_SCHEMA_VERSION = hashlib.sha256(
    repr([(a.key, a.kind, a.thresholds, a.values, a.event_valued)
          for a in ATTRIBUTES]).encode('utf-8')
).hexdigest()[:12]


class QuestionParser:
    """Maps free-text questions to predicates, with a database-backed cache."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            logger.warning('ANTHROPIC_API_KEY not set — free-text questions disabled.')
            self.client = None
        else:
            self.client = Anthropic(api_key=ANTHROPIC_API_KEY)
        self.model = GAME_MODEL
        self.system_prompt = _SYSTEM_TMPL.format(vocabulary=_VOCABULARY)

        # Tier 1 cache: per-process, instant. Tier 2 is the shared table, which
        # survives restarts and warms new workers.
        self._local: OrderedDict[str, dict] = OrderedDict()
        self._local_lock = threading.Lock()
        self._conn = None
        self._conn_lock = threading.RLock()
        self._table_ready = False

    # -- cache -------------------------------------------------------------
    #
    # Two tiers, because opening a TiDB Serverless connection costs ~1.1s from
    # this region — as much as the model call the cache exists to avoid. A
    # database-only cache would save money and lose all of that saving in
    # latency, so the in-process tier goes first and the connection is reused
    # rather than dialled per lookup.

    def _connection(self):
        """A reused connection, reconnected if the server dropped it.

        Gunicorn sync workers serve one request at a time, so a per-process
        connection is safe there; the lock covers the threaded dev server.
        """
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.ping(reconnect=True)
                    return self._conn
                except Exception:
                    self._conn = None  # fall through and redial

            kwargs = {}
            if DB_SSL:
                kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
            self._conn = pymysql.connect(
                host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
                database=DB_NAME, charset='utf8mb4', autocommit=True,
                connect_timeout=5, read_timeout=10, **kwargs,
            )
            return self._conn

    def _local_get(self, key: str):
        with self._local_lock:
            if key in self._local:
                self._local.move_to_end(key)
                return self._local[key]
        return None

    def _local_put(self, key: str, payload: dict) -> None:
        with self._local_lock:
            self._local[key] = payload
            self._local.move_to_end(key)
            while len(self._local) > LOCAL_CACHE_SIZE:
                self._local.popitem(last=False)

    def _cache_get(self, key: str) -> dict | None:
        local = self._local_get(key)
        if local is not None:
            return local

        try:
            conn = self._connection()
            with self._conn_lock:
                cur = conn.cursor()
                cur.execute(
                    'SELECT `predicate` FROM `game_question_cache` '
                    'WHERE `question_hash` = %s', (key,)
                )
                row = cur.fetchone()
                if row:
                    cur.execute(
                        'UPDATE `game_question_cache` SET `hits` = `hits` + 1 '
                        'WHERE `question_hash` = %s', (key,)
                    )
                cur.close()
        except pymysql.err.ProgrammingError as e:
            # 1146 = table not created yet. Expected on a fresh deployment
            # until the first cache write creates it; not worth a warning.
            if e.args and e.args[0] == 1146:
                logger.debug('question cache table not created yet')
            else:
                logger.warning('question cache read failed: %s', e)
            return None
        except Exception as e:
            # A cache outage must not take the game down — fall through to the
            # model and let the request succeed at full price.
            logger.warning('question cache read failed: %s', e)
            self._conn = None
            return None

        if not row:
            return None
        value = row[0]
        if isinstance(value, (str, bytes, bytearray)):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return None
        self._local_put(key, value)
        return value

    def _cache_put(self, key: str, normalized: str, payload: dict) -> None:
        self._local_put(key, payload)
        try:
            conn = self._connection()
            with self._conn_lock:
                cur = conn.cursor()
                if not self._table_ready:
                    cur.execute(CACHE_TABLE_SQL)
                    self._table_ready = True
                cur.execute(
                    'INSERT INTO `game_question_cache` '
                    '(`question_hash`, `normalized`, `predicate`) VALUES (%s, %s, %s) '
                    'ON DUPLICATE KEY UPDATE `predicate` = VALUES(`predicate`)',
                    (key, normalized[:255], json.dumps(payload)),
                )
                cur.close()
        except Exception as e:
            logger.warning('question cache write failed: %s', e)
            self._conn = None

    # -- parsing -----------------------------------------------------------

    def parse(self, question: str) -> tuple[Predicate | None, str | None]:
        """Map a question to a Predicate. Returns (predicate, error_message)."""
        if not question or not question.strip():
            return None, 'Ask a question first.'
        if len(question) > MAX_QUESTION_LENGTH:
            return None, f'Keep questions under {MAX_QUESTION_LENGTH} characters.'

        normalized = _normalize(question)
        if not normalized:
            return None, "That doesn't look like a question."

        key = hashlib.sha256(normalized.encode('utf-8')).hexdigest()

        cached = self._cache_get(key)
        if cached is not None and cached.get('v') == _SCHEMA_VERSION:
            if cached.get('error'):
                return None, cached['error']
            pred = self._to_predicate(cached)
            if pred:
                return pred, None
            # Belt and braces: the stamp matched but the payload still doesn't
            # resolve. Re-ask rather than serve something broken.
            logger.info('unresolvable cache entry for %r — reparsing', normalized)
        elif cached is not None:
            logger.info('cache entry for %r predates the current attribute '
                        'schema — reparsing', normalized)

        if self.client is None:
            return None, 'Free-text questions are unavailable right now.'

        try:
            payload = self._ask_model(question)
        except Exception as e:
            logger.error('question parse failed: %s', e)
            return None, 'Could not read that question. Try rephrasing it.'

        pred = self._to_predicate(payload)
        if pred is None:
            error = payload.get('error') or "I can't answer that kind of question."
            self._cache_put(key, normalized, {'v': _SCHEMA_VERSION, 'error': error})
            return None, error

        self._cache_put(key, normalized, {
            'v': _SCHEMA_VERSION,
            'attribute': pred.key, 'op': pred.op, 'value': str(pred.value),
        })
        return pred, None

    def _ask_model(self, question: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=150,
            # Identical on every request, so it is read back from cache at
            # roughly a tenth of the input price.
            system=[{
                'type': 'text',
                'text': self.system_prompt,
                'cache_control': {'type': 'ephemeral'},
            }],
            messages=[{'role': 'user', 'content': question.strip()}],
            output_config={'format': _OUTPUT_FORMAT},
            # Haiku 4.5 does not run thinking unless asked, but being explicit
            # keeps latency predictable if the default model is ever changed.
            thinking={'type': 'disabled'},
        )
        usage = response.usage
        logger.info(
            'game question parse: model=%s cache_read=%s cache_write=%s',
            response.model, usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        text = next((b.text for b in response.content if b.type == 'text'), '')
        return json.loads(text)

    def _to_predicate(self, payload: dict) -> Predicate | None:
        """Validate a model or cache payload into a Predicate.

        Everything here is untrusted: the model can hallucinate an attribute key
        and a cache row can outlive a schema change. A predicate that doesn't
        typecheck is rejected rather than evaluated.
        """
        key = payload.get('attribute')
        op = payload.get('op')
        raw = payload.get('value')
        if not key or not op:
            return None

        attr = BY_KEY.get(key)
        if attr is None:
            logger.warning('parser returned unknown attribute: %r', key)
            return None

        if attr.kind == 'multi':
            # The value decides the op, not the model. It reaches for 'eq' on a
            # set-valued attribute often enough that rejecting the mismatch
            # meant declining perfectly clear questions ("top 100 in megaminx"),
            # and there is only one sensible reading of each value anyway: a
            # group name means 'has_group', an event id means 'has'.
            value = str(raw).strip()
            if value in EVENT_GROUPS:
                return Predicate(key, 'has_group', value)
            event_id = _coerce_event_id(value)
            if event_id:
                return Predicate(key, 'has', event_id)
            logger.warning('parser returned %r, which is neither an event nor a '
                           'group, for %s', value, key)
            return None

        expected = {'bool': ('is',), 'numeric': ('gte',), 'categorical': ('eq',)}[attr.kind]
        if op not in expected:
            logger.warning('parser used op %r for %s attribute %s', op, attr.kind, key)
            return None

        if attr.kind == 'bool':
            return Predicate(key, 'is', str(raw).strip().lower() in ('true', '1', 'yes'))

        if attr.kind == 'numeric':
            try:
                value = int(float(str(raw).strip()))
            except (TypeError, ValueError):
                return None
            # Snap to the nearest declared threshold. The engine's questions are
            # built from those, so an off-list threshold would create a question
            # that can never be asked again and skews the asked-set bookkeeping.
            if attr.thresholds:
                value = min(attr.thresholds, key=lambda t: abs(t - value))
            return Predicate(key, 'gte', value)

        value = str(raw).strip()
        if attr.values and value not in attr.values:
            # Tolerate a missing leading underscore on continent ids, which the
            # model drops often because that is how they read to a human.
            candidate = f'_{value}'
            if candidate in attr.values:
                return Predicate(key, 'eq', candidate)
            logger.warning('parser returned %r outside %s values', value, key)
            return None

        if key == 'main_event':
            # Stored as a WCA event id. This attribute declares no value list —
            # it is filled from the data — so without this an unresolved display
            # name would match nobody and answer "no" to everything.
            event_id = _coerce_event_id(value)
            if event_id is None:
                logger.warning('parser returned unrecognised event %r', value)
                return None
            return Predicate(key, 'eq', event_id)

        return Predicate(key, 'eq', value)
