"""App-side read path for the `cuber_profiles` matrix.

Deliberately shaped like `services/site_meta.py`: a process-local TTL cache in
front of a short-timeout PyMySQL read, so a database hiccup degrades instead of
hanging a page render. The differences are that there is no baked-in fallback
(a game with no candidates cannot be faked, so the routes surface the outage)
and that the cached value includes a built `Engine` per tier.

Rows are cached per depth and Engines are cached per difficulty, so a worker
pays the load once an hour rather than once a turn. Two Gunicorn workers each
keep their own copy; the matrix is read-only and rebuilt weekly, so there is
nothing to invalidate between them.

Memory is the design constraint here. The machine has 512 MB and runs two
workers, so the Everyone pool (~52k rows) is loaded only when that difficulty
is actually played, and `_shape` folds duplicate strings onto shared objects —
together that keeps a worker holding the largest pool at roughly 100 MB instead
of 320 MB. Everything below five competitions stays in the database entirely.
"""
import json
import logging
import random
import ssl
import threading
import time

import certifi
import pymysql

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL

from .attributes import ATTRIBUTES
from .engine import Engine

logger = logging.getLogger(__name__)

# Difficulty name -> the tiers it plays over. Cumulative: Normal includes the
# Easy pool, Hard includes both, Everyone includes the lot.
TIERS = {
    'easy': (1,),
    'normal': (1, 2),
    'hard': (1, 2, 3),
    'everyone': (1, 2, 3, 4),
}
DEFAULT_TIER = 'normal'

DIFFICULTY_LABELS = {
    'easy': 'the 300 best-known competitors',
    'normal': '1,000 competitors',
    'hard': 'every record holder and current world top 100',
    'everyone': 'every competitor with 5+ competitions',
}

# The matrix only changes when the weekly refresh rebuilds it.
_CACHE_TTL_SECONDS = 3600

# Rows are cached per depth, not all at once. The elite tiers are ~2k rows and
# load in about a second; the Everyone pool is ~50k and takes considerably
# longer. Loading the big one eagerly would make every difficulty pay that cost
# on a cold start — and the Fly machine suspends when idle, so cold starts are
# routine rather than rare.
_ELITE_MAX_TIER = 3
_cache_rows: dict[int, list[dict]] = {}
_cache_expires_at: dict[int, float] = {}
_cache_engines: dict[str, Engine] = {}
_cache_lock = threading.Lock()

# Profiles fetched individually from the non-resident tier.
_TAIL_CACHE_SIZE = 512
_tail_cache: dict[str, dict] = {}
_total_profiles: int | None = None


# Per-request reads are single indexed rows or a capped search, so a short
# timeout is right — better to degrade than to hang a page. The bulk pool load
# fetches ~52k rows and legitimately takes seconds, and timing it out leaves the
# worker with no pool at all, so it gets its own budget.
_READ_TIMEOUT = 15
_BULK_READ_TIMEOUT = 90


def _connect(read_timeout: int = _READ_TIMEOUT):
    kwargs = {}
    if DB_SSL:
        # certifi rather than the OS trust store, matching WCAService — some
        # hosts carry a stale root that breaks TiDB Cloud cert verification.
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset='utf8mb4',
        connect_timeout=5, read_timeout=read_timeout,
        **kwargs,
    )


_shared_conn = None
_conn_lock = threading.RLock()


def _shared():
    """A reused connection for the per-request paths.

    Opening a TiDB Serverless connection costs ~1.1s from this region, which
    dominated both the long-tail lookup and the name autocomplete — the
    handshake was taking longer than the query. The hourly bulk load still dials
    its own, since that one is rare and long-running.
    """
    global _shared_conn
    with _conn_lock:
        if _shared_conn is not None:
            try:
                _shared_conn.ping(reconnect=True)
                return _shared_conn
            except Exception:
                _shared_conn = None
        _shared_conn = _connect()
        return _shared_conn


def _query(sql: str, params: tuple):
    """Run a read on the shared connection, dropping it if it goes bad."""
    global _shared_conn
    with _conn_lock:
        try:
            conn = _shared()
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall()
            cur.close()
            return rows
        except Exception:
            _shared_conn = None
            raise


_COLUMNS = 'wca_id, name, country_id, continent_id, fame, tier, attrs'

# Tiers held in process memory. Tier 5 — competitors with fewer than five
# competitions — stays in the database: there are ~245k of them, the guessing
# engine could not separate them anyway, and "you guess mine" reaches them with
# a single indexed lookup instead.
RESIDENT_TIERS = 4


def _read_all(max_tier: int) -> list[dict]:
    conn = _connect(read_timeout=_BULK_READ_TIMEOUT)
    try:
        cur = conn.cursor()
        cur.execute(
            f'SELECT {_COLUMNS} FROM `cuber_profiles` WHERE tier <= %s',
            (min(max_tier, RESIDENT_TIERS),),
        )
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return _shape(rows)


def _shape(rows) -> list[dict]:
    """Turn raw DB tuples into profile dicts.

    Keys and repeated string values are folded onto shared objects. json.loads
    allocates a fresh string for every key in every row, so a 50k-row pool
    otherwise carries ~1.8M duplicate copies of the same 37 attribute names —
    which measured as the single largest use of memory in the worker, well
    ahead of the values themselves.
    """
    profiles = []
    keys: dict[str, str] = {a.key: a.key for a in ATTRIBUTES}
    # Country, continent and event values repeat across tens of thousands of
    # rows, so they collapse onto one object each too.
    values: dict[str, str] = {}

    for wca_id, name, country, continent, fame, tier, attrs in rows:
        # TiDB may hand back a JSON column as str or as already-decoded dict
        # depending on driver version; normalize rather than assume.
        if isinstance(attrs, (str, bytes, bytearray)):
            try:
                attrs = json.loads(attrs)
            except (json.JSONDecodeError, TypeError):
                logger.warning('cuber_profiles.%s has unreadable attrs', wca_id)
                continue

        folded = {}
        for k, v in (attrs or {}).items():
            if type(v) is str:
                v = values.setdefault(v, v)
            elif type(v) is list:
                v = tuple(values.setdefault(x, x) if type(x) is str else x
                          for x in v)
            folded[keys.setdefault(k, k)] = v

        profiles.append({
            'wca_id': wca_id,
            'name': name,
            'country_id': values.setdefault(country, country) if country else country,
            'continent_id': (values.setdefault(continent, continent)
                             if continent else continent),
            'fame': int(fame),
            'tier': int(tier),
            'attrs': folded,
        })
    return profiles


def load_profiles(max_tier: int = _ELITE_MAX_TIER,
                  force_refresh: bool = False) -> list[dict]:
    """Profile rows down to `max_tier`, cached per process for an hour.

    Returns an empty list if the table is missing or unreadable — callers should
    treat that as "the game is unavailable" rather than "nobody qualifies".
    """
    depth = min(max_tier, RESIDENT_TIERS)

    now = time.monotonic()
    if not force_refresh and now < _cache_expires_at.get(depth, 0):
        return _cache_rows[depth]

    with _cache_lock:
        # Another thread may have refreshed while we waited for the lock.
        now = time.monotonic()
        if not force_refresh and now < _cache_expires_at.get(depth, 0):
            return _cache_rows[depth]

        try:
            rows = _read_all(depth)
        except Exception as e:
            logger.warning('Could not read cuber_profiles: %s', e)
            rows = _cache_rows.get(depth, [])

        _cache_rows[depth] = rows
        _cache_expires_at[depth] = time.monotonic() + _CACHE_TTL_SECONDS
        # Engines are derived from these rows, so they have to go too.
        _cache_engines.clear()
        return rows


def get_engine(difficulty: str = DEFAULT_TIER) -> Engine | None:
    """The cached Engine for a difficulty, or None if the matrix is unavailable."""
    tiers = TIERS.get(difficulty)
    if tiers is None:
        raise ValueError(f'unknown difficulty: {difficulty}')

    rows = load_profiles(max(tiers))
    if not rows:
        return None

    with _cache_lock:
        engine = _cache_engines.get(difficulty)
        if engine is None:
            pool = [r for r in rows if r['tier'] in tiers]
            if not pool:
                return None
            engine = Engine(pool)
            _cache_engines[difficulty] = engine
        return engine


def _cached_rows() -> list[dict]:
    """Every row currently resident, across whatever depths have been loaded."""
    if not _cache_rows:
        return load_profiles()
    deepest = max(_cache_rows)
    return _cache_rows[deepest]


def find_by_wca_id(wca_id: str) -> dict | None:
    """Look up one profile, falling back to the database for the long tail.

    Most lookups hit the resident pool. "You guess mine" on the Everyone setting
    can hide any competitor, though, including the ~245k who are not resident —
    one indexed row is cheap, where keeping them all in memory is not.
    """
    if not wca_id:
        return None
    for row in _cached_rows():
        if row['wca_id'] == wca_id:
            return row

    cached = _tail_cache.get(wca_id)
    if cached is not None:
        return cached

    try:
        rows = _query(
            f'SELECT {_COLUMNS} FROM `cuber_profiles` WHERE wca_id = %s',
            (wca_id,),
        )
    except Exception as e:
        logger.warning('profile lookup failed for %s: %s', wca_id, e)
        return None

    shaped = _shape(rows)
    if not shaped:
        return None
    # Bounded: a game asks about its secret once per turn, and this keeps that
    # to one round trip per game rather than one per question.
    if len(_tail_cache) > _TAIL_CACHE_SIZE:
        _tail_cache.clear()
    _tail_cache[wca_id] = shaped[0]
    return shaped[0]


def random_profile(difficulty: str) -> dict | None:
    """A random competitor to hide in "you guess mine".

    On Everyone this reaches past the resident pool to every competitor in the
    database, so the secret really can be anyone.
    """
    if difficulty != 'everyone':
        engine = get_engine(difficulty)
        return engine.pick_secret() if engine else None

    try:
        # Random offset rather than ORDER BY RAND(), which would sort the whole
        # table on every game start. The count is cached for the same reason the
        # rows are — the table only changes on the weekly rebuild.
        global _total_profiles
        if _total_profiles is None:
            _total_profiles = _query('SELECT COUNT(*) FROM `cuber_profiles`', ())[0][0]
        total = _total_profiles
        if not total:
            return None
        rows = _query(
            f'SELECT {_COLUMNS} FROM `cuber_profiles` LIMIT 1 OFFSET %s',
            (random.randrange(total),),
        )
    except Exception as e:
        logger.warning('random profile lookup failed: %s', e)
        engine = get_engine('hard')
        return engine.pick_secret() if engine else None

    shaped = _shape(rows)
    return shaped[0] if shaped else None


def search_by_name(query: str, limit: int = 8, include_all: bool = False) -> list[dict]:
    """Substring name search for the guess autocomplete.

    The resident pool is searched in process — it is small, this is typed per
    keystroke, and a round trip per keystroke would be slow and pointless.

    `include_all` additionally searches the database, which is what makes the
    Everyone setting playable: the competitor being hidden may be one of the
    ~245k who are not resident, and a player who cannot type their name cannot
    guess them.
    """
    q = (query or '').strip().casefold()
    if len(q) < 2:
        return []

    hits = [r for r in _cached_rows() if q in r['name'].casefold()]
    if include_all and len(hits) < limit:
        seen = {r['wca_id'] for r in hits}
        hits += [r for r in _search_db(q, limit) if r['wca_id'] not in seen]

    hits.sort(key=lambda r: (not r['name'].casefold().startswith(q), -r['fame']))
    return hits[:limit]


def _search_db(q: str, limit: int) -> list[dict]:
    """Name search across every profile, resident or not.

    LOWER() on both sides rather than a bare LIKE: the table inherits TiDB's
    default utf8mb4_bin collation, under which LIKE is case-sensitive, so a
    lowercased query silently matched nothing at all.
    """
    try:
        rows = _query(
            f'SELECT {_COLUMNS} FROM `cuber_profiles` '
            f'WHERE LOWER(name) LIKE %s ORDER BY fame DESC LIMIT %s',
            (f'%{q}%', limit),
        )
        return _shape(rows)
    except Exception as e:
        logger.warning('name search failed: %s', e)
        return []
