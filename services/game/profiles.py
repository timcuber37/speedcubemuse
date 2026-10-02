"""App-side read path for the `cuber_profiles` matrix.

Deliberately shaped like `services/site_meta.py`: a process-local TTL cache in
front of a short-timeout PyMySQL read, so a database hiccup degrades instead of
hanging a page render. The differences are that there is no baked-in fallback
(a game with no candidates cannot be faked, so the routes surface the outage)
and that the cached value includes a built `Engine` per tier.

Rows are cached per depth and Engines are cached per difficulty. An hourly
check of the monthly rebuild's generation marker avoids re-reading unchanged
profiles. Two Gunicorn workers each keep their own copy.

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
from array import array
from bisect import bisect_right
from heapq import nlargest

import certifi
import pymysql

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL
from services.site_meta import GAME_PROFILES_KEY

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

# Check for a new generation hourly. A full read every 30 days also recovers if
# a writer changed profiles without publishing the marker. The rebuild script
# always publishes it, so a shorter interval would only repeat the largest
# reads this module makes against a matrix that changes monthly.
_CACHE_TTL_SECONDS = 3600
_FULL_REFRESH_SECONDS = 30 * 86400
# A failed/empty read is not a fresh snapshot. Briefly back off, then try again
# so restoring database access does not leave games unavailable for an hour.
_CACHE_RETRY_SECONDS = 15

# Rows are cached per depth, not all at once. The elite tiers are ~2k rows and
# load in about a second; the Everyone pool is ~50k and takes considerably
# longer. Loading the big one eagerly would make every difficulty pay that cost
# on a new worker or after a deployment.
_ELITE_MAX_TIER = 3
_cache_rows: dict[int, list[dict]] = {}
_cache_expires_at: dict[int, float] = {}
_cache_loaded_at: dict[int, float] = {}
_cache_generation: str | None = None
_cache_engines: dict[str, Engine] = {}
_cache_lock = threading.Lock()

# Profiles fetched individually from the non-resident tier.
_TAIL_CACHE_SIZE = 512
_tail_cache: dict[str, dict] = {}

# A compact directory serves random selection and name search. It contains no
# feature JSON, so it is much smaller than the Everyone guessing pool.
_all_ids: list[str] = []
_name_index = None
_ids_generation: str | None = None
_ids_expires_at = 0.0
_ids_loaded_at = 0.0
_ids_lock = threading.Lock()

_SEARCH_CACHE_SECONDS = 900
_SEARCH_CACHE_SIZE = 512
_search_cache: dict[tuple[str, int], tuple[float, list[dict]]] = {}


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


def _read_generation() -> str | None:
    """The marker written only after the monthly profile rebuild completes."""
    try:
        rows = _query('SELECT `value` FROM `site_meta` WHERE `key` = %s',
                      (GAME_PROFILES_KEY,))
    except pymysql.err.ProgrammingError as e:
        if e.args[0] != 1146:  # An older local DB can lack site_meta entirely.
            raise
        rows = []
    if rows:
        return str(rows[0][0])
    # Existing deployments predate the marker. A point read of the first
    # primary-key row gives them a changing fingerprint until the next rebuild.
    rows = _query('SELECT wca_id, updated_at FROM `cuber_profiles` '
                  'ORDER BY wca_id LIMIT 1', ())
    return f'legacy:{rows[0][0]}:{rows[0][1]}' if rows else None


class _CompactNameIndex:
    """Names in one UTF-8 buffer, searched in fame order without DB scans."""

    def __init__(self, rows):
        self.ids = []
        self.names = bytearray()
        self.offsets = array('I', [0])
        self.fame = array('i')
        for wca_id, name, score in rows:
            self.ids.append(wca_id)
            self.names.extend(name.casefold().encode('utf-8'))
            self.offsets.append(len(self.names))
            self.fame.append(int(score))
        self.by_fame = array('I', sorted(range(len(self.ids)),
                                         key=lambda i: -self.fame[i]))

    def find(self, query: str, limit: int) -> list[str]:
        if limit <= 0:
            return []
        needle = query.encode('utf-8')
        first = self.names.find(needle)
        if first < 0:
            return []

        # Common, short searches find their best matches quickly in fame order.
        if len(needle) < 3:
            return self._find_by_fame(needle, limit)

        # For a specific query, let the byte search skip most names in C. The
        # offsets map hits back to profiles, rejecting matches across names.
        matches = []
        start = first
        last_index = -1
        while start >= 0:
            i = bisect_right(self.offsets, start) - 1
            if (i != last_index and start + len(needle) <= self.offsets[i + 1]):
                matches.append(i)
                last_index = i
                if len(matches) >= 512:
                    return self._find_by_fame(needle, limit)
            start = self.names.find(needle, start + 1)
        best = nlargest(limit, matches, key=lambda i: (self.fame[i], -i))
        return [self.ids[i] for i in best]

    def _find_by_fame(self, needle: bytes, limit: int) -> list[str]:
        matches = []
        for i in self.by_fame:
            if self.names.find(needle, self.offsets[i], self.offsets[i + 1]) >= 0:
                matches.append(self.ids[i])
                if len(matches) >= limit:
                    break
        return matches


def _fresh_covering_rows(depth: int, now: float) -> list[dict] | None:
    """A cached larger pool can also serve every smaller difficulty."""
    for loaded_depth in sorted(_cache_rows, reverse=True):
        if loaded_depth < depth or now >= _cache_expires_at.get(loaded_depth, 0):
            continue
        rows = _cache_rows[loaded_depth]
        if loaded_depth != depth:
            rows = [row for row in rows if row['tier'] <= depth]
            _cache_rows[depth] = rows
            _cache_expires_at[depth] = _cache_expires_at[loaded_depth]
            _cache_loaded_at[depth] = _cache_loaded_at[loaded_depth]
        return rows
    return None

# Tiers held in process memory. Tier 5 — competitors with fewer than five
# competitions — stays in the database: there are ~245k of them, the guessing
# engine could not separate them anyway, and "you guess mine" reaches them with
# a single indexed lookup instead.
RESIDENT_TIERS = 4


def _read_all(max_tier: int) -> list[dict]:
    started = time.perf_counter()
    conn = _connect(read_timeout=_BULK_READ_TIMEOUT)
    try:
        with conn.cursor(pymysql.cursors.SSCursor) as cur:
            cur.execute(
                f'SELECT {_COLUMNS} FROM `cuber_profiles` WHERE tier <= %s',
                (min(max_tier, RESIDENT_TIERS),),
            )
            profiles = _shape(cur)
    finally:
        conn.close()
    logger.info('loaded game profiles depth=%d rows=%d wall_ms=%.1f',
                max_tier, len(profiles), (time.perf_counter() - started) * 1000)
    return profiles


def _read_directory() -> _CompactNameIndex:
    """Stream the public directory into memory without a second row copy."""
    conn = _connect(read_timeout=_BULK_READ_TIMEOUT)
    try:
        with conn.cursor(pymysql.cursors.SSCursor) as cur:
            cur.execute('SELECT wca_id, name, fame FROM `cuber_profiles`')
            return _CompactNameIndex(cur)
    finally:
        conn.close()


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
    """Profile rows down to `max_tier`, refreshed when the generation changes.

    Failed or empty reads retain any previous snapshot and retry after a short
    cooldown. With no snapshot, return an empty list to signal unavailability.
    """
    global _cache_generation, _ids_expires_at
    depth = min(max_tier, RESIDENT_TIERS)

    now = time.monotonic()
    if not force_refresh and now < _cache_expires_at.get(depth, 0):
        return _cache_rows.get(depth, [])

    with _cache_lock:
        # Another thread may have refreshed while we waited for the lock.
        now = time.monotonic()
        if not force_refresh:
            rows = _fresh_covering_rows(depth, now)
            if rows is not None:
                return rows

        try:
            generation = _read_generation()
        except Exception as e:
            logger.warning('Could not check cuber_profiles generation: %s', e)
            if depth in _cache_rows and not force_refresh:
                _cache_expires_at[depth] = now + _CACHE_RETRY_SECONDS
                return _cache_rows[depth]
            generation = None

        if (not force_refresh and generation is not None
                and generation == _cache_generation and depth in _cache_rows
                and now - _cache_loaded_at.get(depth, 0) < _FULL_REFRESH_SECONDS):
            _cache_expires_at[depth] = now + _CACHE_TTL_SECONDS
            return _cache_rows[depth]

        try:
            rows = _read_all(depth)
        except Exception as e:
            logger.warning('Could not read cuber_profiles: %s', e)
            rows = None

        if not rows:
            if rows is not None:
                logger.warning('cuber_profiles returned no rows for tier <= %s', depth)
            _cache_expires_at[depth] = time.monotonic() + _CACHE_RETRY_SECONDS
            # Do not store []: an empty deeper pool would also hide valid rows
            # from name search. Keep existing engines for a retained snapshot.
            return _cache_rows.get(depth, [])

        if generation != _cache_generation or depth in _cache_rows:
            if generation != _cache_generation:
                _ids_expires_at = 0.0
            _cache_rows.clear()
            _cache_expires_at.clear()
            _cache_loaded_at.clear()
            _cache_engines.clear()
            _tail_cache.clear()
            _search_cache.clear()
        _cache_generation = generation
        _cache_rows[depth] = rows
        _cache_expires_at[depth] = time.monotonic() + _CACHE_TTL_SECONDS
        _cache_loaded_at[depth] = time.monotonic()
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
        return []
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
        wca_id = random.choice(_profile_ids())
        profile = find_by_wca_id(wca_id)
        if profile is not None:
            return profile
    except Exception as e:
        logger.warning('random profile lookup failed: %s', e)
    engine = get_engine('hard')
    return engine.pick_secret() if engine else None


def _profile_ids() -> list[str]:
    """Cache a compact ID/name directory for Everyone's random draw and search."""
    global _all_ids, _name_index, _ids_generation, _ids_expires_at
    global _ids_loaded_at
    now = time.monotonic()
    if now < _ids_expires_at:
        return _ids_or_unavailable()

    with _ids_lock:
        now = time.monotonic()
        if now < _ids_expires_at:
            return _ids_or_unavailable()
        try:
            generation = _read_generation()
        except Exception as e:
            logger.warning('Could not check profile ID generation: %s', e)
            if _all_ids:
                _ids_expires_at = now + _CACHE_RETRY_SECONDS
                return _all_ids
            generation = None

        if (_all_ids and generation is not None and generation == _ids_generation
                and now - _ids_loaded_at < _FULL_REFRESH_SECONDS):
            _ids_expires_at = now + _CACHE_TTL_SECONDS
            return _all_ids

        try:
            started = time.perf_counter()
            directory = _read_directory()
            ids = directory.ids
            if not ids:
                raise ValueError('cuber_profiles has no competitors')
        except Exception:
            # Back off even with nothing cached. Search is rate-limit exempt,
            # so otherwise every keystroke would start another full-table scan
            # while the database is slow or throttled.
            _ids_expires_at = now + _CACHE_RETRY_SECONDS
            if _all_ids:
                return _all_ids
            raise

        if generation != _ids_generation:
            _tail_cache.clear()
            _search_cache.clear()
        _all_ids = ids
        _name_index = directory
        _ids_generation = generation
        _ids_expires_at = now + _CACHE_TTL_SECONDS
        _ids_loaded_at = now
        logger.info('loaded game name directory rows=%d wall_ms=%.1f',
                    len(ids), (time.perf_counter() - started) * 1000)
        return ids


def _ids_or_unavailable() -> list[str]:
    """The cached directory, or an error while a failed cold load cools down."""
    if _all_ids:
        return _all_ids
    raise RuntimeError('profile directory unavailable; retrying shortly')


def search_by_name(query: str, limit: int = 8, include_all: bool = False) -> list[dict]:
    """Substring name search for the guess autocomplete.

    The resident pool is searched in process — it is small, this is typed per
    keystroke, and a round trip per keystroke would be slow and pointless.

    `include_all` additionally searches the compact name directory and fetches
    matching rows by indexed ID. The hidden competitor may be one of the ~245k
    who are not resident, so those names must remain searchable.
    """
    q = (query or '').strip().casefold()
    if len(q) < 2:
        return []

    cached = _cached_rows()
    if not cached and not include_all:
        cached = load_profiles()
    hits = [r for r in cached if q in r['name'].casefold()]
    if include_all and len(hits) < limit:
        seen = {r['wca_id'] for r in hits}
        hits += [r for r in _search_db(q, limit) if r['wca_id'] not in seen]

    hits.sort(key=lambda r: (not r['name'].casefold().startswith(q), -r['fame']))
    return hits[:limit]


def _search_db(q: str, limit: int) -> list[dict]:
    """Search a local name index, then fetch only the matching public fields."""
    key = (q, limit)
    now = time.monotonic()
    cached = _search_cache.get(key)
    if cached and now < cached[0]:
        return cached[1]

    try:
        _profile_ids()
        ids = _name_index.find(q, limit)
        if not ids:
            hits = []
        else:
            placeholders = ', '.join(['%s'] * len(ids))
            rows = _query(
                'SELECT wca_id, name, country_id, continent_id, fame '
                f'FROM `cuber_profiles` WHERE wca_id IN ({placeholders})',
                tuple(ids),
            )
            by_id = {wca_id: dict(wca_id=wca_id, name=name,
                                  country_id=country, continent_id=continent,
                                  fame=int(fame))
                     for wca_id, name, country, continent, fame in rows}
            hits = [by_id[wca_id] for wca_id in ids if wca_id in by_id]
        if len(_search_cache) >= _SEARCH_CACHE_SIZE:
            _search_cache.clear()
        _search_cache[key] = (now + _SEARCH_CACHE_SECONDS, hits)
        return hits
    except Exception as e:
        logger.warning('name search failed: %s', e)
        return []
