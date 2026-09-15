"""Build the `cuber_profiles` feature matrix that Guess the Cuber plays over.

Runs weekly, right after `update_database.py` reloads the WCA export. Everything
the game knows about a cuber is computed here and stored as one JSON blob per
person, so a turn of the game never touches `results` (6.7M rows) — it reads a
few thousand precomputed rows and does arithmetic.

Eligibility, per the feature spec: anyone who has held a continental record or
better, or who is currently in the world top 100 for any event.

Every attribute is computed with a handful of GROUP BY aggregates over the whole
pool at once. Resist the urge to add a per-person query — the pool is a few
thousand people and a per-person round trip to TiDB would turn a two-minute job
into an hour.

Usage:
    python scripts/build_cuber_profiles.py            # build and write
    python scripts/build_cuber_profiles.py --dry-run  # report only, no writes
"""
import argparse
import json
import logging
import math
import ssl
import sys
from datetime import datetime
from pathlib import Path

import certifi
import pymysql
from dotenv import load_dotenv

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

load_dotenv()

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL  # noqa: E402
from services.game.attributes import (ATTRIBUTES, EVENT_GROUPS,  # noqa: E402
                                      event_weight, group_of)

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('build_cuber_profiles')

# Continental record codes. 'WR' is handled separately; 'NR' is national and
# does not by itself make someone eligible.
CONTINENTAL = ('ER', 'NAR', 'SAR', 'AsR', 'AfR', 'OcR')

POOL_TABLE = '_gc_pool'

CREATE_PROFILES_SQL = """
CREATE TABLE IF NOT EXISTS `cuber_profiles` (
  `wca_id`       VARCHAR(10)  NOT NULL PRIMARY KEY,
  `name`         VARCHAR(120) NOT NULL,
  `country_id`   VARCHAR(50),
  `continent_id` VARCHAR(50),
  `fame`         INT     NOT NULL,
  `tier`         TINYINT NOT NULL,
  `attrs`        JSON    NOT NULL,
  `updated_at`   TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  KEY `idx_tier` (`tier`)
)
"""

# Tier boundaries by fame rank, within the elite (record / top-100) group.
# Play pools are cumulative: Easy is tier 1, Normal tiers 1-2, Hard tiers 1-3.
TIER_1_SIZE = 300
TIER_2_SIZE = 1000

# Tier 4 is the "Everyone" pool the app loads into memory: every remaining
# competitor with at least this many competitions.
#
# Tier 5 is everybody else, and the app never loads it. Those profiles exist so
# "you guess mine" can hide *any* WCA competitor — that mode holds one secret
# and answers questions about it, so it needs a single indexed row, not the
# pool. The guessing engine does need every candidate resident, and below this
# threshold the data runs out anyway: 155k competitors have been to exactly one
# competition and 50k to two, and their profiles are identical in nearly every
# attribute the game can ask about.
MIN_EVERYONE_COMPS = 5
TIER_EVERYONE = 4
TIER_LONG_TAIL = 5

# How fast an achievement fades from public memory, in years. A record set this
# season counts full; one from five years ago counts 0.6; the floor keeps
# historic achievements meaningful rather than erasing them — a 2008 world
# record still says something about the person who set it.
RECENCY_HALF_LIFE = 5.0
RECENCY_FLOOR = 0.20


def recency_weight(year: int, current_year: int) -> float:
    """Decay an achievement by how long ago it happened."""
    if not year:
        return RECENCY_FLOOR
    age = max(0, current_year - int(year))
    return RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * 0.5 ** (age / RECENCY_HALF_LIFE)


def get_connection() -> pymysql.Connection:
    kwargs = {}
    if DB_SSL:
        # certifi rather than the OS trust store, matching WCAService and
        # update_database.py — some hosts carry a stale root that breaks
        # verification of TiDB Cloud's valid cert.
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    return pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset='utf8mb4', autocommit=True, **kwargs,
    )


def _fetch(cur, sql, args=None):
    cur.execute(sql, args)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Step 1 — the eligible pool
# ---------------------------------------------------------------------------

def build_pool(cur) -> set[str]:
    """Stage the profile pool and return the record/top-100 subset.

    Two groups go in. The *elite* — a continental record or better, or a current
    world top 100 — are the Easy/Normal/Hard tiers, as before. On top of those
    come everyone with at least MIN_EVERYONE_COMPS competitions, which is what
    the "Everyone" tier plays over.

    That threshold is where the data runs out, not an arbitrary line: 155k of
    WCA's 297k competitors have been to exactly one competition and 50k to two,
    and those profiles are identical in almost every attribute the game asks
    about. Including them would mean a pool the search cannot separate.
    """
    cur.execute(f'DROP TABLE IF EXISTS `{POOL_TABLE}`')
    cur.execute(
        f'CREATE TABLE `{POOL_TABLE}` (person_id VARCHAR(10) NOT NULL PRIMARY KEY)'
    )

    placeholders = ', '.join(['%s'] * len(CONTINENTAL))
    # One scan of `results` covering both record columns. There is no index on
    # the record columns, so this is a full scan of 6.8M rows — a minute or so,
    # and the reason this script is a weekly job rather than a request path.
    log.info('Scanning results for continental-record-or-better holders...')
    cur.execute(
        f'INSERT IGNORE INTO `{POOL_TABLE}` (person_id) '
        f'SELECT DISTINCT person_id FROM `results` '
        f'WHERE regional_single_record IN (%s, {placeholders}) '
        f'   OR regional_average_record IN (%s, {placeholders})',
        ('WR', *CONTINENTAL, 'WR', *CONTINENTAL),
    )
    log.info('  record holders: %d', cur.rowcount)

    for table in ('ranks_single', 'ranks_average'):
        cur.execute(
            f'INSERT IGNORE INTO `{POOL_TABLE}` (person_id) '
            f'SELECT DISTINCT person_id FROM `{table}` WHERE world_rank <= 100'
        )
        log.info('  + top-100 from %s: %d new', table, cur.rowcount)

    elite = {r[0] for r in _fetch(cur, f'SELECT person_id FROM `{POOL_TABLE}`')}
    log.info('Elite pool (record / top-100): %d cubers', len(elite))

    log.info('Adding every remaining competitor...')
    cur.execute(
        f'INSERT IGNORE INTO `{POOL_TABLE}` (person_id) '
        f'SELECT DISTINCT person_id FROM `results`'
    )
    total = _fetch(cur, f'SELECT COUNT(*) FROM `{POOL_TABLE}`')[0][0]
    log.info('Full profile pool: %d cubers', total)
    return elite


# ---------------------------------------------------------------------------
# Step 2 — attribute families, one aggregate query each
# ---------------------------------------------------------------------------

def fetch_identity(cur) -> dict[str, dict]:
    """Name, country, continent, gender. sub_id=1 is the person's current identity."""
    rows = _fetch(cur, f"""
        SELECT p.wca_id, p.name, p.country_id, p.gender, c.continent_id
          FROM `persons` p
          JOIN `{POOL_TABLE}` g ON p.wca_id = g.person_id
          LEFT JOIN `countries` c ON p.country_id = c.id
         WHERE p.sub_id = 1
    """)
    return {
        wca_id: {
            'name': name,
            'country_id': country,
            'continent_id': continent,
            'gender': (gender or '').strip().lower() or None,
        }
        for wca_id, name, country, gender, continent in rows
    }


def fetch_records(cur) -> dict[str, dict]:
    """World / continental / national record counts.

    Every record column is COALESCEd before comparison. The export writes NULL
    rather than '' in the oldest rows, and `NULL IN (...)` is NULL, not false —
    so `SUM(a) + SUM(b)` silently collapses to NULL for anyone whose results
    are all-NULL in one column. That zeroed out the Asian record of the 1982
    competitors and dropped them out of their own eligibility criterion.
    """
    cont = ', '.join(['%s'] * len(CONTINENTAL))
    rows = _fetch(cur, f"""
        SELECT r.person_id,
               SUM(COALESCE(r.regional_single_record,  '') = 'WR'),
               SUM(COALESCE(r.regional_average_record, '') = 'WR'),
               SUM(COALESCE(r.regional_single_record,  '') IN ({cont}))
             + SUM(COALESCE(r.regional_average_record, '') IN ({cont})),
               SUM(COALESCE(r.regional_single_record,  '') = 'NR')
             + SUM(COALESCE(r.regional_average_record, '') = 'NR')
          FROM `results` r
          JOIN `{POOL_TABLE}` g ON r.person_id = g.person_id
         GROUP BY r.person_id
    """, (*CONTINENTAL, *CONTINENTAL))
    return {
        pid: {
            'wr_single_count': int(wr_s or 0),
            'wr_average_count': int(wr_a or 0),
            'cr_count': int(cr or 0),
            'nr_count': int(nr or 0),
        }
        for pid, wr_s, wr_a, cr, nr in rows
    }


def fetch_weighted_records(cur, current_year: int) -> dict[str, dict]:
    """Records scored by which event they were set in and how long ago.

    Separate from `fetch_records`, which keeps the plain counts the *game* asks
    about ("have they set at least 2 world records?"). Those must stay raw — a
    player answering that question is not thinking about event popularity. Only
    the fame score, which decides difficulty tiers, is weighted.

    Only WR and CR rows are read here, so this is ~5k rows rather than the full
    record set, and weighting them in Python keeps the tiers legible and
    tunable instead of buried in a SQL CASE ladder.
    """
    cont = ', '.join(['%s'] * len(CONTINENTAL))
    rows = _fetch(cur, f"""
        SELECT r.person_id, r.event_id, c.year,
               COALESCE(r.regional_single_record,  ''),
               COALESCE(r.regional_average_record, '')
          FROM `results` r
          JOIN `{POOL_TABLE}` g ON r.person_id = g.person_id
          JOIN `competitions` c ON r.competition_id = c.id
         WHERE COALESCE(r.regional_single_record,  '') IN (%s, {cont})
            OR COALESCE(r.regional_average_record, '') IN (%s, {cont})
    """, ('WR', *CONTINENTAL, 'WR', *CONTINENTAL))

    out: dict[str, dict] = {}
    for person_id, event_id, year, single, average in rows:
        weight = event_weight(event_id) * recency_weight(year, current_year)
        acc = out.setdefault(person_id, {'_wr_weighted': 0.0, '_cr_weighted': 0.0})
        # A single result can be both a single and an average record, and each
        # is its own achievement.
        for code in (single, average):
            if code == 'WR':
                acc['_wr_weighted'] += weight
            elif code in CONTINENTAL:
                acc['_cr_weighted'] += weight
    return out


def fetch_career(cur) -> dict[str, dict]:
    """Volume and era: competitions, events, years, countries travelled to."""
    rows = _fetch(cur, f"""
        SELECT r.person_id,
               COUNT(*),
               COUNT(DISTINCT r.competition_id),
               COUNT(DISTINCT r.event_id),
               MIN(c.year), MAX(c.year),
               COUNT(DISTINCT c.country_id)
          FROM `results` r
          JOIN `{POOL_TABLE}` g ON r.person_id = g.person_id
          JOIN `competitions` c ON r.competition_id = c.id
         GROUP BY r.person_id
    """)
    return {
        pid: {
            'total_results': int(total or 0),
            'comp_count': int(comps or 0),
            'events_competed': int(events or 0),
            'first_year': int(first or 0),
            'last_year': int(last or 0),
            'years_active': max(0, int(last or 0) - int(first or 0)) + 1,
            'countries_competed_in': int(countries or 0),
        }
        for pid, total, comps, events, first, last, countries in rows
    }


def fetch_rankings(cur) -> dict[str, dict]:
    """Current world ranks, collapsed to the facts the game asks about.

    Single and average ranks are unioned and reduced to each person's best rank
    per event, because "are they top 100 in 3x3" shouldn't depend on which of
    the two the player had in mind.

    Continent and country ranks come along for the "currently holds a record"
    attributes. The record hierarchy needs no special handling here: rank 1 in
    the world is necessarily rank 1 in your continent and your country, so
    `continent_rank = 1` already means "continental record or better".
    """
    rows = _fetch(cur, f"""
        SELECT u.person_id, u.event_id,
               MIN(u.world_rank), MIN(u.continent_rank), MIN(u.country_rank)
          FROM (
                SELECT person_id, event_id, world_rank, continent_rank, country_rank
                  FROM `ranks_single`
                 UNION ALL
                SELECT person_id, event_id, world_rank, continent_rank, country_rank
                  FROM `ranks_average`
               ) u
          JOIN `{POOL_TABLE}` g ON u.person_id = g.person_id
         WHERE u.world_rank > 0
         GROUP BY u.person_id, u.event_id
    """)

    by_person: dict[str, list[tuple]] = {}
    for pid, event_id, world, continent, country in rows:
        by_person.setdefault(pid, []).append(
            (event_id, int(world), int(continent or 0), int(country or 0))
        )

    out = {}
    for pid, ranks in by_person.items():
        best_event, best_rank = min(((e, w) for e, w, _c, _n in ranks),
                                    key=lambda er: er[1])
        group = group_of(best_event)
        out[pid] = {
            'main_event': best_event,
            'main_event_group': group,
            'top100_events': sorted(e for e, w, _c, _n in ranks if w <= 100),
            'top100_event_count': sum(1 for _e, w, _c, _n in ranks if w <= 100),
            'top10_any': best_rank <= 10,
            'currently_wr': best_rank == 1,
            # Fame input: how much attention the event they lead actually gets.
            '_current_wr_weight': max(
                (event_weight(e) for e, w, _c, _n in ranks if w == 1), default=0.0),
            'currently_cr_or_better': any(c == 1 for _e, _w, c, _n in ranks),
            'currently_nr_or_better': any(n == 1 for _e, _w, _c, n in ranks),
            'is_bld_specialist': best_event in EVENT_GROUPS['blind_events'],
            'is_bigcube_specialist': best_event in EVENT_GROUPS['big_cubes'],
            'is_sideevent_specialist': best_event in EVENT_GROUPS['side_events'],
            'is_fmc_specialist': best_event == '333fm',
            'is_oh_specialist': best_event == '333oh',
        }
    return out


def fetch_personal_bests(cur) -> dict[str, dict]:
    """3x3 milestones. `best` is centiseconds, so sub-10 is < 1000."""
    singles = dict(_fetch(cur, f"""
        SELECT rs.person_id, rs.best FROM `ranks_single` rs
          JOIN `{POOL_TABLE}` g ON rs.person_id = g.person_id
         WHERE rs.event_id = '333' AND rs.best > 0
    """))
    averages = dict(_fetch(cur, f"""
        SELECT ra.person_id, ra.best FROM `ranks_average` ra
          JOIN `{POOL_TABLE}` g ON ra.person_id = g.person_id
         WHERE ra.event_id = '333' AND ra.best > 0
    """))
    ids = set(singles) | set(averages)
    return {
        pid: {
            'sub10_333_single': 0 < singles.get(pid, 0) < 1000,
            'sub6_333_single': 0 < singles.get(pid, 0) < 600,
            'sub10_333_average': 0 < averages.get(pid, 0) < 1000,
        }
        for pid in ids
    }


def fetch_worlds(cur, current_year: int) -> dict[str, dict]:
    """World Championship appearances and podiums.

    World Championship competition ids are WC<year> (WC2003 ... WC2025), which
    is more dependable here than the `championships` table — that table is
    loaded by the refresh job but nothing else in the app reads it, so its shape
    is unverified.
    """
    rows = _fetch(cur, f"""
        SELECT r.person_id, r.event_id, c.year, r.round_type_id, r.pos, r.best
          FROM `results` r
          JOIN `{POOL_TABLE}` g ON r.person_id = g.person_id
          JOIN `competitions` c ON r.competition_id = c.id
         WHERE r.competition_id REGEXP '^WC[0-9]{{4}}$'
    """)

    out: dict[str, dict] = {}
    for person_id, event_id, year, round_type, pos, best in rows:
        acc = out.setdefault(person_id, {
            'competed_at_worlds': True,
            'has_worlds_podium': False,
            '_worlds_podiums': 0,
            '_podium_weighted': 0.0,   # fame inputs, stripped in assemble()
        })
        if round_type in ('f', 'c') and (pos or 0) <= 3 and (pos or 0) > 0 \
                and (best or 0) > 0:
            acc['has_worlds_podium'] = True
            acc['_worlds_podiums'] += 1
            # A 3x3 Worlds podium is the single most recognizable result in the
            # sport; a 5BLD one is known to the people who follow 5BLD.
            acc['_podium_weighted'] += (event_weight(event_id)
                                        * recency_weight(year, current_year))
    return out


# ---------------------------------------------------------------------------
# Step 3 — assemble, score, tier
# ---------------------------------------------------------------------------

def assemble(sources: dict[str, dict[str, dict]], total_events: int,
             elite: set[str]) -> list[dict]:
    """Merge the attribute families into one row per cuber."""
    identity = sources['identity']
    current_year = datetime.now().year
    profiles = []

    for wca_id, ident in identity.items():
        attrs: dict = {}
        for family in ('records', 'career', 'rankings', 'bests', 'worlds',
                       'weighted'):
            attrs.update(sources[family].get(wca_id, {}))

        attrs['gender'] = ident['gender']
        attrs['country_id'] = ident['country_id']
        attrs['continent_id'] = ident['continent_id']

        # Derived facts that need more than one family.
        #
        # Records are CUMULATIVE, because WR > CR > NR. Someone who has set a
        # world record has by definition cleared the continental and national
        # bars too, so "continental record or better" must be true for them even
        # if they never set a record that was *recorded* as continental. The FMC
        # world record holder has exactly that shape — one WR, zero CRs, zero
        # NRs — and answering "no" to "continental record or better" is the bug
        # this arithmetic exists to prevent.
        wr = attrs.get('wr_single_count', 0) + attrs.get('wr_average_count', 0)
        cr = attrs.get('cr_count', 0)
        nr = attrs.get('nr_count', 0)
        attrs['has_wr'] = wr > 0
        attrs['has_cr_or_better'] = (wr + cr) > 0
        attrs['has_nr_or_better'] = (wr + cr + nr) > 0

        attrs['is_active'] = attrs.get('last_year', 0) >= current_year - 1
        attrs['competed_all_events'] = attrs.get('events_competed', 0) >= total_events

        # Fame inputs, not attributes — popped before the schema check below so
        # they never reach the game or the stored blob.
        fame_inputs = {
            'worlds_podiums': attrs.pop('_worlds_podiums', 0),
            'podium_weighted': attrs.pop('_podium_weighted', 0.0),
            'wr_weighted': attrs.pop('_wr_weighted', 0.0),
            'cr_weighted': attrs.pop('_cr_weighted', 0.0),
            'current_wr_weight': attrs.pop('_current_wr_weight', 0.0),
        }

        # Defaults for anyone a family had no row for — a cuber with no ranks
        # row, say. Without this the engine would read missing as False and the
        # distinction between "no" and "unknown" would be lost silently.
        for attr in ATTRIBUTES:
            if attr.key in attrs:
                continue
            attrs[attr.key] = {
                'bool': False, 'numeric': 0, 'multi': [],
            }.get(attr.kind, None)

        profiles.append({
            'wca_id': wca_id,
            'name': ident['name'],
            'country_id': ident['country_id'],
            'continent_id': ident['continent_id'],
            'fame': fame_score(attrs, fame_inputs, current_year),
            'is_elite': wca_id in elite,
            'attrs': attrs,
        })

    return profiles


def fame_score(attrs: dict, fame_inputs: dict, current_year: int) -> int:
    """A recognizability proxy, used only to slice the pool into difficulty tiers.

    Weighted toward what a fan would actually recognize someone for. Three
    things shape it beyond raw achievement counts:

    *Which event.* A 3x3 world record makes the rounds; a 5BLD one is known
    inside its own community. Every record, podium and ranking is scaled by
    EVENT_TIER_WEIGHT, so counts alone no longer decide the order.

    *How recently.* Records and podiums decay on a five-year half-life down to a
    floor, because a title from last season carries far more name recognition
    than one from 2009 — without erasing the 2009 one entirely.

    *How much.* The log term keeps a 20-year veteran with thousands of results
    from outranking a current world record holder on volume alone.

    `attrs` supplies the plain counts the game asks about; `fame_inputs` carries
    the event- and recency-weighted sums, which are deliberately kept out of
    `attrs` so they never become questions.
    """
    score = (
        6.0 * fame_inputs.get('wr_weighted', 0.0)
        + 3.0 * fame_inputs.get('cr_weighted', 0.0)
        + 2.0 * fame_inputs.get('podium_weighted', 0.0)
        + 1.5 * math.log10(attrs.get('total_results', 0) + 1)
        # Being currently ranked is weighted by event too: top 100 in 3x3 is a
        # different achievement from top 100 in 5BLD.
        + 1.0 * sum(event_weight(e) for e in (attrs.get('top100_events') or ()))
    )
    last_year = attrs.get('last_year', 0)
    if last_year >= current_year - 1:
        score += 5.0
    elif last_year >= current_year - 3:
        score += 2.0
    # Holding a record right now, scaled by how much attention that event gets.
    if attrs.get('currently_wr'):
        score += 8.0 * fame_inputs.get('current_wr_weight', 1.0)
    return int(round(score * 10))


def assign_tiers(profiles: list[dict]) -> None:
    """Rank by fame and slice into difficulty tiers, in place.

    Only the elite are ranked against each other; everyone else lands in the
    Everyone tier regardless of fame. Keeping the two groups separate means
    adding 50k ordinary competitors can never push a world record holder out of
    the Easy tier.
    """
    profiles.sort(key=lambda p: (p['is_elite'], p['fame']), reverse=True)
    rank = 0
    for profile in profiles:
        if not profile.pop('is_elite'):
            profile['tier'] = (
                TIER_EVERYONE
                if profile['attrs'].get('comp_count', 0) >= MIN_EVERYONE_COMPS
                else TIER_LONG_TAIL
            )
            continue
        if rank < TIER_1_SIZE:
            profile['tier'] = 1
        elif rank < TIER_2_SIZE:
            profile['tier'] = 2
        else:
            profile['tier'] = 3
        rank += 1


# ---------------------------------------------------------------------------
# Step 4 — write, and report
# ---------------------------------------------------------------------------

def write_profiles(conn, profiles: list[dict]) -> None:
    cur = conn.cursor()
    cur.execute(CREATE_PROFILES_SQL)
    cur.execute('DELETE FROM `cuber_profiles`')

    sql = ('INSERT INTO `cuber_profiles` '
           '(wca_id, name, country_id, continent_id, fame, tier, attrs) '
           'VALUES (%s, %s, %s, %s, %s, %s, %s)')
    batch, size = [], 500
    for p in profiles:
        batch.append((p['wca_id'], p['name'], p['country_id'], p['continent_id'],
                      p['fame'], p['tier'], json.dumps(p['attrs'])))
        if len(batch) >= size:
            cur.executemany(sql, batch)
            batch.clear()
    if batch:
        cur.executemany(sql, batch)
    cur.close()
    log.info('Wrote %d profiles', len(profiles))


def report(profiles: list[dict]) -> None:
    """Print pool size, tier split, and per-attribute balance.

    The balance column is the thing to read: a boolean attribute that is true
    for under 5% or over 95% of the pool is a near-useless question — it almost
    never splits the candidates — and should be dropped or rebucketed.
    """
    labels = {1: 'Easy', 2: 'Normal', 3: 'Hard',
              4: 'Everyone', 5: 'tail (DB only)'}
    print('\n' + '=' * 62)
    print(f'  Profiles written: {len(profiles):,}')
    for tier in sorted(labels):
        count = sum(1 for p in profiles if p['tier'] == tier)
        print(f'    tier {tier} ({labels[tier]:<14}) {count:>8,}')

    # Balance is measured over the pool the engine actually plays — tiers 1-4.
    # Including the long tail would swamp every statistic with 245k
    # near-identical one-competition profiles and make healthy attributes look
    # broken.
    profiles = [p for p in profiles if p['tier'] <= 4]
    n = len(profiles)
    print(f'  Resident pool (engine plays these): {n:,}')
    print('=' * 62)
    print(f'  {"attribute":<26} {"kind":<12} balance')
    print('-' * 62)

    for attr in ATTRIBUTES:
        values = [p['attrs'].get(attr.key) for p in profiles]
        if attr.kind == 'bool':
            share = sum(1 for v in values if v) / n if n else 0
            flag = '  <-- weak split' if not 0.05 <= share <= 0.95 else ''
            print(f'  {attr.key:<26} {"bool":<12} {share:>6.1%}{flag}')
        elif attr.kind == 'numeric':
            parts = []
            for t in attr.thresholds:
                share = sum(1 for v in values if (v or 0) >= t) / n if n else 0
                parts.append(f'>={t}:{share:.0%}')
            print(f'  {attr.key:<26} {"numeric":<12} {" ".join(parts)}')
        elif attr.kind == 'multi':
            held = sum(1 for v in values if v) / n if n else 0
            distinct = len({x for v in values for x in (v or ())})
            parts = [f'any:{held:.0%}', f'{distinct} values']
            for group, members in EVENT_GROUPS.items():
                share = sum(
                    1 for v in values if set(v or ()) & set(members)
                ) / n if n else 0
                parts.append(f'{group}:{share:.0%}')
            print(f'  {attr.key:<26} {"multi":<12} {" ".join(parts)}')
        else:
            distinct = len({v for v in values if v is not None})
            top = max(
                (sum(1 for v in values if v == x) / n
                 for x in {v for v in values if v is not None}),
                default=0,
            )
            print(f'  {attr.key:<26} {"categorical":<12} '
                  f'{distinct} values, largest {top:.0%}')
    print('=' * 62 + '\n')


def run(dry_run: bool = False) -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        elite = build_pool(cur)

        total_events = _fetch(
            cur, 'SELECT COUNT(*) FROM `events` WHERE `rank` < 900'
        )[0][0]

        current_year = datetime.now().year
        log.info('Computing attribute families...')
        sources = {
            'identity': fetch_identity(cur),
            'records': fetch_records(cur),
            'career': fetch_career(cur),
            'rankings': fetch_rankings(cur),
            'bests': fetch_personal_bests(cur),
            'worlds': fetch_worlds(cur, current_year),
            'weighted': fetch_weighted_records(cur, current_year),
        }
        for name, data in sources.items():
            log.info('  %-10s %d rows', name, len(data))

        profiles = assemble(sources, total_events, elite)
        assign_tiers(profiles)
        report(profiles)

        if dry_run:
            log.info('--dry-run: nothing written')
        else:
            write_profiles(conn, profiles)

        cur.execute(f'DROP TABLE IF EXISTS `{POOL_TABLE}`')
        cur.close()
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the cuber_profiles feature matrix for Guess the Cuber'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report, but write nothing')
    args = parser.parse_args()
    run(dry_run=args.dry_run)
