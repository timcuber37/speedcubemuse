"""Build the `cuber_profiles` feature matrix that Guess the Cuber plays over.

Runs on every scheduled refresh, right after `update_database.py` reloads the
WCA export. Everything the game knows about a cuber is computed here and stored
as one JSON blob per person, so a turn of the game never touches `results` (7M
rows). It reads precomputed rows and does arithmetic.

The attributes are computed from the export files themselves, not from the
tables loaded into TiDB: one streaming pass over `results`, one over each rank
table, and small lookups for people, competitions, countries and events. That
takes a minute or two on the machine running the job and needs a few GB of
memory. The database only receives the finished ~25 MB of profiles. The note
that opens Steps 1 and 2 explains why this moved out of TiDB.

Usage:
    python scripts/build_cuber_profiles.py                    # download, build, write
    python scripts/build_cuber_profiles.py --export tsv.zip   # build from a local export
    python scripts/build_cuber_profiles.py --dry-run          # report only, no DB at all
"""
import argparse
import json
import logging
import math
import re
import ssl
import sys
from datetime import datetime
from pathlib import Path

import certifi
import pymysql
from dotenv import load_dotenv

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

load_dotenv()

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL  # noqa: E402
from services import site_meta  # noqa: E402
from services.game.attributes import (ATTRIBUTES, CURRENT_EVENTS,  # noqa: E402
                                      EVENT_GROUPS, event_weight, group_of)
import wca_export  # noqa: E402
from wca_export import ExportReader  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
log = logging.getLogger('build_cuber_profiles')

# Continental record codes. 'WR' is handled separately; 'NR' is national and
# does not by itself make someone eligible.
CONTINENTAL = ('ER', 'NAR', 'SAR', 'AsR', 'AfR', 'OcR')

# The staging table the TiDB version of this build used. A run that failed
# midway left it behind with ~300k rows, so writing now drops it.
_LEGACY_POOL_TABLE = '_gc_pool'

# The live table the app reads, and the two names the swap in write_profiles()
# moves tables through.
PROFILES_TABLE = 'cuber_profiles'
_STAGING_TABLE = 'cuber_profiles_next'
_REPLACED_TABLE = 'cuber_profiles_old'

CREATE_PROFILES_SQL = """
CREATE TABLE IF NOT EXISTS `{table}` (
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

# What a current world ranking is worth, by how high it is. Bands are
# exclusive — an event counts once, at the best band it reaches, so a top-25
# finish scores 3.0 rather than 3.0 + 1.0 + 0.3 stacked.
#
# The 100 band keeps its original weight of 1.0, so this adds a reward for
# elite rankings and a small one for depth without re-scaling what was already
# there. These are scoring bands only: pool eligibility still needs a top 100
# or a record, since admitting every top-500 would take the pool from ~1,800
# to tens of thousands.
RANK_BANDS = ((25, 3.0), (100, 1.0), (500, 0.3))


def rank_band_weight(world_rank: int) -> float:
    """Score multiplier for one event's current world ranking."""
    for threshold, weight in RANK_BANDS:
        if world_rank <= threshold:
            return weight
    return 0.0


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


# ---------------------------------------------------------------------------
# Steps 1 and 2 — the pool and its attributes, read straight from the export
# ---------------------------------------------------------------------------
#
# These used to be GROUP BY queries inside TiDB, staged through a `_gc_pool`
# table. That aggregation was the most expensive thing the refresh asked of the
# database, and on 2026-11-01 one of its queries was cancelled for exceeding the
# server's memory limit, which left the game serving a month-old matrix. The
# job's runner already holds the export, so the same rules run here instead and
# TiDB only receives the finished rows.
#
# Each function names the query it replaced, and the semantics carry over
# exactly, joins included: a result whose competition is missing from
# `competitions` drops out of every family that joined to it.

_RECORD_CODES = frozenset(('WR', *CONTINENTAL))
_WORLDS_ID = re.compile(r'WC[0-9]{4}')


def _int(value: str | None) -> int:
    """A numeric field. NULL or blank reads as 0, as the SQL's `x or 0` did."""
    return int(value) if value else 0


def _min(a: int | None, b: int | None) -> int | None:
    """MIN() over nullable values: NULLs are ignored unless that's all there is."""
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def read_current_events(export: ExportReader) -> set[str]:
    """The official events the export hasn't retired.

    CURRENT_EVENTS decides what counts as official, so an event the WCA has
    added to the export but not made official yet (FTO, at the moment) stays
    out. `rank >= 900` is how the export marks a retired event, so one retired
    in a future export drops out on its own.
    """
    live = {event_id for event_id, rank in export.rows('events', ('id', 'rank'))
            if rank is not None and int(rank) < 900}
    return live & set(CURRENT_EVENTS)


def read_competitions(export: ExportReader) -> dict[str, tuple[int | None, str | None]]:
    """Competition id -> (year, country), the two columns the attributes join on."""
    return {
        comp_id: (int(year) if year else None, country)
        for comp_id, year, country in export.rows(
            'competitions', ('id', 'year', 'country_id'))
    }


def scan_results(export: ExportReader, competitions: dict,
                 current_events: set[str], current_year: int) -> dict:
    """Every family that comes from `results`, in one pass over ~7M rows.

    Replaces build_pool's record-holder scan, fetch_records,
    fetch_weighted_records, fetch_career and fetch_worlds.

    *Records* keep the plain counts the game asks about ("have they set at
    least 2 world records?"). Those must stay raw — a player answering that
    question is not thinking about event popularity. *Weighted* records score
    each WR and CR by event and age, and only feed the fame score that decides
    difficulty tiers. A single result can be both a single and an average
    record, and each is its own achievement.

    The oldest rows carry NULL rather than '' in the record columns. The SQL had
    to COALESCE them, because `NULL IN (...)` is NULL rather than false; that
    once zeroed out the Asian record of the 1982 competitors.

    World Championship competition ids are WC<year> (WC2003 ... WC2025), which
    is more dependable than the `championships` table, whose shape nothing else
    in the app relies on.
    """
    record_holders: set[str] = set()
    records: dict[str, list[int]] = {}
    weighted: dict[str, dict] = {}
    # person -> [results, competitions, events, first year, last year, countries]
    career: dict[str, list] = {}
    worlds: dict[str, dict] = {}

    columns = ('person_id', 'competition_id', 'event_id', 'round_type_id', 'pos',
               'best', 'regional_single_record', 'regional_average_record')
    for pid, comp_id, event_id, round_type, pos, best, single, average in \
            export.rows('results', columns):
        single = single or ''
        average = average or ''

        counts = records.get(pid)
        if counts is None:
            counts = records[pid] = [0, 0, 0, 0]
        counts[0] += single == 'WR'
        counts[1] += average == 'WR'
        counts[2] += (single in CONTINENTAL) + (average in CONTINENTAL)
        counts[3] += (single == 'NR') + (average == 'NR')
        is_record = single in _RECORD_CODES or average in _RECORD_CODES
        if is_record:
            record_holders.add(pid)

        competition = competitions.get(comp_id)
        if competition is None:
            continue   # the remaining families joined `competitions`
        year, country = competition

        acc = career.get(pid)
        if acc is None:
            acc = career[pid] = [0, set(), set(), None, None, set()]
        acc[0] += 1
        acc[1].add(comp_id)
        acc[2].add(event_id)
        acc[3] = _min(acc[3], year)
        if year is not None:
            acc[4] = year if acc[4] is None else max(acc[4], year)
        if country is not None:   # COUNT(DISTINCT ...) skips NULL
            acc[5].add(country)

        if is_record:
            weight = event_weight(event_id) * recency_weight(year, current_year)
            acc = weighted.setdefault(pid, {'_wr_weighted': 0.0, '_cr_weighted': 0.0})
            for code in (single, average):
                if code == 'WR':
                    acc['_wr_weighted'] += weight
                elif code in CONTINENTAL:
                    acc['_cr_weighted'] += weight

        if _WORLDS_ID.fullmatch(comp_id):
            acc = worlds.setdefault(pid, {
                'competed_at_worlds': True,
                'has_worlds_podium': False,
                '_worlds_podiums': 0,
                '_podium_weighted': 0.0,   # fame inputs, stripped in assemble()
            })
            if round_type in ('f', 'c') and 0 < _int(pos) <= 3 and _int(best) > 0:
                acc['has_worlds_podium'] = True
                acc['_worlds_podiums'] += 1
                # A 3x3 Worlds podium is the single most recognizable result in
                # the sport; a 5BLD one is known to the people who follow 5BLD.
                acc['_podium_weighted'] += (event_weight(event_id)
                                            * recency_weight(year, current_year))

    return {
        'people': set(records),
        'record_holders': record_holders,
        'records': {
            pid: {'wr_single_count': wr_s, 'wr_average_count': wr_a,
                  'cr_count': cr, 'nr_count': nr}
            for pid, (wr_s, wr_a, cr, nr) in records.items()
        },
        'weighted': weighted,
        'career': {
            pid: {
                'total_results': total,
                'comp_count': len(comps),
                'events_competed': len(events),
                'first_year': first or 0,
                'last_year': last or 0,
                'years_active': max(0, (last or 0) - (first or 0)) + 1,
                'countries_competed_in': len(countries),
                # Every one of them, not just as many: events_competed also
                # counts retired events, so 17 distinct events can still be
                # missing a current one.
                'competed_all_events': current_events <= events,
            }
            for pid, (total, comps, events, first, last, countries) in career.items()
        },
        'worlds': worlds,
    }


def scan_ranks(export: ExportReader, current_events: set[str]) -> dict:
    """Current standings and 3x3 bests, in one pass over each rank table.

    Replaces build_pool's top-100 queries, the UNION ALL in fetch_rankings and
    fetch_personal_bests. Single and average ranks are combined into each
    person's best rank per event, ignoring NULLs as MIN() did.

    Retired events don't count toward the top 100 here, for the same reason they
    are excluded from the attributes: a frozen top-100 in an event the WCA no
    longer runs is not a current standing.
    """
    top100: set[str] = set()
    ranks: dict[str, dict[str, list]] = {}
    bests: dict[str, dict[str, int]] = {'ranks_single': {}, 'ranks_average': {}}

    columns = ('person_id', 'event_id', 'best', 'world_rank',
               'continent_rank', 'country_rank')
    for table in ('ranks_single', 'ranks_average'):
        found = bests[table]
        for pid, event_id, best, world, continent, country in export.rows(table, columns):
            if event_id == '333' and _int(best) > 0:
                found[pid] = _int(best)

            world = _int(world)
            if world <= 0:
                continue
            if world <= 100 and event_id in current_events:
                top100.add(pid)

            continent = int(continent) if continent else None
            country = int(country) if country else None
            by_event = ranks.setdefault(pid, {})
            held = by_event.get(event_id)
            if held is None:
                by_event[event_id] = [world, continent, country]
            else:
                held[0] = min(held[0], world)
                held[1] = _min(held[1], continent)
                held[2] = _min(held[2], country)

    return {'top100': top100, 'ranks': ranks,
            'singles': bests['ranks_single'], 'averages': bests['ranks_average']}


def read_identity(export: ExportReader, pool: set[str]) -> dict[str, dict]:
    """Name, country, continent, gender. sub_id=1 is the person's current identity."""
    continents = dict(export.rows('countries', ('id', 'continent_id')))
    return {
        wca_id: {
            'name': name,
            'country_id': country,
            'continent_id': continents.get(country),
            'gender': (gender or '').strip().lower() or None,
        }
        for wca_id, sub_id, name, country, gender in export.rows(
            'persons', ('wca_id', 'sub_id', 'name', 'country_id', 'gender'))
        if _int(sub_id) == 1 and wca_id in pool
    }


def rankings_from(ranks: dict[str, dict[str, list]],
                  current_events: set[str]) -> dict[str, dict]:
    """Current world ranks, collapsed to the facts the game asks about.

    A top-100 single OR average qualifies for that event; a person in both
    rankings still contributes only one event to top100_event_count.

    Continent and country ranks come along for the "currently holds a record"
    attributes. The record hierarchy needs no special handling here: rank 1 in
    the world is necessarily rank 1 in your continent and your country, so
    `continent_rank = 1` already means "continental record or better".
    """
    out = {}
    for pid, by_event in ranks.items():
        all_ranks = [(event_id, world, continent or 0, country or 0)
                     for event_id, (world, continent, country) in by_event.items()]
        # Retired events are dropped before anything here is derived. Every
        # attribute below is a claim about the present — "currently top 100",
        # "currently holds a world record" — and a frozen ranking in an event
        # the WCA stopped running satisfies none of them. Because so few people
        # ever competed in those events, their ranks are often a person's
        # numerically best: Erik Akkersdijk came out as a Multi-Blind (old
        # style) specialist, and 26% of the elite pool led with a dead event.
        current = [r for r in all_ranks if r[0] in current_events]

        best_event, best_rank = (
            min(((e, w) for e, w, _c, _n in current), key=lambda er: er[1])
            if current else (None, 10 ** 9)
        )
        group = group_of(best_event) if best_event else None
        out[pid] = {
            'main_event': best_event,
            'main_event_group': group,
            'top100_events': sorted(e for e, w, _c, _n in current if w <= 100),
            'top100_event_count': sum(1 for _e, w, _c, _n in current if w <= 100),
            # Fame input: every current ranking, scored by how high it is and
            # how followed the event is. Kept out of the attributes because
            # top-25 / top-100 / top-500 are nested — offering all three as
            # questions would mean asking "top 100?" after "top 25? yes".
            '_rank_weighted': sum(
                event_weight(e) * rank_band_weight(w) for e, w, _c, _n in current),
            'top10_any': best_rank <= 10,
            'currently_wr': best_rank == 1,
            # Fame input: how much attention the event they lead actually gets.
            '_current_wr_weight': max(
                (event_weight(e) for e, w, _c, _n in current if w == 1), default=0.0),
            'currently_cr_or_better': any(c == 1 for _e, _w, c, _n in current),
            'currently_nr_or_better': any(n == 1 for _e, _w, _c, n in current),
            'is_bld_specialist': best_event in EVENT_GROUPS['blind_events'],
            'is_bigcube_specialist': best_event in EVENT_GROUPS['big_cubes'],
            'is_sideevent_specialist': best_event in EVENT_GROUPS['side_events'],
            'is_fmc_specialist': best_event == '333fm',
            'is_oh_specialist': best_event == '333oh',
        }
    return out


def personal_bests_from(singles: dict[str, int],
                        averages: dict[str, int]) -> dict[str, dict]:
    """3x3 milestones. `best` is centiseconds, so sub-10 is < 1000."""
    return {
        pid: {
            'sub10_333_single': 0 < singles.get(pid, 0) < 1000,
            'sub6_333_single': 0 < singles.get(pid, 0) < 600,
            'sub10_333_average': 0 < averages.get(pid, 0) < 1000,
        }
        for pid in set(singles) | set(averages)
    }


def compute_sources(export: ExportReader,
                    current_year: int) -> tuple[dict[str, dict], set[str]]:
    """Every attribute family, and the elite subset of the pool.

    The pool is every competitor with a result, plus the elite: a continental
    record or better, or a current world top 100. The elite are the
    Easy/Normal/Hard tiers; everyone with MIN_EVERYONE_COMPS or more
    competitions is the "Everyone" tier.

    That threshold is where the data runs out, not an arbitrary line: about half
    of WCA's competitors have been to exactly one competition, and those
    profiles are identical in almost every attribute the game asks about.
    """
    current_events = read_current_events(export)
    competitions = read_competitions(export)

    log.info('Scanning results...')
    results = scan_results(export, competitions, current_events, current_year)
    log.info('Scanning rankings...')
    standings = scan_ranks(export, current_events)

    elite = results['record_holders'] | standings['top100']
    log.info('Elite pool (record / top-100): %d cubers', len(elite))
    pool = results['people'] | elite
    log.info('Full profile pool: %d cubers', len(pool))

    in_pool = lambda d: {pid: v for pid, v in d.items() if pid in pool}  # noqa: E731
    sources = {
        'identity': read_identity(export, pool),
        'records': results['records'],
        'career': results['career'],
        'rankings': rankings_from(in_pool(standings['ranks']), current_events),
        'bests': personal_bests_from(in_pool(standings['singles']),
                                     in_pool(standings['averages'])),
        'worlds': results['worlds'],
        'weighted': results['weighted'],
    }
    return sources, elite


# ---------------------------------------------------------------------------
# Step 3 — assemble, score, tier
# ---------------------------------------------------------------------------

def assemble(sources: dict[str, dict[str, dict]],
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

        # Fame inputs, not attributes — popped before the schema check below so
        # they never reach the game or the stored blob.
        fame_inputs = {
            'worlds_podiums': attrs.pop('_worlds_podiums', 0),
            'podium_weighted': attrs.pop('_podium_weighted', 0.0),
            'wr_weighted': attrs.pop('_wr_weighted', 0.0),
            'cr_weighted': attrs.pop('_cr_weighted', 0.0),
            'current_wr_weight': attrs.pop('_current_wr_weight', 0.0),
            'rank_weighted': attrs.pop('_rank_weighted', 0.0),
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
        # Current rankings, banded by how high and scaled by event: top 25 in
        # 3x3 is a different achievement from top 500 in 5BLD.
        + 1.0 * fame_inputs.get('rank_weighted', 0.0)
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
    """Write the matrix to a staging table, then swap it in atomically.

    Refilling `cuber_profiles` in place (TRUNCATE, then ~600 batched INSERTs)
    left a minute or two when it was empty or half written. A web worker that
    loaded profiles then, on a cold start for instance, would cache a partial
    pool for up to an hour. The rows now go into a staging table, which a single
    multi-table RENAME TABLE swaps in. TiDB performs that rename atomically, so
    a reader sees the old table or the new one, never a gap or a partial one.

    A staging table that comes up short is never swapped in: the old matrix
    stays live and the error fails the run. DROP TABLE is as cheap as the
    TRUNCATE it replaces; DELETE would have been billed per row.
    """
    cur = conn.cursor()
    try:
        # Leftovers from a run that failed partway.
        cur.execute(f'DROP TABLE IF EXISTS `{_STAGING_TABLE}`')
        cur.execute(f'DROP TABLE IF EXISTS `{_REPLACED_TABLE}`')
        cur.execute(CREATE_PROFILES_SQL.format(table=_STAGING_TABLE))

        sql = (f'INSERT INTO `{_STAGING_TABLE}` '
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

        cur.execute(f'SELECT COUNT(*) FROM `{_STAGING_TABLE}`')
        written = cur.fetchone()[0]
        if written != len(profiles):
            raise RuntimeError(f'{_STAGING_TABLE} has {written} rows, expected '
                               f'{len(profiles)}; leaving {PROFILES_TABLE} as it was')

        # The swap needs a live table to move aside, and a first build has none.
        cur.execute(CREATE_PROFILES_SQL.format(table=PROFILES_TABLE))
        cur.execute(f'RENAME TABLE `{PROFILES_TABLE}` TO `{_REPLACED_TABLE}`, '
                    f'`{_STAGING_TABLE}` TO `{PROFILES_TABLE}`')
        cur.execute(f'DROP TABLE IF EXISTS `{_REPLACED_TABLE}`')
        cur.execute(f'DROP TABLE IF EXISTS `{_LEGACY_POOL_TABLE}`')
    finally:
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


def run(export: bytes | str | Path | ExportReader | None = None,
        dry_run: bool = False, export_date: str | None = None) -> list[dict]:
    """Build the matrix from a TSV export and, unless `dry_run`, publish it.

    `export` is the TSV export zip, as bytes or a path. Without one, the latest
    export is downloaded. Any failure raises, so a broken build fails whatever
    ran it instead of leaving the game on stale profiles unnoticed.

    The published marker records the export's date, read from the zip's own
    metadata.json. `export_date` is the fallback for a zip without one. An
    undated matrix looks stale to update_database.py, so every run would
    rebuild it.
    """
    if export is None:
        meta = wca_export.fetch_metadata()
        url = wca_export.tsv_url(meta)
        if not url:
            raise RuntimeError(f'No TSV download URL in the export API response: {list(meta)}')
        export = wca_export.download(url, 'TSV export')
    reader = export if isinstance(export, ExportReader) else ExportReader(export)
    export_date = reader.export_date() or export_date
    log.info('Building cuber_profiles from the %s export', export_date or 'undated')

    log.info('Computing attribute families...')
    sources, elite = compute_sources(reader, datetime.now().year)
    for name, data in sources.items():
        log.info('  %-10s %d rows', name, len(data))

    profiles = assemble(sources, elite)
    assign_tiers(profiles)
    report(profiles)

    if dry_run:
        log.info('--dry-run: nothing written')
        return profiles

    conn = get_connection()
    try:
        write_profiles(conn, profiles)
        site_meta.publish_game_profiles_generation(
            conn, site_meta.game_profiles_generation(export_date))
    finally:
        conn.close()
    return profiles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the cuber_profiles feature matrix for Guess the Cuber'
    )
    parser.add_argument('--export', metavar='ZIP',
                        help='A downloaded WCA TSV export zip (default: download the latest)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Compute and report, but write nothing')
    args = parser.parse_args()
    run(export=args.export, dry_run=args.dry_run)
