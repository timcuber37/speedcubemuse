"""Integrity tests for the `cuber_profiles` feature matrix.

These check the data the game plays over: that the pool is the right shape, that
every attribute declared in the schema actually got computed, and that a handful
of cubers whose facts are independently known came out right.

Requires a built matrix:
    python scripts/build_cuber_profiles.py
    python -m pytest tests/test_game_profiles.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

from build_cuber_profiles import (RECENCY_FLOOR, fame_score,  # noqa: E402
                                  recency_weight)

from services.game.attributes import (ATTRIBUTES, BY_KEY, EVENT_GROUPS,
                                      EVENT_NAMES, EVENT_TIERS, event_weight,
                                      group_of)
from services.game.profiles import (RESIDENT_TIERS, TIERS, get_engine,
                                    load_profiles, search_by_name)

# The resident pool is the elite tiers plus everyone with 5+ competitions.
# A result far outside this band means a pool query broke, not that the sport
# changed.
MIN_POOL, MAX_POOL = 20_000, 150_000

# Tiers 1-3 are the record / top-100 competitors, ranked against each other.
ELITE_TIERS = (1, 2, 3)
MIN_ELITE, MAX_ELITE = 1_200, 6_000


@pytest.fixture(scope='module')
def profiles():
    # The full resident pool, not the default elite depth — these tests assert
    # on the whole thing the app can hold in memory.
    rows = load_profiles(RESIDENT_TIERS)
    if not rows:
        pytest.skip('cuber_profiles is empty — run scripts/build_cuber_profiles.py')
    return rows


@pytest.fixture(scope='module')
def elite(profiles):
    return [r for r in profiles if r['tier'] in ELITE_TIERS]


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------

def test_pool_size_is_plausible(profiles):
    assert MIN_POOL <= len(profiles) <= MAX_POOL


def test_wca_ids_are_unique(profiles):
    ids = [r['wca_id'] for r in profiles]
    assert len(ids) == len(set(ids))


def test_every_tier_is_populated(profiles):
    for tier in (1, 2, 3, 4):
        assert sum(1 for r in profiles if r['tier'] == tier) > 0, f'tier {tier} empty'


def test_elite_pool_is_the_expected_size(profiles):
    elite = [r for r in profiles if r['tier'] in ELITE_TIERS]
    assert MIN_ELITE <= len(elite) <= MAX_ELITE


def test_long_tail_is_never_loaded(profiles):
    """Tier 5 stays in the database; loading ~245k rows would exhaust the box."""
    assert all(r['tier'] <= RESIDENT_TIERS for r in profiles)


def test_tiers_are_ordered_by_fame(profiles):
    """Within the elite, tier 1 outranks tier 2 and tier 2 outranks tier 3.

    Tier 4 is deliberately excluded: it is ranked separately, so that adding
    50k ordinary competitors can never displace a world record holder from the
    Easy tier.
    """
    by_tier = {t: [r['fame'] for r in profiles if r['tier'] == t] for t in ELITE_TIERS}
    assert min(by_tier[1]) >= max(by_tier[2])
    assert min(by_tier[2]) >= max(by_tier[3])


def test_names_and_countries_are_present(profiles):
    for row in profiles:
        assert row['name'].strip(), row['wca_id']
        assert row['country_id'], row['wca_id']


# ---------------------------------------------------------------------------
# Attribute completeness
# ---------------------------------------------------------------------------

def test_every_declared_attribute_is_computed(profiles):
    """A schema entry with no matching column would silently never be asked."""
    for attr in ATTRIBUTES:
        missing = [r['wca_id'] for r in profiles if attr.key not in r['attrs']]
        assert not missing, f'{attr.key} missing for {len(missing)} profiles'


def test_no_unexpected_attributes(profiles):
    """Catches a builder field that outlived its schema entry."""
    for row in profiles[:200]:
        extra = set(row['attrs']) - set(BY_KEY)
        assert not extra, f'{row["wca_id"]} has undeclared attrs: {extra}'


def test_attribute_types_match_their_kind(profiles):
    for row in profiles:
        for attr in ATTRIBUTES:
            value = row['attrs'][attr.key]
            if attr.kind == 'bool':
                assert isinstance(value, bool), f'{attr.key} on {row["wca_id"]}'
            elif attr.kind == 'numeric':
                assert isinstance(value, (int, float)) and not isinstance(value, bool), \
                    f'{attr.key} on {row["wca_id"]}'
            elif attr.kind == 'multi':
                # Tuple in memory, list in the database: _shape folds repeated
                # strings onto shared objects and freezes the sequence.
                assert isinstance(value, (list, tuple)), \
                    f'{attr.key} on {row["wca_id"]}'
            else:
                assert value is None or isinstance(value, str), \
                    f'{attr.key} on {row["wca_id"]}'


def test_every_boolean_attribute_splits_the_pool(profiles):
    """An attribute true for everyone (or no one) is a question worth deleting."""
    n = len(profiles)
    for attr in ATTRIBUTES:
        if attr.kind != 'bool':
            continue
        share = sum(1 for r in profiles if r['attrs'][attr.key]) / n
        assert 0.0 < share < 1.0, f'{attr.key} is constant across the pool'


def test_numeric_thresholds_are_well_spread_on_the_elite_pool(elite):
    """Thresholds are tuned against the elite pool, so check the band there.

    The build script's --dry-run report surfaces this; asserting it here stops
    a retuned threshold from silently rotting later.
    """
    n = len(elite)
    for attr in ATTRIBUTES:
        if attr.kind != 'numeric':
            continue
        for t in attr.thresholds:
            share = sum(1 for r in elite if (r['attrs'][attr.key] or 0) >= t) / n
            assert 0.005 < share < 0.995, (
                f'{attr.key} >= {t} matches {share:.1%} of the elite pool — '
                f'retune the thresholds in services/game/attributes.py'
            )


def test_numeric_thresholds_are_reachable_on_every_pool(profiles):
    """Weaker check across the whole resident pool: nothing is 0% or 100%.

    Rarity is fine here and expected — only ~0.5% of the Everyone pool has ever
    set a world record, and that question is enormously informative when the
    answer is yes. What would be dead weight is a threshold nobody meets at all.
    """
    n = len(profiles)
    for attr in ATTRIBUTES:
        if attr.kind != 'numeric':
            continue
        for t in attr.thresholds:
            hits = sum(1 for r in profiles if (r['attrs'][attr.key] or 0) >= t)
            assert 0 < hits < n, (
                f'{attr.key} >= {t} matches {hits} of {n} — it can never split'
            )


# ---------------------------------------------------------------------------
# Internal consistency
# ---------------------------------------------------------------------------

def test_record_flags_agree_with_their_counts(profiles):
    for row in profiles:
        a = row['attrs']
        wr = a['wr_single_count'] + a['wr_average_count']
        assert a['has_wr'] == (wr > 0)
        assert a['has_cr_or_better'] == (wr + a['cr_count'] > 0)
        assert a['has_nr_or_better'] == (wr + a['cr_count'] + a['nr_count'] > 0)


def test_records_are_cumulative(profiles):
    """WR > CR > NR. A world record satisfies every weaker bar.

    The regression this guards: the FMC world record holder has one WR and zero
    recorded CRs, and used to answer "no" to "continental record or better".
    """
    for row in profiles:
        a = row['attrs']
        if a['has_wr']:
            assert a['has_cr_or_better'], row['wca_id']
        if a['has_cr_or_better']:
            assert a['has_nr_or_better'], row['wca_id']
        if a['currently_wr']:
            assert a['currently_cr_or_better'], row['wca_id']
        if a['currently_cr_or_better']:
            assert a['currently_nr_or_better'], row['wca_id']


def test_current_record_holder_is_top_of_a_ranking(profiles):
    """`currently_wr` comes from world_rank == 1, so it implies a top-100 event."""
    for row in profiles:
        if row['attrs']['currently_wr']:
            assert row['attrs']['top100_events'], row['wca_id']
            assert row['attrs']['top10_any'], row['wca_id']


def test_career_years_are_coherent(profiles):
    """WCA results start at the 1982 World Championship, not the 2003 restart."""
    for row in profiles:
        a = row['attrs']
        assert 1982 <= a['first_year'] <= a['last_year'] <= 2100, row['wca_id']
        assert a['years_active'] >= 1


def test_specialists_are_mutually_exclusive(profiles):
    """Specialty is derived from a single best event, so at most one can hold."""
    keys = ('is_bld_specialist', 'is_bigcube_specialist', 'is_sideevent_specialist',
            'is_fmc_specialist', 'is_oh_specialist')
    for row in profiles:
        assert sum(bool(row['attrs'][k]) for k in keys) <= 1, row['wca_id']


def test_specialist_flags_agree_with_the_event_groups(profiles):
    """The booleans and EVENT_GROUPS must not drift apart."""
    pairs = [
        ('is_bld_specialist', 'blind_events'),
        ('is_bigcube_specialist', 'big_cubes'),
        ('is_sideevent_specialist', 'side_events'),
    ]
    for row in profiles:
        main = row['attrs']['main_event']
        for key, group in pairs:
            assert row['attrs'][key] == (main in EVENT_GROUPS[group]), \
                f'{row["wca_id"]} {key} vs main_event {main}'


def test_main_event_group_matches_main_event(profiles):
    for row in profiles:
        assert row['attrs']['main_event_group'] == group_of(row['attrs']['main_event'])


def test_event_groups_cover_the_user_facing_definitions():
    """These groupings are how cubers talk; changing them changes the game."""
    assert set(EVENT_GROUPS['big_cubes']) == {'444', '555', '666', '777'}
    assert set(EVENT_GROUPS['blind_events']) == {'333bf', '444bf', '555bf', '333mbf'}
    assert set(EVENT_GROUPS['side_events']) == {'pyram', 'minx', 'skewb', 'sq1', 'clock'}
    # No event may sit in two groups, or group questions would overlap.
    seen = [e for members in EVENT_GROUPS.values() for e in members]
    assert len(seen) == len(set(seen))


def test_shared_strings_are_folded(profiles):
    """Attribute keys must be one object each, not one per row.

    json.loads allocates a fresh key string per row; at 52k rows that was the
    single largest use of memory in the worker (192 MB -> 69 MB once folded).
    """
    sample = profiles[:2000]
    if len(sample) < 2:
        pytest.skip('need at least two profiles')
    first = sample[0]['attrs']
    for row in sample[1:]:
        for key in row['attrs']:
            # `is`, not `==`: the point is that they are the same object.
            assert any(key is k for k in first), f'{key} not folded'
            break


def test_top100_events_is_consistent_with_its_count(profiles):
    for row in profiles:
        a = row['attrs']
        assert len(a['top100_events']) == a['top100_event_count'], row['wca_id']


# ---------------------------------------------------------------------------
# Fame weighting — event popularity and recency
# ---------------------------------------------------------------------------

def test_event_tiers_cover_every_current_event():
    """A current event missing from the tiers silently falls to the lowest one."""
    current = set(EVENT_NAMES) - {'magic', 'mmagic', '333ft', '333mbo', 'fto'}
    missing = sorted(current - set(EVENT_TIERS))
    assert not missing, f'current events with no popularity tier: {missing}'


def test_event_weights_descend_by_tier():
    """3x3 outweighs the popular side events, which outweigh the rest."""
    assert (event_weight('333')                     # tier 1
            > event_weight('222') == event_weight('333oh')   # tier 2
            > event_weight('555') == event_weight('333bf')   # tier 3
            > event_weight('555bf') == event_weight('333fm'))  # tier 4
    assert event_weight('333') / event_weight('555bf') >= 3, (
        'the spread between 3x3 and the PB events is too small to matter'
    )


def test_unlisted_events_get_the_lowest_weight():
    """Retired events and anything the WCA adds later must not rank high."""
    assert event_weight('magic') == event_weight('555bf')
    assert event_weight('a-brand-new-event') == event_weight('555bf')


def test_recent_achievements_outweigh_old_ones():
    year = 2026
    fresh = recency_weight(year, year)
    five = recency_weight(year - 5, year)
    twenty = recency_weight(year - 20, year)

    assert fresh == 1.0
    assert fresh > five > twenty
    assert five == pytest.approx(0.6, abs=0.01), 'five years should roughly halve'
    # A floor, not a cliff: an old world record still says something.
    assert twenty >= RECENCY_FLOOR
    assert recency_weight(1982, year) >= RECENCY_FLOOR


def test_recency_never_rewards_a_future_date():
    """Competitions are announced ahead of time; a future year must not exceed 1."""
    assert recency_weight(2030, 2026) == 1.0


def test_fame_rewards_a_recent_headline_record_over_an_old_niche_one():
    """The whole point of the weighting, as one comparison.

    Measured as what the record *adds*, not as a total. Career volume, still
    competing, and current rankings are a large shared base that both
    competitors earn independently of which record they hold — comparing totals
    would dilute the very thing under test.
    """
    base = {'total_results': 500, 'last_year': 2026, 'top100_events': ('333',)}
    without = fame_score(base, {}, 2026)
    headline = fame_score(
        base, {'wr_weighted': event_weight('333') * recency_weight(2026, 2026)}, 2026)
    niche = fame_score(
        base, {'wr_weighted': event_weight('555bf') * recency_weight(2006, 2026)}, 2026)

    assert (headline - without) > (niche - without) * 10, (
        f'a fresh 3x3 WR adds {headline - without} and a 20-year-old 5BLD WR '
        f'adds {niche - without} — the bias is too weak to matter'
    )


def test_weighting_inputs_never_leak_into_attributes(profiles):
    """Fame inputs are popped in assemble(); none may reach the game."""
    for row in profiles[:500]:
        leaked = [k for k in row['attrs'] if k.startswith('_')]
        assert not leaked, f'{row["wca_id"]} carries fame inputs as attrs: {leaked}'


def test_every_event_in_the_data_has_a_display_name(profiles):
    """An event id with no name reaches the player as a raw code.

    The WCA's retired events — Magic, Master Magic, With Feet, old-style
    Multi-Blind — are still somebody's strongest event, so they surface as
    questions even though nobody competes in them now.
    """
    seen = set()
    for row in profiles:
        if row['attrs'].get('main_event'):
            seen.add(row['attrs']['main_event'])
        seen.update(row['attrs'].get('top100_events') or ())

    missing = sorted(seen - set(EVENT_NAMES))
    assert not missing, f'event ids with no display name: {missing}'


def test_top100_events_covers_more_than_3x3(profiles):
    """The whole point of making this multi-valued rather than a 3x3 boolean."""
    non_333 = {
        e for row in profiles for e in row['attrs']['top100_events'] if e != '333'
    }
    assert len(non_333) > 10, f'only found top-100 entries for {non_333}'


def test_elite_tiers_meet_the_stated_criteria(profiles):
    """Tiers 1-3 mean CR-or-better, or a current world top 100 somewhere."""
    for row in (r for r in profiles if r['tier'] in ELITE_TIERS):
        a = row['attrs']
        assert a['has_cr_or_better'] or a['top100_event_count'] > 0, (
            f'{row["wca_id"]} ({row["name"]}) meets no eligibility criterion'
        )


def test_worlds_podium_implies_worlds_appearance(profiles):
    for row in profiles:
        if row['attrs']['has_worlds_podium']:
            assert row['attrs']['competed_at_worlds'], row['wca_id']


def test_sub6_implies_sub10(profiles):
    for row in profiles:
        if row['attrs']['sub6_333_single']:
            assert row['attrs']['sub10_333_single'], row['wca_id']


# ---------------------------------------------------------------------------
# Spot checks against independently known facts
# ---------------------------------------------------------------------------

def test_known_cuber_facts(profiles):
    """Max Park is the safest possible fixture: multiple WRs, American, active."""
    by_id = {r['wca_id']: r for r in profiles}
    max_park = by_id.get('2012PARK03')
    if max_park is None:
        pytest.skip('2012PARK03 not in pool — WCA ID may have changed')

    assert max_park['country_id'] == 'USA'
    assert max_park['continent_id'] == '_North America'
    assert max_park['attrs']['has_wr']
    assert max_park['attrs']['sub6_333_single']
    assert max_park['attrs']['competed_at_worlds']
    assert max_park['tier'] == 1, 'a multi-WR holder should be in the easy tier'


def test_search_finds_a_known_name(profiles):
    hits = search_by_name('Max Park')
    assert any(h['wca_id'] == '2012PARK03' for h in hits)


def test_search_ignores_too_short_queries():
    assert search_by_name('a') == []
    assert search_by_name('') == []


# ---------------------------------------------------------------------------
# Engine wiring
# ---------------------------------------------------------------------------

def test_every_difficulty_builds_an_engine():
    for difficulty in TIERS:
        engine = get_engine(difficulty)
        assert engine is not None, difficulty
        assert engine.rows


def test_difficulty_pools_are_nested():
    """Normal must contain Easy, and Hard must contain Normal."""
    pools = {d: {r['wca_id'] for r in get_engine(d).rows} for d in TIERS}
    assert pools['easy'] < pools['normal'] < pools['hard']


def test_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError):
        get_engine('impossible')
