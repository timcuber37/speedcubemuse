"""Offline tests for what an answer log settles across different attributes.

No database: the cubers here are synthetic, with their derived attributes built
the same way scripts/build_cuber_profiles.py builds them. The DB-backed check
that every relation holds for every real profile is in test_game_profiles.py.
"""
import random

import pytest

from services.game.attributes import (ATTRIBUTES, CURRENT_EVENTS, EVENT_GROUPS,
                                      RELATIONS, Predicate, group_of,
                                      pool_relations, predicate_from_id)
from services.game.engine import Engine, answer_for

DEFAULTS = {'bool': False, 'numeric': 0, 'multi': (), 'categorical': None}


def cuber(wca_id, *, country, continent, main_event=None, top100=(), top10=False,
          wr_single=0, wr_average=0, cr=0, nr=0, current_rank=None, **attrs):
    """A profile whose derived attributes agree with each other, as the builder's do.

    `current_rank` is the best current world/continent/country rank as 'wr',
    'cr' or 'nr'.
    """
    full = {a.key: DEFAULTS[a.kind] for a in ATTRIBUTES}
    full.update(attrs)
    wr = wr_single + wr_average
    full.update({
        'country_id': country,
        'continent_id': continent,
        'main_event': main_event,
        'main_event_group': group_of(main_event) if main_event else None,
        'is_bld_specialist': main_event in EVENT_GROUPS['blind_events'],
        'is_bigcube_specialist': main_event in EVENT_GROUPS['big_cubes'],
        'is_sideevent_specialist': main_event in EVENT_GROUPS['side_events'],
        'is_fmc_specialist': main_event == '333fm',
        'is_oh_specialist': main_event == '333oh',
        'top100_events': tuple(sorted(top100)),
        'top100_event_count': len(top100),
        'top10_any': top10,
        'currently_wr': current_rank == 'wr',
        'currently_cr_or_better': current_rank in ('wr', 'cr'),
        'currently_nr_or_better': current_rank in ('wr', 'cr', 'nr'),
        'wr_single_count': wr_single,
        'wr_average_count': wr_average,
        'cr_count': cr,
        'nr_count': nr,
        'has_wr': wr > 0,
        'has_cr_or_better': wr + cr > 0,
        'has_nr_or_better': wr + cr + nr > 0,
        # The fixture's 17-event cubers have every current event.
        'competed_all_events': full.get('events_competed', 0) >= len(CURRENT_EVENTS),
    })
    if full['sub6_333_single'] or full['sub10_333_average']:
        full['sub10_333_single'] = True
    if full['first_year']:
        full['years_active'] = full['last_year'] - full['first_year'] + 1
    return {'wca_id': wca_id, 'name': wca_id, 'attrs': full}


ROWS = [
    cuber('usa-333', country='USA', continent='_North America', main_event='333',
          first_year=2019, last_year=2026, comp_count=40, total_results=800,
          events_competed=12, sub10_333_single=True, sub10_333_average=True,
          sub6_333_single=True, competed_at_worlds=True, is_active=True, gender='m'),
    cuber('usa-minx', country='USA', continent='_North America', main_event='minx',
          first_year=2015, last_year=2025, comp_count=90, total_results=1200,
          events_competed=16, nr=3, current_rank='nr', gender='f'),
    cuber('can-sq1', country='Canada', continent='_North America', main_event='sq1',
          top100=('sq1',), first_year=2012, last_year=2024, comp_count=60,
          total_results=700, events_competed=17, cr=2, nr=8, gender='m'),
    cuber('ger-444', country='Germany', continent='_Europe', main_event='444',
          top100=('444', '555'), first_year=2010, last_year=2026, comp_count=160,
          total_results=2500, events_competed=17, sub10_333_single=True, cr=4,
          nr=20, current_rank='cr', gender='m'),
    cuber('fra-bld', country='France', continent='_Europe', main_event='333bf',
          top100=('333bf',), top10=True, first_year=2016, last_year=2023,
          comp_count=30, total_results=300, events_competed=9, nr=2, gender='f'),
    cuber('chn-oh', country='China', continent='_Asia', main_event='333oh',
          top100=('333', '333oh'), top10=True, wr_single=1, current_rank='wr',
          first_year=2017, last_year=2026, comp_count=45, total_results=600,
          events_competed=8, sub6_333_single=True, sub10_333_average=True,
          competed_at_worlds=True, has_worlds_podium=True, gender='m'),
    cuber('aus-fmc', country='Australia', continent='_Oceania', main_event='333fm',
          first_year=2013, last_year=2019, comp_count=22, total_results=150,
          events_competed=10, gender='m'),
    cuber('bra-222', country='Brazil', continent='_South America', main_event='222',
          first_year=2021, last_year=2026, comp_count=8, total_results=90,
          events_competed=6, gender='o'),
]


@pytest.fixture(scope='module')
def engine():
    return Engine(ROWS)


def known(engine, log, pred_id):
    return engine.knowledge(log).value(predicate_from_id(pred_id))


# ---------------------------------------------------------------------------
# The reported game
# ---------------------------------------------------------------------------

# A real 39-question game, kept verbatim. The player's cuber is in the
# Everyone pool. These are the player's answers, not the table's.
REPORTED_GAME = [
    ('main_event_group:eq:side_events', 'no'),
    ('years_active:gte:5', 'yes'),
    ('sub10_333_single:is:True', 'yes'),
    ('sub10_333_average:is:True', 'yes'),
    ('countries_competed_in:gte:2', 'no'),
    ('comp_count:gte:20', 'yes'),
    ('events_competed:gte:12', 'yes'),
    ('comp_count:gte:40', 'yes'),
    ('country_id:eq:USA', 'yes'),
    ('sub6_333_single:is:True', 'yes'),
    ('events_competed:gte:15', 'no'),
    ('main_event_group:eq:big_cubes', 'no'),
    ('main_event:eq:333', 'yes'),
    ('competed_at_worlds:is:True', 'yes'),
    ('comp_count:gte:75', 'no'),
    ('total_results:gte:400', 'yes'),
    ('years_active:gte:8', 'no'),
    ('top100_event_count:gte:1', 'no'),
    ('is_oh_specialist:is:True', 'no'),
    ('continent_id:eq:_North America', 'yes'),
    ('is_active:is:True', 'yes'),
    ('is_sideevent_specialist:is:True', 'no'),
    ('total_results:gte:800', 'yes'),
    ('top100_events:has_group:side_events', 'no'),
    ('top100_events:has:333', 'no'),
    ('gender:eq:m', 'yes'),
    ('is_bigcube_specialist:is:True', 'no'),
    ('is_fmc_specialist:is:True', 'no'),
    ('has_nr_or_better:is:True', 'no'),
    ('top100_events:has:222', 'no'),
    ('main_event_group:eq:blind_events', 'no'),
    ('top100_events:has:333oh', 'no'),
    ('top10_any:is:True', 'no'),
    ('top100_events:has_group:big_cubes', 'no'),
    ('nr_count:gte:1', 'no'),
    ('is_bld_specialist:is:True', 'no'),
    ('cr_count:gte:1', 'no'),
    ('competed_all_events:is:True', 'no'),
    ('top100_events:has:333fm', 'no'),
]

# Question number (1-based) -> the earlier answer that already settled it.
REPEATED = {
    19: 'strongest event 3x3',
    20: 'represents the USA',
    22: 'strongest event not a side event',
    24: 'not top 100 in anything',
    25: 'not top 100 in anything',
    27: 'strongest event not a big cube event',
    28: 'strongest event 3x3',
    30: 'not top 100 in anything',
    31: 'strongest event 3x3',
    32: 'not top 100 in anything',
    33: 'not top 100 in anything',
    34: 'not top 100 in anything',
    35: 'no national record or better',
    36: 'strongest event not a blindfolded event',
    37: 'no national record or better',
    38: 'fewer than 15 events',
    39: 'not top 100 in anything',
}


@pytest.mark.parametrize('number', range(1, len(REPORTED_GAME) + 1))
def test_reported_game_repeats_are_settled_and_nothing_else(engine, number):
    pred_id, answer = REPORTED_GAME[number - 1]
    value = known(engine, REPORTED_GAME[:number - 1], pred_id)
    if number in REPEATED:
        # Settled, and settled to the answer the player then gave.
        assert value is (answer == 'yes'), REPEATED[number]
    else:
        assert value is None


# ---------------------------------------------------------------------------
# Each family of relations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('log,pred_id,expected', [
    # Country -> continent, and continent -> every country outside it.
    ([('country_id:eq:USA', 'yes')], 'continent_id:eq:_North America', True),
    ([('country_id:eq:USA', 'yes')], 'continent_id:eq:_Europe', False),
    ([('continent_id:eq:_Europe', 'yes')], 'country_id:eq:USA', False),
    ([('continent_id:eq:_Europe', 'no')], 'country_id:eq:Germany', False),
    ([('continent_id:eq:_Europe', 'yes')], 'country_id:eq:Germany', None),
    ([('country_id:eq:USA', 'no')], 'continent_id:eq:_North America', None),
    # Strongest event <-> its group <-> the specialist flag.
    ([('main_event:eq:333', 'yes')], 'is_oh_specialist:is:True', False),
    ([('main_event:eq:333', 'yes')], 'main_event_group:eq:blind_events', False),
    ([('main_event:eq:minx', 'yes')], 'main_event_group:eq:side_events', True),
    ([('main_event:eq:minx', 'yes')], 'is_sideevent_specialist:is:True', True),
    ([('is_bld_specialist:is:True', 'yes')], 'main_event_group:eq:blind_events', True),
    ([('is_bld_specialist:is:True', 'yes')], 'main_event:eq:333', False),
    ([('main_event_group:eq:big_cubes', 'no')], 'main_event:eq:444', False),
    ([('main_event_group:eq:big_cubes', 'no')], 'is_bigcube_specialist:is:True', False),
    ([('is_fmc_specialist:is:True', 'yes')], 'main_event:eq:333fm', True),
    ([('main_event_group:eq:side_events', 'yes')], 'main_event:eq:sq1', None),
    # Top-100 count <-> top-100 events.
    ([('top100_event_count:gte:1', 'no')], 'top100_events:has:333', False),
    ([('top100_event_count:gte:1', 'no')], 'top10_any:is:True', False),
    ([('top100_event_count:gte:1', 'no')], 'currently_wr:is:True', False),
    ([('top100_events:has:sq1', 'yes')], 'top100_event_count:gte:1', True),
    ([('top10_any:is:True', 'yes')], 'top100_event_count:gte:1', True),
    ([('top100_events:has:333', 'yes'), ('top100_events:has:333oh', 'yes')],
     'top100_event_count:gte:2', True),
    ([('top100_event_count:gte:2', 'no'), ('top100_events:has:333', 'yes')],
     'top100_events:has:444', False),
    ([('top100_events:has_group:big_cubes', 'yes')], 'top100_event_count:gte:1', True),
    # The best-ranked event is top 100 whenever anything is.
    ([('main_event:eq:sq1', 'yes'), ('top100_event_count:gte:1', 'yes')],
     'top100_events:has:sq1', True),
    ([('main_event:eq:sq1', 'yes'), ('top100_events:has:sq1', 'no')],
     'top100_event_count:gte:1', False),
    ([('is_bld_specialist:is:True', 'yes'), ('top100_event_count:gte:1', 'yes')],
     'top100_events:has_group:blind_events', True),
    # Records: the "or better" flags against the counts.
    ([('has_nr_or_better:is:True', 'no')], 'nr_count:gte:1', False),
    ([('has_nr_or_better:is:True', 'no')], 'cr_count:gte:5', False),
    ([('has_nr_or_better:is:True', 'no')], 'wr_single_count:gte:1', False),
    ([('has_nr_or_better:is:True', 'no')], 'currently_nr_or_better:is:True', False),
    ([('nr_count:gte:10', 'yes')], 'has_nr_or_better:is:True', True),
    ([('wr_average_count:gte:2', 'yes')], 'has_cr_or_better:is:True', True),
    ([('has_nr_or_better:is:True', 'yes'), ('has_cr_or_better:is:True', 'no')],
     'nr_count:gte:1', True),
    ([('has_wr:is:True', 'yes'), ('wr_single_count:gte:1', 'no')],
     'wr_average_count:gte:1', True),
    ([('currently_cr_or_better:is:True', 'yes')], 'has_cr_or_better:is:True', True),
    ([('has_nr_or_better:is:True', 'yes')], 'nr_count:gte:1', None),
    # 3x3 milestones.
    ([('sub10_333_average:is:True', 'yes')], 'sub10_333_single:is:True', True),
    ([('sub10_333_single:is:True', 'no')], 'sub10_333_average:is:True', False),
    ([('top100_events:has:333', 'yes')], 'sub6_333_single:is:True', True),
    ([('sub10_333_single:is:True', 'yes')], 'sub10_333_average:is:True', None),
    # Career orderings.
    ([('first_year:gte:2019', 'yes')], 'last_year:gte:2017', True),
    ([('last_year:gte:2017', 'no')], 'first_year:gte:2019', False),
    ([('comp_count:gte:150', 'yes')], 'total_results:gte:100', True),
    ([('total_results:gte:100', 'no')], 'comp_count:gte:150', False),
    ([('events_competed:gte:15', 'no')], 'competed_all_events:is:True', False),
    ([('competed_all_events:is:True', 'yes')], 'events_competed:gte:17', True),
])
def test_relation(engine, log, pred_id, expected):
    assert known(engine, log, pred_id) is expected


def test_chains_run_through_several_relations(engine):
    """USA -> North America -> not Europe -> not Germany."""
    log = [('country_id:eq:USA', 'yes')]
    assert known(engine, log, 'country_id:eq:Germany') is False
    # Not top 100 anywhere -> not top 10 -> no current world record -> and so on.
    log = [('top100_event_count:gte:1', 'no')]
    assert known(engine, log, 'currently_wr:is:True') is False


def test_probably_counts_and_dont_know_does_not(engine):
    assert known(engine, [('country_id:eq:USA', 'probably')],
                 'continent_id:eq:_North America') is True
    assert known(engine, [('country_id:eq:USA', 'probably_not')],
                 'continent_id:eq:_North America') is None
    assert known(engine, [('country_id:eq:USA', 'dont_know')],
                 'continent_id:eq:_North America') is None


def test_contradictory_answers_keep_the_earlier_one(engine):
    """Possible with soft answers or a log from before a relation existed."""
    log = [('country_id:eq:USA', 'yes'), ('continent_id:eq:_North America', 'no')]
    assert known(engine, log, 'continent_id:eq:_North America') is True
    log = [('comp_count:gte:40', 'yes'), ('comp_count:gte:20', 'no')]
    assert known(engine, log, 'comp_count:gte:20') is True


def test_unknown_predicates_in_the_log_are_ignored(engine):
    assert engine.knowledge([('retired_attr:is:True', 'yes')]).value(
        Predicate('has_wr', 'is', True)) is None


def test_implied_excludes_the_questions_actually_asked(engine):
    log = [('country_id:eq:USA', 'yes')]
    settled = engine.implied(log)
    assert 'country_id:eq:USA' not in settled
    assert 'continent_id:eq:_North America' in settled
    assert 'country_id:eq:Germany' in settled


# ---------------------------------------------------------------------------
# Soundness: truthful answers only ever settle true facts
# ---------------------------------------------------------------------------

def test_every_relation_holds_for_the_synthetic_cubers():
    clauses = RELATIONS + tuple(pool_relations(ROWS))
    for row in ROWS:
        for clause in clauses:
            assert any(p.test(row['attrs']) is t for p, t in clause), (row['wca_id'], clause)


@pytest.mark.parametrize('seed', range(40))
def test_truthful_answers_settle_only_what_is_true(engine, seed):
    rng = random.Random(seed)
    secret = rng.choice(ROWS)
    asked = rng.sample(engine._predicates, 12)
    log = [(p.as_id(), answer_for(secret['attrs'], p)) for p in asked]
    knowledge = engine.knowledge(log)
    for pred in engine._predicates:
        value = knowledge.value(pred)
        if value is not None:
            assert value is pred.test(secret['attrs']), (secret['wca_id'], pred, log)


def test_pool_relations_skip_countries_with_conflicting_continents():
    rows = [
        {'attrs': {'country_id': 'X', 'continent_id': '_Asia'}},
        {'attrs': {'country_id': 'X', 'continent_id': '_Europe'}},
        {'attrs': {'country_id': 'Y', 'continent_id': '_Asia'}},
    ]
    clauses = pool_relations(rows)
    assert clauses == [((Predicate('country_id', 'eq', 'Y'), False),
                        (Predicate('continent_id', 'eq', '_Asia'), True))]


# ---------------------------------------------------------------------------
# End to end: the engine stops repeating itself
# ---------------------------------------------------------------------------

def test_engine_never_asks_a_settled_question(engine):
    for secret in ROWS:
        log, belief = [], engine.initial_belief()
        while not engine.should_guess(belief, log):
            pred = engine.pick_question(belief, log)
            if pred is None:
                break
            assert engine.knowledge(log).value(pred) is None, pred.render()
            answer = answer_for(secret['attrs'], pred)
            log.append((pred.as_id(), answer))
            belief = engine.update(belief, pred.as_id(), answer)
