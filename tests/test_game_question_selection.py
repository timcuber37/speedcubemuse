"""Offline regressions for avoiding overlapping questions in a game."""
from itertools import product
from unittest.mock import patch

import pytest

from services.game.attributes import Predicate
from services.game.engine import ANSWERS, Engine, answer_for


@pytest.fixture
def engine():
    rows = []
    for first, last, gender in product((2008, 2014, 2018), (2020, 2026), ('m', 'f')):
        rows.append({
            'wca_id': f'test-{len(rows)}',
            'name': f'Test {len(rows)}',
            'attrs': {
                'first_year': first,
                'last_year': last,
                'years_active': last - first + 1,
                'is_active': last >= 2025,
                'gender': gender,
            },
        })
    return Engine(rows)


@pytest.mark.parametrize('answered,blocked_key', [
    (Predicate('first_year', 'gte', 2016), 'years_active'),
    # A replayed threshold need not be in the generated question catalog.
    (Predicate('years_active', 'gte', 10), 'first_year'),
    (Predicate('last_year', 'gte', 2024), 'is_active'),
    (Predicate('is_active', 'is', True), 'last_year'),
    (Predicate('is_active', 'is', False), 'last_year'),
])
@pytest.mark.parametrize('answer', ANSWERS)
def test_related_question_types_do_not_follow_each_other(engine, answered, blocked_key, answer):
    log = [(answered.as_id(), answer)]
    belief = engine.replay(log)
    blocked = {p.as_id() for p in engine._predicates if p.key == blocked_key}
    assert blocked
    assert blocked <= engine.overlapping(log)
    # Overlap suppression must not pretend the other answers are known.
    assert blocked.isdisjoint(engine.implied(log))

    for _ in range(12):
        question = engine.pick_question(belief, log)
        if question is None:
            break
        assert question.key != blocked_key
        answer = answer_for(engine.rows[0]['attrs'], question)
        log.append((question.as_id(), answer))
        belief = engine.update(belief, question.as_id(), answer)


def test_useful_threshold_refinements_remain_available():
    engine = Engine([
        {'attrs': {'first_year': 2017, 'years_active': 10}},
        {'attrs': {'first_year': 2023, 'years_active': 4}},
    ])
    log = [(Predicate('first_year', 'gte', 2016).as_id(), 'yes')]
    question = engine.pick_question(engine.replay(log), log)
    assert question is not None
    assert question.key == 'first_year'
    assert question.value > 2016
    assert question.test(engine.rows[0]['attrs']) != question.test(engine.rows[1]['attrs'])


def test_career_start_does_not_block_recent_activity(engine):
    log = [(Predicate('first_year', 'gte', 2016).as_id(), 'yes')]
    blocked = engine.overlapping(log)
    assert Predicate('last_year', 'gte', 2024).as_id() not in blocked
    assert Predicate('is_active', 'is', True).as_id() not in blocked


def test_undo_releases_the_question_family(engine):
    log = [(Predicate('years_active', 'gte', 10).as_id(), 'dont_know')]
    assert engine.overlapping(log)
    assert engine.overlapping(log[:-1]) == set()


def test_old_logs_with_both_question_types_keep_the_first_choice(engine):
    log = [
        (Predicate('years_active', 'gte', 10).as_id(), 'yes'),
        (Predicate('first_year', 'gte', 2016).as_id(), 'no'),
    ]
    blocked = engine.overlapping(log)
    assert Predicate('first_year', 'gte', 2019).as_id() in blocked
    assert Predicate('years_active', 'gte', 12).as_id() not in blocked


def test_unknown_questions_do_not_choose_a_family(engine):
    assert engine.overlapping([('retired_attr:is:True', 'yes')]) == set()


def test_overlap_exclusions_do_not_count_towards_the_question_cap(engine):
    log = [(Predicate('years_active', 'gte', 10).as_id(), 'dont_know')]
    assert not engine.should_guess(engine.replay(log), log)


def test_turn_endpoint_excludes_overlaps_and_reports_actual_question_count(engine):
    from app import app

    with (patch('blueprints.game.profiles.get_engine', return_value=engine),
          app.test_client() as client):
        response = client.post('/api/game/akinator/turn', json={
            'difficulty': 'easy',
            'answers': [{'id': 'years_active:gte:10', 'answer': 'yes'}],
        })
    assert response.status_code == 200
    assert response.json['done'] is False
    assert response.json['asked'] == 1
    assert not response.json['question']['id'].startswith('first_year:')
