"""Offline regression checks for scoring correctness and large-pool work limits."""
import math
import random
from unittest.mock import patch

import pytest

from services.game.attributes import ATTRIBUTES, EVENT_NAMES, Predicate
from services.game.engine import Engine, LIKELIHOOD, MIN_QUESTION_GAIN, SAMPLE_SIZE


def posterior_gain(weights, hits):
    """Independent reference: construct both full posterior distributions."""
    total = sum(weights)
    norm = [w / total for w in weights]

    def entropy(values):
        return -sum(w * math.log2(w) for w in values if w > 0)

    expected = 0.0
    for answer in ('yes', 'no'):
        posterior = [w * LIKELIHOOD[answer][hit] for w, hit in zip(norm, hits)]
        mass = sum(posterior)
        if mass:
            expected += mass * entropy(w / mass for w in posterior)
    return entropy(norm) - expected


def sample_rows(count):
    rng = random.Random(182)
    events = tuple(EVENT_NAMES)
    values = {
        attr.key: ([0, *attr.thresholds] if attr.kind == 'numeric' else
                   list(attr.values) if attr.values else
                   [f'country-{i}' for i in range(160)] if attr.key == 'country_id' else list(events))
        for attr in ATTRIBUTES
    }
    rows = []
    for i in range(count):
        attrs = {}
        for attr in ATTRIBUTES:
            attrs[attr.key] = (rng.random() < 0.2 if attr.kind == 'bool' else
                               tuple(rng.sample(events, rng.randrange(4))) if attr.kind == 'multi' else
                               rng.choice(values[attr.key]))
        rows.append({'wca_id': f'test-{i}', 'name': f'Test {i}', 'attrs': attrs})
    return rows


@pytest.mark.parametrize('size', [2, 17, 3000])
def test_partition_score_equals_full_posterior_entropy(size):
    rng = random.Random(size)
    for _ in range(12):
        weights = [10 ** rng.uniform(-9, 0) for _ in range(size)]
        hits = [rng.choice((True, False)) for _ in weights]
        true_mass = sum(w for w, hit in zip(weights, hits) if hit) / sum(weights)
        assert Engine._information_gain(true_mass) == pytest.approx(
            posterior_gain(weights, hits), abs=1e-12)


def test_score_keeps_the_likelihood_model_instead_of_assuming_a_perfect_answer(monkeypatch):
    # Also guards against assuming the noise must be symmetric if it is tuned.
    monkeypatch.setitem(LIKELIHOOD, 'yes', {True: 0.8, False: 0.1})
    monkeypatch.setitem(LIKELIHOOD, 'no', {True: 0.2, False: 0.9})
    weights, hits = [0.2, 0.3, 0.5], [True, True, False]
    assert Engine._information_gain(0.5) == pytest.approx(posterior_gain(weights, hits))
    assert Engine._information_gain(0.5) < 1


@pytest.mark.parametrize('mass', [0, 1, -1e-15, 1 + 1e-15])
def test_unsplit_or_rounded_partitions_have_zero_gain(mass):
    assert Engine._information_gain(mass) == 0


def test_grouping_preserves_all_predicate_types_and_missing_values():
    rows = sample_rows(200)
    engine = Engine(rows)
    weighted = [(row['attrs'], (i + 1) / 20100) for i, row in enumerate(rows)]
    weighted += [({}, 0.2), ({attr.key: None for attr in ATTRIBUTES}, 0.2)]
    assert {p.op for p in engine._predicates} == {'eq', 'is', 'gte', 'has', 'has_group'}
    for pred in engine._predicates:
        grouped = Engine._group_weights(weighted, pred.key)
        actual = sum(w for attrs, w in grouped if pred.test(attrs))
        expected = sum(w for attrs, w in weighted if pred.test(attrs))
        assert actual == pytest.approx(expected, abs=1e-12), pred


@pytest.mark.parametrize('values', [
    [['333', '444'], ('333', '444'), {'333', '444'}, [], None],
    [5, '5', 'bad number', {}, [1]],
])
def test_grouping_keeps_list_and_unexpected_value_handling(values):
    pred = (Predicate('top100_events', 'has_group', 'big_cubes') if isinstance(values[0], list)
            else Predicate('comp_count', 'gte', 5))
    weighted = [({pred.key: value}, 1 / len(values)) for value in values]
    grouped = Engine._group_weights(weighted, pred.key)
    assert sum(w for attrs, w in grouped if pred.test(attrs)) == pytest.approx(
        sum(w for attrs, w in weighted if pred.test(attrs)))


def test_picks_a_maximum_gain_question_after_weighted_answers_and_skips():
    engine = Engine(sample_rows(90))
    answers = [(Predicate('first_year', 'gte', 2019).as_id(), 'yes'),
               (Predicate('gender', 'eq', 'm').as_id(), 'dont_know'),
               (Predicate('has_wr', 'is', True).as_id(), 'probably_not')]
    belief = engine.replay(answers)
    excluded = engine.implied(answers) | {p for p, _ in answers}
    scores = {pid: posterior_gain(belief, [p.test(r['attrs']) for r in engine.rows])
              for pid, p in engine._by_id.items() if pid not in excluded}
    picked = engine.pick_question(belief, answers)
    assert picked.as_id() not in excluded
    # Equally good questions may exchange order through floating-point roundoff.
    assert scores[picked.as_id()] == pytest.approx(max(scores.values()), abs=1e-12)


@pytest.mark.parametrize('minority', [0.0001, 0.001])
def test_low_information_stopping_rule_is_preserved(minority):
    engine = Engine([{'attrs': {'has_wr': True}}, {'attrs': {'has_wr': False}}])
    belief = [minority, 1 - minority]
    gain = posterior_gain(belief, [True, False])
    assert (engine.pick_question(belief, []) is None) == (gain < MIN_QUESTION_GAIN)


def test_empty_or_identical_candidates_stop_without_a_question():
    for rows in ([], [{'attrs': {}}], [{'attrs': {}}, {'attrs': {}}]):
        engine = Engine(rows)
        assert engine.pick_question(engine.initial_belief(), []) is None


def test_large_pool_scoring_avoids_per_candidate_entropy_and_keeps_sample_limit():
    # Repeat varied profiles to reach Everyone's scale without a database fixture.
    rows = sample_rows(520) * 100
    engine = Engine(rows)
    with (patch('services.game.engine.math.log2', wraps=math.log2) as logarithm,
          patch.object(Engine, '_group_weights', wraps=Engine._group_weights) as group):
        picked = engine.pick_question(engine.initial_belief(), [])
    assert picked is not None
    assert logarithm.call_count <= 6 * len(engine._predicates)
    assert group.call_count <= len(ATTRIBUTES)
    assert all(len(call.args[0]) <= SAMPLE_SIZE for call in group.call_args_list)
    share = sum(picked.test(row['attrs']) for row in rows) / len(rows)
    assert 0.05 < share < 0.95


def test_turn_stage_logs_elapsed_and_cpu_time_even_on_worker_abort(caplog):
    from blueprints.game import _time_turn_stage
    with caplog.at_level('INFO', logger='blueprints.game'):
        with pytest.raises(SystemExit):
            with _time_turn_stage('pick', 'everyone', 52000, 7):
                raise SystemExit(1)
    assert 'stage=pick difficulty=everyone candidates=52000 answers=7' in caplog.text
    assert 'wall_ms=' in caplog.text and 'cpu_ms=' in caplog.text


def test_everyone_turn_returns_questions_and_keeps_answer_replay(caplog):
    from app import app
    engine = Engine(sample_rows(100))
    with (patch('blueprints.game.profiles.get_engine', return_value=engine),
          caplog.at_level('INFO', logger='blueprints.game'),
          app.test_client() as client):
        first = client.post('/api/game/akinator/turn', json={
            'difficulty': 'everyone', 'answers': [],
        })
        assert first.status_code == 200
        first_question = first.json['question']['id']
        second = client.post('/api/game/akinator/turn', json={
            'difficulty': 'everyone',
            'answers': [{'id': first_question, 'answer': 'dont_know'}],
        })
        assert second.status_code == 200
        assert second.json['asked'] == 1
        assert second.json['question']['id'] != first_question
        assert second.json['remaining'] == 100
    for stage in ('load', 'replay', 'pick'):
        assert f'stage={stage} difficulty=everyone' in caplog.text
