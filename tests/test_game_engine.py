"""Tests for the Guess the Cuber engine.

The self-play tests here are the real quality gate for the feature. The engine
can only be as good as the attribute set discriminates, and there is no way to
tell whether ~33 attributes separate a few thousand cubers by reading the list —
you have to run the search against every candidate and measure.

These tests run entirely on the live `cuber_profiles` table, so they skip when
the database is unreachable (see conftest.py) or the table hasn't been built.

    python -m pytest tests/test_game_engine.py -v
"""
import random
import statistics
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.game.attributes import (ATTRIBUTES, EVENT_GROUPS, Predicate,
                                      _display, predicate_from_id)
from services.game.engine import (
    GUESS_THRESHOLD,
    LIKELIHOOD,
    MAX_QUESTIONS,
    Engine,
    answer_for,
    decode_log,
)
from services.game.profiles import RESIDENT_TIERS, load_profiles

# Quality bars. If a change to the attribute set pushes these up, the fix is
# better attributes, not a looser bar — a game that takes 30 questions to land
# has stopped being fun.
MEDIAN_QUESTIONS_MAX = 15
P95_QUESTIONS_MAX = 25

# With 10% of answers flipped, the Bayesian update should still land the right
# cuber most of the time. A hard filter would score near zero here.
NOISY_ACCURACY_MIN = 0.70


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def profiles():
    rows = load_profiles(RESIDENT_TIERS)
    if not rows:
        pytest.skip('cuber_profiles is empty — run scripts/build_cuber_profiles.py')
    return rows


@pytest.fixture(scope='module')
def easy_engine(profiles):
    rows = [r for r in profiles if r['tier'] == 1]
    if not rows:
        pytest.skip('no tier-1 profiles')
    return Engine(rows)


@pytest.fixture(scope='module')
def full_engine(profiles):
    """The Hard pool — record holders and current world top 100."""
    return Engine([r for r in profiles if r['tier'] <= 3])


@pytest.fixture(scope='module')
def everyone_engine(profiles):
    """The Everyone pool: ~52k competitors with five or more competitions."""
    return Engine(profiles)


# ---------------------------------------------------------------------------
# Unit behaviour
# ---------------------------------------------------------------------------

def _sample_predicates(attr):
    """One representative predicate per op the attribute supports."""
    if attr.kind == 'bool':
        return [Predicate(attr.key, 'is', True)]
    if attr.kind == 'numeric':
        return [Predicate(attr.key, 'gte', attr.thresholds[0])]
    if attr.kind == 'multi':
        preds = [Predicate(attr.key, 'has', '333')]
        if attr.event_valued:
            preds += [Predicate(attr.key, 'has_group', g) for g in EVENT_GROUPS]
        return preds
    return [Predicate(attr.key, 'eq', attr.values[0] if attr.values else 'X')]


def test_predicate_roundtrip_covers_every_attribute():
    """Every attribute must survive as_id() -> predicate_from_id()."""
    for attr in ATTRIBUTES:
        for pred in _sample_predicates(attr):
            assert predicate_from_id(pred.as_id()) == pred


def test_every_attribute_renders_a_question():
    """A question with an unfilled {value} placeholder would reach the player."""
    for attr in ATTRIBUTES:
        for pred in _sample_predicates(attr):
            text = pred.render()
            assert '{' not in text and text.endswith('?'), f'{attr.key}: {text}'


def test_every_question_the_engine_can_ask_reads_naturally(everyone_engine):
    """Render every predicate the real pool can generate, not just samples.

    The engine draws categorical values from the data, so a value present in the
    database but missing from a display map reaches players as a raw code. That
    is how "Is the cuber o?" shipped — WCA stores gender as m/f/o and the map
    only covered m and f.

    Runs over the *widest* pool on purpose. An earlier version used the Hard
    pool and passed while the game asked "is their strongest event mmagic?",
    because the WCA's retired events only show up once the long tail is in.
    """
    for pred in everyone_engine._predicates:
        text = pred.render()
        assert text.endswith('?'), text
        assert '{' not in text, text

        # Check the substituted value rather than picking words out of the
        # sentence — question text legitimately contains short words
        # ("...a big cube event (4x4 and up)?").
        shown = str(_display(pred.key, pred.op, pred.value))
        assert '_' not in shown, f'internal code leaked: {text!r}'
        assert len(shown) > 2 or shown.isdigit(), f'raw code in question: {text!r}'


def test_group_questions_read_naturally():
    """Group predicates must not leak the internal snake_case group key."""
    for group in EVENT_GROUPS:
        text = Predicate('top100_events', 'has_group', group).render()
        assert '_' not in text, text
    assert 'Fewest Moves' in Predicate('main_event', 'eq', '333fm').render()


def test_multi_predicates_evaluate_membership_and_groups():
    attrs = {'top100_events': ['444', '333fm']}
    assert Predicate('top100_events', 'has', '444').test(attrs)
    assert not Predicate('top100_events', 'has', '333').test(attrs)
    assert Predicate('top100_events', 'has_group', 'big_cubes').test(attrs)
    assert not Predicate('top100_events', 'has_group', 'blind_events').test(attrs)
    # An empty set must not match any group.
    assert not Predicate('top100_events', 'has_group', 'big_cubes').test(
        {'top100_events': []})


def test_dont_know_leaves_belief_unchanged(easy_engine):
    belief = easy_engine.initial_belief()
    pred_id = Predicate('has_wr', 'is', True).as_id()
    assert easy_engine.update(belief, pred_id, 'dont_know') == belief


def test_answers_move_belief_in_opposite_directions(easy_engine):
    """'yes' must favour matching candidates and 'no' must favour the rest."""
    belief = easy_engine.initial_belief()
    pred = Predicate('has_wr', 'is', True)
    idx = next(i for i, r in enumerate(easy_engine.rows) if pred.test(r['attrs']))

    after_yes = easy_engine.update(belief, pred.as_id(), 'yes')
    after_no = easy_engine.update(belief, pred.as_id(), 'no')
    assert after_yes[idx] > belief[idx] > after_no[idx]


def test_unknown_predicate_in_the_log_is_ignored(easy_engine):
    """A retired attribute in a stale client's log must not skew the belief."""
    belief = easy_engine.initial_belief()
    assert easy_engine.update(belief, 'retired_attr:is:True', 'yes') == belief


# ---------------------------------------------------------------------------
# Logical implication — never ask what an answer already settled
# ---------------------------------------------------------------------------

def _settled(engine, pred, answer):
    """Predicate ids the engine considers settled by one answer."""
    return engine.implied([(pred.as_id(), answer)])


def test_meeting_a_threshold_settles_the_weaker_ones(everyone_engine):
    """'Yes, 75+ competitions' answers '40+?' and '20+?' too."""
    settled = _settled(everyone_engine, Predicate('comp_count', 'gte', 75), 'yes')
    assert Predicate('comp_count', 'gte', 40).as_id() in settled
    assert Predicate('comp_count', 'gte', 20).as_id() in settled
    # The stronger one is still worth asking.
    assert Predicate('comp_count', 'gte', 150).as_id() not in settled


def test_missing_a_threshold_settles_the_stronger_ones(everyone_engine):
    """'No, not 40+' answers '75+?' and '150+?' too."""
    settled = _settled(everyone_engine, Predicate('comp_count', 'gte', 40), 'no')
    assert Predicate('comp_count', 'gte', 75).as_id() in settled
    assert Predicate('comp_count', 'gte', 150).as_id() in settled
    assert Predicate('comp_count', 'gte', 20).as_id() not in settled


def test_pinning_a_category_settles_every_other_value(everyone_engine):
    """The big one: 89% of redundant questions were another country or event.

    Answering "yes, they're Swiss" used to be followed by "are they German?",
    because a soft belief update leaves enough mass elsewhere that the question
    still scores as informative.
    """
    settled = _settled(
        everyone_engine, Predicate('continent_id', 'eq', '_Europe'), 'yes')
    for other in ('_Asia', '_Africa', '_North America', '_Oceania'):
        pred_id = Predicate('continent_id', 'eq', other).as_id()
        if pred_id in everyone_engine._by_id:
            assert pred_id in settled, other
    # A different attribute is untouched.
    assert not any(s.startswith('country_id:') for s in settled)


def test_ruling_out_one_category_settles_nothing_else(everyone_engine):
    """"Not European" leaves every other continent open."""
    settled = _settled(
        everyone_engine, Predicate('continent_id', 'eq', '_Europe'), 'no')
    assert Predicate('continent_id', 'eq', '_Asia').as_id() not in settled


def test_record_hierarchy_settles_downward_on_yes(everyone_engine):
    """A world record is also a continental and a national record."""
    settled = _settled(everyone_engine, Predicate('has_wr', 'is', True), 'yes')
    assert Predicate('has_cr_or_better', 'is', True).as_id() in settled
    assert Predicate('has_nr_or_better', 'is', True).as_id() in settled


def test_record_hierarchy_settles_upward_on_no(everyone_engine):
    """No national record means no continental or world record either."""
    settled = _settled(
        everyone_engine, Predicate('has_nr_or_better', 'is', True), 'no')
    assert Predicate('has_cr_or_better', 'is', True).as_id() in settled
    assert Predicate('has_wr', 'is', True).as_id() in settled


def test_event_membership_settles_its_group(everyone_engine):
    """Top 100 in 4x4 means top 100 in a big cube event."""
    settled = _settled(
        everyone_engine, Predicate('top100_events', 'has', '444'), 'yes')
    assert Predicate('top100_events', 'has_group', 'big_cubes').as_id() in settled


def test_empty_group_settles_each_member(everyone_engine):
    """No big cube ranking means none in 4x4, 5x5, 6x6 or 7x7."""
    settled = _settled(
        everyone_engine, Predicate('top100_events', 'has_group', 'big_cubes'), 'no')
    for event in EVENT_GROUPS['big_cubes']:
        pred_id = Predicate('top100_events', 'has', event).as_id()
        if pred_id in everyone_engine._by_id:
            assert pred_id in settled, event


def test_dont_know_settles_nothing(everyone_engine):
    assert _settled(
        everyone_engine, Predicate('comp_count', 'gte', 75), 'dont_know') == set()


def test_a_settled_question_is_never_asked(everyone_engine):
    """End to end: play a game and assert nothing implied comes back."""
    random.seed(99)
    rng = random.Random(99)
    secret = rng.choice(everyone_engine.rows)
    belief = everyone_engine.initial_belief()
    log: list[tuple[str, str]] = []

    for _ in range(40):
        if everyone_engine.should_guess(belief, log):
            break
        pred = everyone_engine.pick_question(belief, log)
        if pred is None:
            break
        assert pred.as_id() not in everyone_engine.implied(log), (
            f'asked "{pred.render()}", already settled by an earlier answer'
        )
        answer = answer_for(secret['attrs'], pred)
        log.append((pred.as_id(), answer))
        belief = everyone_engine.update(belief, pred.as_id(), answer)

    assert len(log) > 3, 'game ended too early to prove anything'


def test_the_cap_counts_real_questions_not_settled_ones(everyone_engine):
    """Regression: settled ids must never count toward MAX_QUESTIONS.

    Folding them into the same set the cap measures ends a game the moment that
    set passes 75 — which happens after about six real questions, since pinning
    a country settles a hundred-odd others at once.
    """
    belief = everyone_engine.initial_belief()
    # Country, not continent: the pool spans 166 countries, so pinning one down
    # settles far more questions than the cap allows. Continent only has six.
    country = next(p for p in everyone_engine._by_id.values()
                   if p.key == 'country_id')
    log = [(country.as_id(), 'yes')]
    settled = everyone_engine.implied(log)

    assert len(settled) > MAX_QUESTIONS, (
        f'this test only means something when one answer settles more than '
        f'{MAX_QUESTIONS} questions; got {len(settled)}'
    )
    assert not everyone_engine.should_guess(belief, log), (
        'the cap fired after a single question'
    )


def test_question_cap_leaves_room_past_the_old_limit():
    """The search may run well past 25 questions before it gives up."""
    assert MAX_QUESTIONS >= 75


def test_useless_questions_end_the_search(everyone_engine):
    """The search stops itself rather than running to the cap.

    This is the real stopping rule. Running to a fixed 75 instead would ask
    dozens of questions carrying ~0.0001 bits each — measurably unable to change
    the answer — and the player would sit through every one.

    Played against a truthful oracle over several secrets. `pick_question`
    samples candidates from the global RNG, so a single game's length swings by
    a wide margin run to run; the seed pins it and the assertions look at the
    set rather than one number.
    """
    random.seed(2024)
    rng = random.Random(2024)
    lengths = []
    for secret in rng.sample(everyone_engine.rows, 6):
        belief = everyone_engine.initial_belief()
        log: list[tuple[str, str]] = []
        for _ in range(MAX_QUESTIONS):
            if everyone_engine.should_guess(belief, log):
                break
            pred = everyone_engine.pick_question(belief, log)
            if pred is None:
                break
            answer = answer_for(secret['attrs'], pred)
            log.append((pred.as_id(), answer))
            belief = everyone_engine.update(belief, pred.as_id(), answer)
        lengths.append(len(log))

    print(f'\n  everyone pool question counts: {lengths}')

    assert all(n <= MAX_QUESTIONS for n in lengths), (
        f'a game ran past the {MAX_QUESTIONS}-question cap: {lengths}'
    )
    # The raised cap has to buy something: at least one game should get past
    # where the old 25-question limit would have cut it off.
    assert max(lengths) > 25, (
        f'no game exceeded 25 questions ({lengths}) — the raised cap is unused'
    )
    # And the gain floor has to do its job: most games should stop on their own
    # rather than grinding to the cap. Some secrets genuinely cannot be narrowed
    # — that is what the cap is for — but it should be the exception.
    stopped_early = sum(1 for n in lengths if n < MAX_QUESTIONS)
    assert stopped_early > len(lengths) // 2, (
        f'only {stopped_early}/{len(lengths)} games stopped before the cap '
        f'({lengths}) — MIN_QUESTION_GAIN is too low to be doing its job'
    )


def test_gain_floor_does_not_cut_the_curated_pools_short(easy_engine):
    """The floor must only bite on the long tail, never on a winnable game."""
    rng = random.Random(3)
    for secret in rng.sample(easy_engine.rows, 20):
        found, asked = _play(easy_engine, secret, rng=rng)
        assert found, f'{secret["name"]} lost after {asked} questions'


def test_sampling_still_picks_a_splitting_question(full_engine):
    """On a pool past the sample size the choice must still be informative."""
    belief = full_engine.initial_belief()
    pred = full_engine.pick_question(belief, [])
    assert pred is not None
    hits = sum(1 for r in full_engine.rows if pred.test(r['attrs']))
    share = hits / len(full_engine.rows)
    assert 0.05 < share < 0.95, f'{pred.render()} splits {share:.1%}'


def test_contradictory_answers_reset_rather_than_crash(easy_engine):
    """Answering both ways about the same fact must not produce a dead belief."""
    belief = easy_engine.initial_belief()
    pred_id = Predicate('has_wr', 'is', True).as_id()
    for _ in range(80):
        belief = easy_engine.update(belief, pred_id, 'yes')
        belief = easy_engine.update(belief, pred_id, 'no')
    assert abs(sum(belief) - 1.0) < 1e-6
    assert all(w >= 0 for w in belief)


def test_belief_stays_normalized(easy_engine):
    belief = easy_engine.initial_belief()
    log = []
    for _ in range(10):
        pred = easy_engine.pick_question(belief, log)
        if pred is None:
            break
        answer = random.choice(('yes', 'no'))
        log.append((pred.as_id(), answer))
        belief = easy_engine.update(belief, pred.as_id(), answer)
        assert abs(sum(belief) - 1.0) < 1e-6


def test_picked_question_is_never_repeated(easy_engine):
    belief = easy_engine.initial_belief()
    log = []
    for _ in range(15):
        pred = easy_engine.pick_question(belief, log)
        if pred is None:
            break
        assert pred.as_id() not in {p for p, _ in log}
        log.append((pred.as_id(), 'yes'))
        belief = easy_engine.update(belief, pred.as_id(), 'yes')


def test_decode_log_drops_untrusted_junk():
    """The answer log arrives from the browser, so it must be filtered."""
    good = Predicate('has_wr', 'is', True).as_id()
    log = decode_log([
        {'id': good, 'answer': 'yes'},
        {'id': good, 'answer': 'nonsense'},        # bad answer
        {'id': 'retired_attr:is:True', 'answer': 'yes'},  # unknown attribute
        {'id': 'malformed', 'answer': 'yes'},
        'not-a-dict',
        None,
    ])
    assert log == [(good, 'yes')]


def test_replay_matches_incremental_updates(easy_engine):
    """Stateless replay must reproduce turn-by-turn state exactly."""
    pairs = [
        (Predicate('has_wr', 'is', True).as_id(), 'yes'),
        (Predicate('is_active', 'is', True).as_id(), 'no'),
        (Predicate('comp_count', 'gte', 50).as_id(), 'probably'),
    ]
    incremental = easy_engine.initial_belief()
    for pred_id, answer in pairs:
        incremental = easy_engine.update(incremental, pred_id, answer)

    replayed = easy_engine.replay(pairs)
    assert replayed == pytest.approx(incremental)


# ---------------------------------------------------------------------------
# Self-play — the quality gate
# ---------------------------------------------------------------------------

def _play(engine, secret, noise=0.0, rng=None):
    """Play one full game against a known secret. Returns (found, questions)."""
    rng = rng or random.Random(0)
    belief = engine.initial_belief()
    log: list[tuple[str, str]] = []

    for turn in range(MAX_QUESTIONS):
        if engine.should_guess(belief, log):
            break
        pred = engine.pick_question(belief, log)
        if pred is None:
            break

        answer = answer_for(secret['attrs'], pred)
        if noise and rng.random() < noise:
            answer = 'no' if answer == 'yes' else 'yes'
        log.append((pred.as_id(), answer))
        belief = engine.update(belief, pred.as_id(), answer)

    top = engine.top_candidates(belief, 1)[0][0]
    return top['wca_id'] == secret['wca_id'], len(log)


def test_self_play_converges_on_easy_tier(easy_engine):
    """Against a perfect oracle the engine should find everyone, quickly."""
    rng = random.Random(1234)
    sample = rng.sample(easy_engine.rows, min(120, len(easy_engine.rows)))

    results = [_play(easy_engine, s, rng=rng) for s in sample]
    found = [r for r, _ in results]
    counts = [q for _, q in results]

    accuracy = sum(found) / len(found)
    median = statistics.median(counts)
    p95 = sorted(counts)[int(len(counts) * 0.95) - 1]

    print(f'\n  easy tier: accuracy {accuracy:.1%}, '
          f'median {median} questions, p95 {p95}')

    assert accuracy > 0.95, f'engine only found {accuracy:.1%} with perfect answers'
    assert median <= MEDIAN_QUESTIONS_MAX
    assert p95 <= P95_QUESTIONS_MAX


def test_everyone_pool_stays_responsive(everyone_engine):
    """The largest pool must answer a turn fast enough to play.

    Question selection is sampled, so this should hold roughly flat as the pool
    grows; without sampling it ran to tens of seconds.
    """
    import time
    belief = everyone_engine.initial_belief()
    t = time.time()
    pred = everyone_engine.pick_question(belief, [])
    pick_ms = (time.time() - t) * 1000
    t = time.time()
    everyone_engine.update(belief, pred.as_id(), 'yes')
    update_ms = (time.time() - t) * 1000

    print(f'\n  everyone pool ({len(everyone_engine.rows):,}): '
          f'pick {pick_ms:.0f}ms, update {update_ms:.0f}ms')
    assert pick_ms < 2000, f'question selection took {pick_ms:.0f}ms'
    assert update_ms < 500, f'belief update took {update_ms:.0f}ms'


def test_self_play_on_everyone_pool(everyone_engine):
    """Self-play against the ~52k pool, where stalling is expected.

    The bar is deliberately lower than the curated tiers: with 52k candidates
    the search will sometimes run out of separating questions, which is exactly
    why the caller offers a shortlist rather than naming one name.
    """
    rng = random.Random(31)
    sample = rng.sample(everyone_engine.rows, 25)
    results = [_play(everyone_engine, s, rng=rng) for s in sample]
    accuracy = sum(r for r, _ in results) / len(results)
    median = statistics.median([q for _, q in results])
    print(f'\n  everyone pool ({len(everyone_engine.rows):,}): '
          f'accuracy {accuracy:.0%}, median {median} questions')
    assert accuracy > 0.4, (
        f'only {accuracy:.0%} found on the Everyone pool — worse than expected '
        f'even allowing for undifferentiated competitors'
    )


def test_self_play_on_full_pool(full_engine):
    """The hard tier is where the attribute set is most likely to run out."""
    rng = random.Random(99)
    sample = rng.sample(full_engine.rows, min(60, len(full_engine.rows)))

    results = [_play(full_engine, s, rng=rng) for s in sample]
    accuracy = sum(r for r, _ in results) / len(results)
    median = statistics.median([q for _, q in results])

    print(f'\n  full pool ({len(full_engine.rows):,}): '
          f'accuracy {accuracy:.1%}, median {median} questions')

    assert accuracy > 0.85, (
        f'only {accuracy:.1%} found on the full pool — the attribute set does '
        f'not discriminate well enough at tier 3'
    )


def test_bayesian_update_tolerates_wrong_answers(easy_engine):
    """The reason beliefs beat hard filtering: recovery from a bad answer."""
    rng = random.Random(7)
    sample = rng.sample(easy_engine.rows, min(80, len(easy_engine.rows)))

    found = [_play(easy_engine, s, noise=0.10, rng=rng)[0] for s in sample]
    accuracy = sum(found) / len(found)
    print(f'\n  with 10% wrong answers: accuracy {accuracy:.1%}')

    assert accuracy >= NOISY_ACCURACY_MIN, (
        f'accuracy collapsed to {accuracy:.1%} with 10% noise — the belief '
        f'update is not recovering from wrong answers'
    )


def test_guess_threshold_is_the_documented_bar():
    """The app names a candidate outright only once it clears 90%."""
    assert GUESS_THRESHOLD == 0.90


def test_confidence_decides_naming_versus_shortlist(easy_engine):
    """`is_confident` must agree with the threshold at both ends."""
    engine = easy_engine
    n = len(engine.rows)

    flat = engine.initial_belief()
    assert not engine.is_confident(flat), 'a uniform belief is never confident'

    certain = [0.0] * n
    certain[0] = 1.0
    assert engine.is_confident(certain)

    just_under = [(1 - GUESS_THRESHOLD + 0.01) / (n - 1)] * n
    just_under[0] = GUESS_THRESHOLD - 0.01
    assert not engine.is_confident(just_under)

    just_over = list(just_under)
    just_over[0] = GUESS_THRESHOLD + 0.01
    assert engine.is_confident(just_over)


def test_an_outright_guess_is_reached_by_confidence(easy_engine):
    """Games on a curated pool should end by clearing the bar, not by giving up.

    If this starts failing, the search is running out of questions before it
    gets confident — which would mean the shortlist is doing work the engine
    should be doing itself.
    """
    rng = random.Random(12)
    confident = 0
    sample = rng.sample(easy_engine.rows, 20)
    for secret in sample:
        belief = easy_engine.initial_belief()
        log: list[tuple[str, str]] = []
        for _ in range(MAX_QUESTIONS):
            if easy_engine.should_guess(belief, log):
                break
            pred = easy_engine.pick_question(belief, log)
            if pred is None:
                break
            answer = answer_for(secret['attrs'], pred)
            log.append((pred.as_id(), answer))
            belief = easy_engine.update(belief, pred.as_id(), answer)
        confident += easy_engine.is_confident(belief)

    assert confident == len(sample), (
        f'only {confident}/{len(sample)} easy-tier games reached {GUESS_THRESHOLD:.0%}'
    )


def test_likelihoods_are_symmetric_and_sane():
    """A malformed likelihood table would skew every game subtly."""
    for answer, lik in LIKELIHOOD.items():
        assert abs(lik[True] + lik[False] - 1.0) < 1e-9, answer
        assert 0 < lik[True] < 1 and 0 < lik[False] < 1, answer
    assert LIKELIHOOD['dont_know'][True] == 0.5
    assert LIKELIHOOD['yes'][True] > LIKELIHOOD['probably'][True] > 0.5
    assert GUESS_THRESHOLD < 1.0
