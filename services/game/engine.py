"""The guessing engine: Bayesian candidate filter + information-gain question picker.

No database, no network, no model — this is arithmetic over the precomputed
matrix, which is why Mode 2 costs nothing per turn.

Two design choices worth understanding before changing anything here:

**Beliefs, not filters.** An answer multiplies candidate weights rather than
eliminating candidates. A player who misremembers one fact ("did they compete
before 2010?" — most people genuinely don't know) would make a hard-filtered
game unwinnable, and the failure is invisible: the engine confidently converges
on the wrong person with no way back. Weighting lets contrary evidence
accumulate and pull the right answer back to the top.

**Stateless replay.** The engine never holds a session. `replay()` rebuilds the
full belief state from an answer log on every request. That matters because the
web app runs two Gunicorn workers with no sticky sessions and the Fly machine
suspends when idle — in-process game state would silently break between turns.
"""
from __future__ import annotations

import math
import random

from .attributes import (ATTRIBUTES, BY_KEY, EVENT_GROUPS, Predicate,
                         candidate_predicates, predicate_from_id)


def _closure(start: str, edges: dict[str, set[str]]) -> set[str]:
    """Everything reachable from `start`, excluding itself."""
    seen: set[str] = set()
    stack = list(edges.get(start, ()))
    while stack:
        key = stack.pop()
        if key in seen or key == start:
            continue
        seen.add(key)
        stack.extend(edges.get(key, ()))
    return seen


# Declared boolean implications, forward and reverse. Built once: the schema is
# fixed at import, and walking it per question would be wasted work.
_IMPLIES: dict[str, set[str]] = {
    a.key: set(a.implies) for a in ATTRIBUTES if a.implies
}
_IMPLIED_BY: dict[str, set[str]] = {}
for _key, _targets in _IMPLIES.items():
    for _t in _targets:
        _IMPLIED_BY.setdefault(_t, set()).add(_key)

_implies_closure = lambda key: _closure(key, _IMPLIES)        # noqa: E731
_implied_by_closure = lambda key: _closure(key, _IMPLIED_BY)  # noqa: E731

# Candidates sampled when scoring questions on a large pool. 3,000 keeps the
# estimate tight while bounding a turn to a few hundred thousand predicate
# evaluations (~35 ms) no matter how big the pool gets.
SAMPLE_SIZE = 3000

# P(player gives this answer | the attribute is actually True / False).
#
# The asymmetry matters: 'yes' and 'no' are deliberately not 1.0/0.0, so a
# single wrong answer damps a candidate rather than killing it. 'dont_know'
# is exactly 0.5/0.5, which leaves the belief untouched — the question is
# simply skipped, but it stays on the asked list so it isn't offered again.
LIKELIHOOD: dict[str, dict[bool, float]] = {
    'yes':          {True: 0.92, False: 0.08},
    'probably':     {True: 0.75, False: 0.25},
    'dont_know':    {True: 0.50, False: 0.50},
    'probably_not': {True: 0.25, False: 0.75},
    'no':           {True: 0.08, False: 0.92},
}

ANSWERS = tuple(LIKELIHOOD)

# Stop asking and name a candidate outright once it holds this much of the
# probability mass. Below it the caller offers a shortlist instead.
#
# Measured across Easy, Hard and Everyone, every outright guess at 0.85 was
# already correct, so this is headroom rather than a fix for wrong guesses. It
# costs about two extra questions on the largest pool and none on the curated
# ones, where the belief jumps clear of any of these thresholds in a single
# answer. Raising it further does start to cost: at 0.95 the Everyone pool needs
# ~9 more questions and pushes a game that would have been named into the
# shortlist instead.
GUESS_THRESHOLD = 0.90

# Hard cap on questions before the engine must guess anyway.
MAX_QUESTIONS = 75

# Stop early when the best remaining question carries less than this much
# information, in bits.
#
# The cap alone is not a good stopping rule. Gain collapses as the search
# narrows — measured on the ~52k pool, the median question is worth 0.60 bits at
# Q1, 0.26 at Q15, 0.02 at Q25, 0.001 at Q40 and 0.0001 by Q74 — so running to
# a fixed 75 means asking dozens of questions that cannot change the answer.
# This floor is what "or until it's narrowed down" actually means: keep going
# while questions still separate candidates, and stop when they don't.
#
# Set low deliberately. On the curated pools games converge on confidence long
# before it applies; it exists to cut the dead tail on the largest pool, not to
# end games early.
MIN_QUESTION_GAIN = 0.001

# Candidates below this share of the leader's weight stop being evaluated.
# Purely a speed measure — at 1e-6 a candidate would need several confidently
# contradicted answers to fall this far, and could not realistically climb back.
_PRUNE_RATIO = 1e-6


class Engine:
    """Wraps one candidate pool. Cheap to construct; safe to cache per tier."""

    def __init__(self, rows: list[dict]):
        """`rows` are `cuber_profiles` records: {wca_id, name, attrs, ...}."""
        self.rows = rows
        # Hoisted so the hot loops index a flat list instead of doing a dict
        # lookup per row.
        self._attrs = [r['attrs'] for r in rows]
        self._predicates = candidate_predicates(rows)
        self._by_id = {p.as_id(): p for p in self._predicates}

    # Predicates are evaluated on demand rather than indexed up front.
    #
    # A precomputed {predicate -> matching row indices} map is the obvious
    # design and it does not scale: at ~52k candidates those sets cost ~140 MB,
    # against ~62 MB for the rows themselves, on a 512 MB machine running two
    # workers. Predicate.test() costs 0.06 us, so re-deriving is cheap — a full
    # belief update over 52k candidates is about 3 ms.
    def _test(self, pred: Predicate, index: int) -> bool:
        return pred.test(self._attrs[index])

    def _resolve(self, pred_id: str) -> Predicate | None:
        """Predicate for an id, including ones this pool would never generate.

        The answer log comes from the browser, so it can name a predicate that
        is valid but outside `_by_id` — for instance after the candidate set
        narrowed and a categorical value stopped being offered.
        """
        pred = self._by_id.get(pred_id)
        if pred is not None:
            return pred
        try:
            return predicate_from_id(pred_id)
        except (ValueError, KeyError):
            return None

    # -- belief state ------------------------------------------------------

    def initial_belief(self) -> list[float]:
        n = len(self.rows)
        return [1.0 / n] * n if n else []

    def update(self, belief: list[float], pred_id: str, answer: str) -> list[float]:
        """Apply one answer. Returns a new normalized belief."""
        if answer not in LIKELIHOOD:
            raise ValueError(f'unknown answer: {answer}')
        if answer == 'dont_know':
            return belief  # 0.5/0.5 is a no-op; skip the work

        pred = self._resolve(pred_id)
        if pred is None:
            return belief  # unknown predicate carries no information

        lik = LIKELIHOOD[answer]
        attrs = self._attrs
        updated = [
            w * (lik[True] if pred.test(attrs[i]) else lik[False])
            for i, w in enumerate(belief)
        ]
        total = sum(updated)
        if total <= 0:
            # Every candidate was contradicted into oblivion. Rather than crash
            # or return a degenerate all-zero belief, fall back to uniform — the
            # player has answered inconsistently and the honest thing is to
            # restart the search rather than pretend to a confident answer.
            return self.initial_belief()
        return [w / total for w in updated]

    def reject(self, belief: list[float], wca_ids) -> list[float]:
        """Zero out candidates the player has already rejected by name.

        Unlike an answer, a rejected guess is certain — the player has told us
        outright it isn't that person — so this eliminates rather than damps.
        """
        rejected = set(wca_ids or ())
        if not rejected:
            return belief
        updated = [
            0.0 if self.rows[i]['wca_id'] in rejected else w
            for i, w in enumerate(belief)
        ]
        total = sum(updated)
        if total <= 0:
            # Every remaining candidate has been rejected. Start over rather
            # than hand back a dead belief the caller can't reason about.
            return self.initial_belief()
        return [w / total for w in updated]

    def replay(
        self, answers: list[tuple[str, str]], rejected=None
    ) -> list[float]:
        """Rebuild belief from an answer log of (predicate_id, answer) pairs."""
        belief = self.initial_belief()
        for pred_id, answer in answers:
            belief = self.update(belief, pred_id, answer)
        return self.reject(belief, rejected)

    # -- logical implication ----------------------------------------------

    def implied(self, answers: list[tuple[str, str]]) -> set[str]:
        """Questions whose answer the log already settles.

        Information gain alone does not catch these, because the belief update
        is soft: answering "yes, 75+ competitions" leaves ~44% of the
        probability mass on candidates with fewer, so "20+ competitions?" still
        scores as informative even though its answer is certain. The player just
        sees the game asking something they have already told it.

        Measured before this existed, 42% of questions on the largest pool were
        logically settled by an earlier answer — nearly all of them another
        value of a categorical the player had already pinned down ("they're
        Swiss" followed by "are they German?").

        'probably' counts as a yes here. Strictly it leaves room, but re-asking
        a weaker form of a question someone has already leaned on reads as
        broken rather than thorough.
        """
        settled: set[str] = set()
        for pred_id, answer in answers:
            if answer == 'dont_know':
                continue
            pred = self._resolve(pred_id)
            if pred is None:
                continue
            settled |= self._entailed(pred, answer in ('yes', 'probably'))
        return settled

    def _entailed(self, pred: Predicate, positive: bool) -> set[str]:
        """Every predicate id settled by one answer."""
        out: set[str] = set()
        attr = BY_KEY.get(pred.key)

        for other_id, other in self._by_id.items():
            if other_id == pred.as_id():
                continue

            if other.key == pred.key:
                # A threshold weaker than one already met, or stronger than one
                # already missed, has a known answer.
                if pred.op == 'gte' and other.op == 'gte':
                    if positive and other.value <= pred.value:
                        out.add(other_id)
                    elif not positive and other.value >= pred.value:
                        out.add(other_id)
                # Categorical values are mutually exclusive: pinning one down
                # rules out every other. This is the big one.
                elif pred.op == 'eq' and other.op == 'eq' and positive:
                    out.add(other_id)
                # Holding a ranking in one event settles its whole group.
                elif pred.op == 'has' and other.op == 'has_group' and positive:
                    if pred.value in EVENT_GROUPS.get(str(other.value), ()):
                        out.add(other_id)
                # Holding none in a group settles every event inside it.
                elif pred.op == 'has_group' and other.op == 'has' and not positive:
                    if other.value in EVENT_GROUPS.get(str(pred.value), ()):
                        out.add(other_id)

            elif pred.op == 'is' and other.op == 'is' and attr is not None:
                # Declared boolean hierarchy, walked transitively. A yes settles
                # everything it implies; a no settles everything that implies it.
                chain = (_implies_closure(pred.key) if positive
                         else _implied_by_closure(pred.key))
                if other.key in chain:
                    out.add(other_id)

        return out

    # -- question selection ------------------------------------------------

    def pick_question(
        self, belief: list[float], answers: list[tuple[str, str]]
    ) -> Predicate | None:
        """The best unasked predicate, by expected information gain.

        Takes the answer log rather than a set of asked ids, so the questions an
        answer already settles are excluded here rather than by each caller.
        Passing a pre-built set was easy to get wrong in a way that looked like
        an improvement: fold the settled ids into the same set `should_guess`
        counts and games end the moment that set passes MAX_QUESTIONS, having
        asked six real questions.

        Scored on a random sample once the candidate set is large. Scoring every
        predicate against every candidate is O(predicates x candidates) and runs
        to tens of seconds on the biggest pools; a sample of a few thousand
        picks the same question almost always, and when it doesn't the cost is a
        fraction of one extra question rather than a wrong answer.
        """
        live = self._live(belief)
        if len(live) <= 1:
            return None

        asked = {pred_id for pred_id, _ in answers} | self.implied(answers)

        pool = (random.sample(live, SAMPLE_SIZE)
                if len(live) > SAMPLE_SIZE else live)

        weights = [belief[i] for i in pool]
        mass = sum(weights)
        if mass <= 0:
            return None
        norm = [w / mass for w in weights]
        base = _entropy(norm)

        best: tuple[float, str] | None = None
        attrs = self._attrs

        for pred_id, pred in self._by_id.items():
            if pred_id in asked:
                continue
            hits = [pred.test(attrs[i]) for i in pool]
            # A predicate that splits nothing carries no information.
            first = hits[0]
            if all(h == first for h in hits):
                continue

            gain = base - self._expected_entropy(norm, hits)
            if best is None or gain > best[0]:
                best = (gain, pred_id)

        if best is None or best[0] < MIN_QUESTION_GAIN:
            # Nothing left that meaningfully separates the candidates. Returning
            # None here is what makes the caller offer a shortlist rather than
            # grind through questions whose answers change nothing.
            return None
        return self._by_id[best[1]]

    @staticmethod
    def _expected_entropy(norm: list[float], hits: list[bool]) -> float:
        """Entropy after this question, averaged over the answers we might get.

        Only the crisp yes/no branches are modelled here. The soft answers are
        mixtures of the same two and don't change which question ranks highest,
        so including them would cost time without changing the choice.
        """
        total = 0.0
        for answer in ('yes', 'no'):
            lik = LIKELIHOOD[answer]
            weights = [w * lik[h] for w, h in zip(norm, hits)]
            mass = sum(weights)
            if mass <= 0:
                continue
            total += mass * _entropy(w / mass for w in weights)
        return total

    def _live(self, belief: list[float]) -> list[int]:
        """Indices still carrying meaningful probability mass."""
        if not belief:
            return []
        floor = max(belief) * _PRUNE_RATIO
        return [i for i, w in enumerate(belief) if w > floor]

    # -- guessing ----------------------------------------------------------

    def top_candidates(self, belief: list[float], n: int = 5) -> list[tuple[dict, float]]:
        ranked = sorted(
            range(len(belief)), key=lambda i: belief[i], reverse=True
        )[:n]
        return [(self.rows[i], belief[i]) for i in ranked]

    def should_guess(self, belief: list[float],
                     answers: list[tuple[str, str]]) -> bool:
        """Whether to stop asking.

        The cap counts questions actually put to the player — `len(answers)` —
        never the wider set of questions their answers have settled, which runs
        to well over a hundred and would end a game after six real questions.
        """
        if not belief:
            return False
        if len(self._live(belief)) <= 1:
            return True
        if len(answers) >= MAX_QUESTIONS:
            return True
        return max(belief) >= GUESS_THRESHOLD

    def is_confident(self, belief: list[float]) -> bool:
        """Whether the leader is strong enough to name outright.

        When it isn't, the caller should offer a shortlist instead. On a large
        pool the search regularly runs out of separating questions with a
        cluster of near-identical candidates still standing — over half of WCA
        competitors have been to exactly one competition and look alike in every
        attribute — and naming one of them confidently is just a confident wrong
        answer.
        """
        return bool(belief) and max(belief) >= GUESS_THRESHOLD

    def pick_secret(self, rng: random.Random | None = None) -> dict:
        """Choose a cuber for the app to hold in Modes 1 and 3."""
        return (rng or random).choice(self.rows)


def _entropy(weights) -> float:
    """Shannon entropy in bits of an already-normalized weight sequence."""
    return -sum(w * math.log2(w) for w in weights if w > 0)


def answer_for(secret_attrs: dict, pred: Predicate) -> str:
    """The truthful answer about a known cuber — used in Modes 1 and 3.

    The app answers from data, so it only ever says 'yes' or 'no'; the softer
    answers exist for humans in Mode 2.
    """
    return 'yes' if pred.test(secret_attrs) else 'no'


def decode_log(raw: list) -> list[tuple[str, str]]:
    """Validate a client-supplied answer log.

    The log arrives from the browser on every turn, so it is untrusted input.
    Unknown predicate ids and unknown answers are dropped rather than raising —
    a stale client (one left open across a deploy that retired an attribute)
    should degrade to a slightly worse game, not a 500.
    """
    clean: list[tuple[str, str]] = []
    for entry in raw or []:
        try:
            pred_id, answer = entry['id'], entry['answer']
        except (TypeError, KeyError):
            continue
        if answer not in LIKELIHOOD:
            continue
        try:
            predicate_from_id(pred_id)
        except (ValueError, KeyError):
            continue
        clean.append((pred_id, answer))
    return clean
