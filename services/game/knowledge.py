"""What an answer log settles, including questions nobody asked.

The engine's belief update is soft on purpose (see engine.py), so the belief
alone can't tell that a question's answer is already known. After "they
represent the USA", about a quarter of the probability mass still sits on
non-Americans, so "are they from North America?" scores as informative. The
player sees the game asking something they just told it.

This module keeps a hard record next to the soft belief. Every answer becomes a
known truth value, and the relations in attributes.RELATIONS carry it to other
questions by unit propagation: whenever all literals of a clause but one are
false, the remaining one must be true. Chains follow automatically. USA settles
North America, which settles "not European", which settles every European
country.

Two kinds of reasoning are built in here rather than written as clauses,
because a clause per pair would mean thousands of them:

    thresholds    comp_count >= 75 settles >= 40 (true) and, if false, >= 150
    categories    pinning one country (or event, or continent) rules out the rest

This record only decides which questions are worth asking. It never touches the
belief, so a player who misremembers one fact can still be brought back by the
other answers. It is rebuilt from the log on every turn, like the belief.
"""
from __future__ import annotations

from collections.abc import Iterable

from .attributes import Clause, Predicate


def _canonical(pred: Predicate) -> tuple[Predicate, bool]:
    """Store booleans as `is True` so `is False` can't become a second fact."""
    if pred.op == 'is' and pred.value is not True:
        return Predicate(pred.key, 'is', True), True
    return pred, False


class ClauseIndex:
    """Relations indexed by the attributes they mention. Built once per pool."""

    def __init__(self, clauses: Iterable[Clause]):
        self.by_key: dict[str, list[Clause]] = {}
        for clause in clauses:
            canonical = []
            for pred, truth in clause:
                pred, flipped = _canonical(pred)
                canonical.append((pred, truth != flipped))
            clause = tuple(canonical)
            for key in {pred.key for pred, _ in clause}:
                self.by_key.setdefault(key, []).append(clause)


class Knowledge:
    """Truth values the answers settle, with every relation applied."""

    def __init__(self, index: ClauseIndex):
        self._index = index
        self._facts: dict[Predicate, bool] = {}
        # Per-attribute summaries that answer whole families of questions:
        # the highest threshold known met, the lowest known missed, and the one
        # categorical value known to hold.
        self._met: dict[str, float] = {}
        self._missed: dict[str, float] = {}
        self._pinned: dict[str, object] = {}

    def value(self, pred: Predicate) -> bool | None:
        """True or False if the answers settle `pred`, None if it's still open."""
        pred, flipped = _canonical(pred)
        known = self._facts.get(pred)
        if known is None:
            known = self._derived(pred)
        if known is None:
            return None
        return known != flipped

    def _derived(self, pred: Predicate) -> bool | None:
        if pred.op == 'gte':
            met = self._met.get(pred.key)
            if met is not None and pred.value <= met:
                return True
            missed = self._missed.get(pred.key)
            if missed is not None and pred.value >= missed:
                return False
        elif pred.op == 'eq' and pred.key in self._pinned:
            return self._pinned[pred.key] == pred.value
        return None

    def add(self, pred: Predicate, truth: bool) -> None:
        """Record an answer and everything that follows from it.

        If the answers contradict each other, the earlier one wins: an answer
        that is already settled is not recorded again, and a clause with every
        literal false derives nothing. Either way this record only stops
        questions from being asked, so a contradiction costs nothing more.
        """
        pending = self._set(pred, truth)
        while pending:
            key = pending.pop()
            for clause in self._index.by_key.get(key, ()):
                forced = self._forced(clause)
                if forced is not None:
                    pending |= self._set(*forced)

    def _set(self, pred: Predicate, truth: bool) -> set[str]:
        """Store one fact. Returns the attribute to re-check, if it was new."""
        if self.value(pred) is not None:
            return set()
        pred, flipped = _canonical(pred)
        truth = truth != flipped
        self._facts[pred] = truth
        key = pred.key
        if pred.op == 'gte':
            if truth:
                self._met[key] = max(self._met.get(key, pred.value), pred.value)
            else:
                self._missed[key] = min(self._missed.get(key, pred.value), pred.value)
        elif pred.op == 'eq' and truth:
            self._pinned[key] = pred.value
        return {key}

    def _forced(self, clause: Clause) -> tuple[Predicate, bool] | None:
        """The one literal a clause still needs, once every other one is false."""
        remaining = None
        for pred, truth in clause:
            known = self.value(pred)
            if known is None:
                if remaining is not None:
                    return None
                remaining = (pred, truth)
            elif known == truth:
                return None
        return remaining
