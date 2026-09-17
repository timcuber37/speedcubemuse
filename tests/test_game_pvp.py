"""State-machine tests for head-to-head, including the rebuttal rule.

Head-to-head stores its state in Supabase, and those tables are created by hand
(`supabase_game_setup.sql`), so on most machines there is nothing to play
against. The match flow is still a state machine with several transitions — and
the rebuttal adds a state where `winner` is only provisional — so it is tested
against an in-memory stand-in for the Supabase client rather than left
unexercised until someone runs a real match.

The fake implements only the query shapes services/game/pvp.py actually uses.

    python -m pytest tests/test_game_pvp.py -v
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from services.game import pvp
from services.game.attributes import Predicate


# ---------------------------------------------------------------------------
# A minimal stand-in for the Supabase client
# ---------------------------------------------------------------------------

# Nullable columns from supabase_game_setup.sql, so an inserted row looks like
# one Postgres would hand back.
_DEFAULTS = {
    'game_matches': {
        'guest_user': None, 'turn': None, 'winner': None, 'outcome': None,
        'status': 'waiting',
    },
}


class _Result:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows, table):
        self._rows = rows
        self._table = table
        self._op = None
        self._payload = None
        self._filters = []

    def select(self, *_cols):
        self._op = 'select'
        return self

    def insert(self, row):
        self._op, self._payload = 'insert', row
        return self

    def update(self, patch):
        self._op, self._payload = 'update', patch
        return self

    def upsert(self, row):
        self._op, self._payload = 'upsert', row
        return self

    def eq(self, column, value):
        self._filters.append((column, value))
        return self

    def order(self, column):
        self._rows.sort(key=lambda r: r.get(column, 0))
        return self

    def limit(self, _n):
        return self

    def _matches(self, row):
        return all(row.get(c) == v for c, v in self._filters)

    def execute(self):
        if self._op == 'select':
            return _Result([dict(r) for r in self._rows if self._matches(r)])
        if self._op == 'insert':
            # Postgres returns every column, defaulting the ones not supplied.
            # Without that the code under test sees a KeyError where production
            # would see None.
            row = dict(_DEFAULTS.get(self._table, {}))
            row.update(self._payload)
            row.setdefault('id', len(self._rows) + 1)
            self._rows.append(row)
            return _Result([dict(row)])
        if self._op == 'update':
            hit = [r for r in self._rows if self._matches(r)]
            for r in hit:
                r.update(self._payload)
            return _Result([dict(r) for r in hit])
        if self._op == 'upsert':
            key = ('match_id', 'user_id')
            for r in self._rows:
                if all(r.get(k) == self._payload.get(k) for k in key):
                    r.update(self._payload)
                    return _Result([dict(r)])
            self._rows.append(dict(self._payload))
            return _Result([dict(self._payload)])
        raise AssertionError(f'unsupported op {self._op}')


class FakeSupabase:
    def __init__(self):
        self.tables = {'game_matches': [], 'game_secrets': [], 'game_moves': []}

    def table(self, name):
        return _Query(self.tables.setdefault(name, []), name)


HOST, GUEST = 'user-host', 'user-guest'
HOST_CUBER, GUEST_CUBER = '2009ZEMD01', '2012PARK03'


@pytest.fixture
def match(monkeypatch):
    """A live two-player match, with both secrets set and the host to move."""
    db = FakeSupabase()
    monkeypatch.setattr(pvp, '_client', lambda: db)
    monkeypatch.setattr(pvp, 'is_ready', lambda: True)

    # Profiles are looked up by wca_id; stub just enough for guess() to resolve.
    people = {
        HOST_CUBER: {'wca_id': HOST_CUBER, 'name': 'Feliks Zemdegs',
                     'country_id': 'Australia', 'tier': 1, 'attrs': {'has_wr': True}},
        GUEST_CUBER: {'wca_id': GUEST_CUBER, 'name': 'Max Park',
                      'country_id': 'USA', 'tier': 1, 'attrs': {'has_wr': True}},
    }
    monkeypatch.setattr(pvp.profiles, 'find_by_wca_id', lambda w: people.get(w))
    monkeypatch.setattr(pvp.profiles, 'TIERS', {'normal': (1, 2)})

    created = pvp.create_match(HOST, 'normal', HOST_CUBER)
    pvp.join_match(GUEST, created['join_code'], GUEST_CUBER)

    row = db.tables['game_matches'][0]
    assert row['status'] == 'active' and row['turn'] == HOST
    return db, created['id']


def _row(db):
    return db.tables['game_matches'][0]


def _kinds(db):
    return [m['kind'] for m in db.tables['game_moves']]


# ---------------------------------------------------------------------------
# Normal play, unchanged by the rebuttal rule
# ---------------------------------------------------------------------------

def test_a_wrong_guess_just_passes_the_turn(match):
    db, match_id = match
    out = pvp.guess(match_id, HOST, HOST_CUBER)   # host's own cuber, not the guest's
    assert out == {'correct': False}
    assert _row(db)['status'] == 'active'
    assert _row(db)['turn'] == GUEST


def test_asking_passes_the_turn(match):
    db, match_id = match
    out = pvp.ask(match_id, HOST, Predicate('has_wr', 'is', True))
    assert out['answer'] == 'yes'          # the guest is hiding Max Park
    assert _row(db)['turn'] == GUEST


def test_you_cannot_move_out_of_turn(match):
    _db, match_id = match
    with pytest.raises(pvp.PvpError, match="not your turn"):
        pvp.guess(match_id, GUEST, HOST_CUBER)


# ---------------------------------------------------------------------------
# The rebuttal
# ---------------------------------------------------------------------------

def test_a_correct_guess_opens_a_rebuttal_rather_than_winning(match):
    """The match is not over — the opponent gets one chance to level it."""
    db, match_id = match
    out = pvp.guess(match_id, HOST, GUEST_CUBER)

    assert out['correct'] is True
    assert out['awaiting_rebuttal'] is True
    assert out['secret']['name'] == 'Max Park'

    row = _row(db)
    assert row['status'] == 'rebuttal'
    assert row['turn'] == GUEST, 'the rebuttal is the opponent\'s to take'
    assert row['winner'] == HOST, 'pending only'
    assert row.get('outcome') is None, 'nothing is decided yet'
    assert 'rebuttal' in _kinds(db)


def test_a_successful_rebuttal_ties_the_match(match):
    db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)          # host opens the rebuttal
    out = pvp.guess(match_id, GUEST, HOST_CUBER)    # guest levels it

    assert out['correct'] is True and out['tie'] is True
    row = _row(db)
    assert row['status'] == 'finished'
    assert row['outcome'] == 'tie'
    assert row['winner'] is None, 'a draw has no winner'
    assert row['turn'] is None
    assert 'tie' in _kinds(db)


def test_a_failed_rebuttal_lets_the_original_guess_stand(match):
    db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    out = pvp.guess(match_id, GUEST, GUEST_CUBER)   # wrong: that's their own

    assert out['correct'] is False and out['rebuttal_failed'] is True
    row = _row(db)
    assert row['status'] == 'finished'
    assert row['outcome'] == 'win'
    assert row['winner'] == HOST
    assert 'win' in _kinds(db)


def test_the_rebuttal_is_one_guess_not_a_free_turn(match):
    """Questions are closed during a rebuttal.

    Allowing them would hand the trailing player information the winner never
    had a chance to use.
    """
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    with pytest.raises(pvp.PvpError, match='no more questions'):
        pvp.ask(match_id, GUEST, Predicate('has_wr', 'is', True))


def test_the_winner_cannot_keep_playing_during_the_rebuttal(match):
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    with pytest.raises(pvp.PvpError, match="not your turn"):
        pvp.guess(match_id, HOST, GUEST_CUBER)


def test_only_one_rebuttal_is_ever_offered(match):
    """A failed rebuttal ends the match; it does not open a counter-rebuttal."""
    db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    pvp.guess(match_id, GUEST, GUEST_CUBER)        # misses
    assert _row(db)['status'] == 'finished'
    with pytest.raises(pvp.PvpError, match='not in play'):
        pvp.guess(match_id, HOST, GUEST_CUBER)


def test_resigning_a_rebuttal_concedes_it(match):
    """Declining the last chance leaves the pending winner's guess standing."""
    db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    out = pvp.resign(match_id, GUEST)

    assert out['resigned'] is True
    row = _row(db)
    assert row['status'] == 'finished'
    assert row['winner'] == HOST
    assert row['outcome'] == 'resign'


# ---------------------------------------------------------------------------
# What each side is told
# ---------------------------------------------------------------------------

def test_state_marks_the_rebuttal_for_the_right_player(match):
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)

    guest = pvp.state(match_id, GUEST)
    assert guest['awaiting_rebuttal'] is True
    assert guest['your_rebuttal'] is True
    assert guest['your_turn'] is True

    host = pvp.state(match_id, HOST)
    assert host['awaiting_rebuttal'] is True
    assert host['your_rebuttal'] is False
    assert host['your_turn'] is False


def test_no_winner_is_declared_while_a_rebuttal_is_open(match):
    """`winner` is provisional here, so neither side is told they have won."""
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    for who in (HOST, GUEST):
        assert pvp.state(match_id, who)['you_won'] is None


def test_state_reports_a_tie_to_both_players(match):
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    pvp.guess(match_id, GUEST, HOST_CUBER)
    for who in (HOST, GUEST):
        s = pvp.state(match_id, who)
        assert s['tie'] is True
        assert s['outcome'] == 'tie'
        assert s['you_won'] is None, 'a draw is not a win for either side'


def test_state_reports_the_winner_after_a_failed_rebuttal(match):
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)
    pvp.guess(match_id, GUEST, GUEST_CUBER)
    assert pvp.state(match_id, HOST)['you_won'] is True
    assert pvp.state(match_id, GUEST)['you_won'] is False


def test_a_secret_never_reaches_the_other_player(match):
    """The rebuttal adds states; it must not add a leak.

    Each player's state carries their own cuber and never the opponent's.
    """
    _db, match_id = match
    pvp.guess(match_id, HOST, GUEST_CUBER)

    host = pvp.state(match_id, HOST)
    guest = pvp.state(match_id, GUEST)
    assert host['your_cuber']['wca_id'] == HOST_CUBER
    assert guest['your_cuber']['wca_id'] == GUEST_CUBER

    # The guest is mid-rebuttal, so the host's cuber is the answer they are
    # trying to find — it must not appear anywhere in the feed they can read.
    # Names may legitimately appear (a guess is public); WCA ids must not.
    assert all(HOST_CUBER not in str(m['payload']) for m in guest['moves'])
