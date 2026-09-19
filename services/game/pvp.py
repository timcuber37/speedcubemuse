"""Head-to-head match state, stored in Supabase.

Schema and the reasoning behind it: supabase_game_setup.sql.

The division of labour is the important part. Clients hold a Realtime
subscription to `game_moves` and read it; they never write. Every mutation goes
through Flask using the service-role client, because each one either reads a
secret (answering a question), writes one (choosing a cuber), or decides a
result — none of which a player's own browser can be trusted to do.

That also means the answer to "is your cuber European?" is computed here, from
the opponent's stored `wca_id` and the precomputed profile, and only the
resulting yes/no is ever published to the match feed.
"""
import logging
import secrets
import string

from services.auth import get_supabase_admin

from . import profiles
from .attributes import Predicate
from .engine import answer_for

logger = logging.getLogger(__name__)

# Unambiguous alphabet: no O/0, I/1, S/5. These get read aloud and typed in.
_CODE_ALPHABET = 'ABCDEFGHJKLMNPQRTUVWXYZ23467889'
_CODE_LENGTH = 6
_CODE_ATTEMPTS = 8


class PvpError(Exception):
    """A rule violation the player should see (wrong turn, match full, ...)."""


_ready: bool | None = None


def is_ready() -> bool:
    """Whether the head-to-head tables exist yet.

    `supabase_game_setup.sql` is run once by hand in the Supabase SQL editor —
    there is no migration runner here — so a deploy can easily land before it
    has been applied. Without this check the lobby renders normally and then
    fails with a generic error the moment someone creates a match, which looks
    like a bug rather than missing setup. Probed once per process and cached;
    tables do not appear and disappear at runtime.
    """
    global _ready
    if _ready is not None:
        return _ready

    client = get_supabase_admin()
    if client is None:
        logger.warning('head-to-head disabled: no Supabase service-role key')
        _ready = False
        return _ready

    try:
        # `outcome` arrived with the rebuttal rule, so selecting it checks both
        # that the tables exist and that the schema is current. Probing only
        # for the table would pass on a database created before that change and
        # then fail mid-match on a CHECK violation the first time someone
        # guessed correctly — a much worse failure than being told up front.
        client.table('game_matches').select('id, outcome').limit(1).execute()
        _ready = True
    except Exception as e:
        logger.warning(
            'head-to-head disabled: %s. Run supabase_game_setup.sql to create '
            'or update the match tables.',
            str(e)[:140],
        )
        _ready = False
    return _ready


def _client():
    client = get_supabase_admin()
    if client is None:
        raise PvpError('Head-to-head is unavailable right now.')
    return client


def _new_code() -> str:
    return ''.join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LENGTH))


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def create_match(user_id: str, difficulty: str, wca_id: str) -> dict:
    """Open a match and store the host's chosen cuber."""
    _require_in_pool(wca_id, difficulty)
    client = _client()

    # Codes are short enough to collide occasionally; retry rather than widen
    # them, since a 6-character code is the point.
    for _ in range(_CODE_ATTEMPTS):
        code = _new_code()
        try:
            res = client.table('game_matches').insert({
                'join_code': code,
                'difficulty': difficulty,
                'host_user': user_id,
                'status': 'waiting',
            }).execute()
            break
        except Exception as e:
            if 'duplicate' not in str(e).lower():
                logger.error('match create failed: %s', e)
                raise PvpError('Could not create the match.')
    else:
        raise PvpError('We couldn\'t create a match code. Please try again.')

    match = res.data[0]
    _set_secret(match['id'], user_id, wca_id)
    return {'id': match['id'], 'join_code': code, 'difficulty': difficulty}


def join_match(user_id: str, join_code: str, wca_id: str) -> dict:
    """Join an open match and store the guest's chosen cuber."""
    client = _client()
    code = (join_code or '').strip().upper()
    if not code:
        raise PvpError('Enter a join code.')

    res = client.table('game_matches').select('*').eq('join_code', code).execute()
    if not res.data:
        raise PvpError('No match was found with that code. Check the code with your friend and try again.')
    match = res.data[0]

    if match['host_user'] == user_id:
        raise PvpError("That's your own match — share the code with someone else.")
    if match['guest_user']:
        raise PvpError('That match already has two players.')
    if match['status'] != 'waiting':
        raise PvpError('That match is no longer open.')

    _require_in_pool(wca_id, match['difficulty'])
    _set_secret(match['id'], user_id, wca_id)

    client.table('game_matches').update({
        'guest_user': user_id,
        'status': 'active',
        # The guest joined last, so the host asks first.
        'turn': match['host_user'],
    }).eq('id', match['id']).execute()

    _record_move(match['id'], user_id, 'joined', {})
    return {'id': match['id'], 'join_code': code, 'difficulty': match['difficulty']}


def _require_in_pool(wca_id: str, difficulty: str) -> None:
    """Reject a cuber the opponent could never guess."""
    row = profiles.find_by_wca_id(wca_id)
    if row is None:
        raise PvpError('Pick a competitor from the suggestions.')
    tiers = profiles.TIERS.get(difficulty, ())
    if row['tier'] not in tiers:
        raise PvpError(
            f"{row['name']} isn't available on {difficulty.title()} difficulty. Choose another competitor."
        )


def _set_secret(match_id: str, user_id: str, wca_id: str) -> None:
    _client().table('game_secrets').upsert({
        'match_id': match_id, 'user_id': user_id, 'wca_id': wca_id,
    }).execute()


def _get_secret(match_id: str, user_id: str) -> dict | None:
    """The profile of the cuber `user_id` is hiding in this match."""
    res = _client().table('game_secrets').select('wca_id') \
        .eq('match_id', match_id).eq('user_id', user_id).execute()
    if not res.data:
        return None
    return profiles.find_by_wca_id(res.data[0]['wca_id'])


# ---------------------------------------------------------------------------
# Play
# ---------------------------------------------------------------------------

def _load_match(match_id: str, user_id: str) -> dict:
    res = _client().table('game_matches').select('*').eq('id', match_id).execute()
    if not res.data:
        raise PvpError('Match not found.')
    match = res.data[0]
    if user_id not in (match['host_user'], match['guest_user']):
        raise PvpError('You are not in this match.')
    return match


def _opponent(match: dict, user_id: str) -> str:
    return match['guest_user'] if match['host_user'] == user_id else match['host_user']


def _require_turn(match: dict, user_id: str) -> None:
    if match['status'] not in ('active', 'rebuttal'):
        raise PvpError('This match is not in play.')
    if match['turn'] != user_id:
        raise PvpError("It's not your turn.")


def ask(match_id: str, user_id: str, pred: Predicate) -> dict:
    """Answer a question about the opponent's cuber and pass the turn back."""
    match = _load_match(match_id, user_id)
    _require_turn(match, user_id)
    if match['status'] == 'rebuttal':
        # The rebuttal is one guess, not a free turn. Letting it buy questions
        # would hand the trailing player information the winner never had.
        raise PvpError('Last chance — name their cuber, no more questions.')

    opponent = _opponent(match, user_id)
    secret = _get_secret(match_id, opponent)
    if secret is None:
        raise PvpError('Your opponent has not chosen a cuber yet.')

    answer = answer_for(secret['attrs'], pred)

    # Only the question and its yes/no reach the feed. The opponent's wca_id
    # never leaves this function.
    _record_move(match_id, user_id, 'question', {
        'question': pred.render(),
        'predicate_id': pred.as_id(),
        'answer': answer,
    })
    _pass_turn(match_id, opponent)
    return {'answer': answer, 'question': pred.render()}


def guess(match_id: str, user_id: str, wca_id: str) -> dict:
    """Name the opponent's cuber. A correct guess ends the match."""
    match = _load_match(match_id, user_id)
    _require_turn(match, user_id)

    opponent = _opponent(match, user_id)
    secret = _get_secret(match_id, opponent)
    if secret is None:
        raise PvpError('Your opponent has not chosen a cuber yet.')

    guessed = profiles.find_by_wca_id(wca_id)
    correct = guessed is not None and guessed['wca_id'] == secret['wca_id']

    _record_move(match_id, user_id, 'guess', {
        'name': guessed['name'] if guessed else wca_id,
        'correct': correct,
    })

    # The rebuttal itself: the trailing player's single chance to level it.
    if match['status'] == 'rebuttal':
        if correct:
            _finish(match_id, winner=None, outcome='tie')
            _record_move(match_id, user_id, 'tie', {'name': secret['name']})
            return {'correct': True, 'tie': True, 'secret': _reveal(secret)}

        # Missed it, so the guess that triggered the rebuttal stands.
        _finish(match_id, winner=match['winner'], outcome='win')
        _record_move(match_id, match['winner'], 'win', {})
        return {'correct': False, 'rebuttal_failed': True}

    if correct:
        # Not over yet. They get one guess to tie, and `winner` holds this
        # player as the pending winner until that resolves.
        _client().table('game_matches').update({
            'status': 'rebuttal', 'winner': user_id, 'turn': opponent,
        }).eq('id', match_id).execute()
        _record_move(match_id, user_id, 'rebuttal', {'name': secret['name']})
        return {'correct': True, 'awaiting_rebuttal': True,
                'secret': _reveal(secret)}

    _pass_turn(match_id, opponent)
    return {'correct': False}


def _finish(match_id: str, winner: str | None, outcome: str) -> None:
    _client().table('game_matches').update({
        'status': 'finished', 'winner': winner, 'outcome': outcome, 'turn': None,
    }).eq('id', match_id).execute()


def resign(match_id: str, user_id: str) -> dict:
    match = _load_match(match_id, user_id)
    opponent = _opponent(match, user_id)
    # Conceding during a rebuttal is declining to take it, so the pending
    # winner's guess stands either way — the opponent wins in both cases.
    _finish(match_id, winner=opponent, outcome='resign')
    _record_move(match_id, user_id, 'resign', {})

    secret = _get_secret(match_id, opponent)
    return {'resigned': True, 'secret': _reveal(secret) if secret else None}


def state(match_id: str, user_id: str) -> dict:
    """Everything a player is allowed to know about the match.

    Used on load and as the reconnect path — Realtime delivers new moves, but a
    player who refreshes or drops the socket needs the history back.
    """
    match = _load_match(match_id, user_id)
    moves = _client().table('game_moves').select('*') \
        .eq('match_id', match_id).order('id').execute().data or []

    mine = _get_secret(match_id, user_id)
    in_rebuttal = match['status'] == 'rebuttal'
    return {
        'id': match['id'],
        'join_code': match['join_code'],
        'difficulty': match['difficulty'],
        'status': match['status'],
        'is_host': match['host_user'] == user_id,
        'your_turn': match['turn'] == user_id,
        'waiting_for_opponent': match['guest_user'] is None,
        'winner': match['winner'],
        'outcome': match.get('outcome'),
        'tie': match.get('outcome') == 'tie',
        # During a rebuttal `winner` is only provisional, so "did you win" has
        # no answer yet — the client shows the last-chance state instead.
        'you_won': (None if in_rebuttal or not match['winner']
                    else match['winner'] == user_id),
        'awaiting_rebuttal': in_rebuttal,
        'your_rebuttal': in_rebuttal and match['turn'] == user_id,
        'your_cuber': _reveal(mine) if mine else None,
        'moves': [
            {
                'id': m['id'],
                'mine': m['actor'] == user_id,
                'kind': m['kind'],
                'payload': m['payload'],
            }
            for m in moves
        ],
    }


def _reveal(row: dict) -> dict:
    return {
        'wca_id': row['wca_id'],
        'name': row['name'],
        'country': row['country_id'],
        'wca_url': f'https://www.worldcubeassociation.org/persons/{row["wca_id"]}',
    }


def _pass_turn(match_id: str, next_user: str) -> None:
    _client().table('game_matches').update({'turn': next_user}) \
        .eq('id', match_id).execute()


def _record_move(match_id: str, actor: str, kind: str, payload: dict) -> None:
    """Append to the move feed. This insert is what Realtime broadcasts."""
    _client().table('game_moves').insert({
        'match_id': match_id, 'actor': actor, 'kind': kind, 'payload': payload,
    }).execute()
