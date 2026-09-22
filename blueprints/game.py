"""Guess the Cuber routes.

A blueprint rather than more routes in app.py, which is already carrying every
route for the rest of the site.

Every endpoint here is stateless between turns. The client holds the answer log
and replays it on each request; the server rebuilds the belief from scratch.
That is not a stylistic choice — the web app runs two Gunicorn workers with no
sticky sessions and the Fly machine suspends when idle, so anything kept in
process memory would vanish or be invisible to the next request. The same
reason Flask-Limiter's in-memory counters reset on suspend.

Mode 2 needs no secret at all (the cuber is in the player's head). Mode 1 keeps
its secret in a Fernet-sealed token the browser carries — see tokens.py for why
a Flask session cookie is not sufficient.
"""
import logging
import time
from contextlib import contextmanager

from flask import Blueprint, jsonify, render_template, request

from config import (GAME_LIMIT_ASK, GAME_LIMIT_CHEAP, GAME_LIMIT_FREE,
                    GAME_RATE_LIMITS, MAX_GUEST_GAME_QUESTIONS,
                    SUPABASE_ANON_KEY, SUPABASE_URL)
from extensions import limiter
from services.auth import get_user_from_token
from services.api_usage import usage_source
from services.game import fame_index, profiles, pvp, tokens
from services.game.attributes import (EVENT_TIER_WEIGHT, EVENT_TIERS,
                                      country_label, event_label,
                                      predicate_from_id)
from services.game.engine import ANSWERS, MAX_QUESTIONS, answer_for, decode_log
from services.game.profiles import DEFAULT_TIER, TIERS
from services.game.question_parser import QuestionParser

logger = logging.getLogger(__name__)

game_bp = Blueprint('game', __name__)

question_parser = QuestionParser()

# Guards on the client-supplied answer log. It is replayed in full on every
# turn, so an unbounded log is both a compute cost and an obvious abuse vector.
MAX_LOG_ENTRIES = MAX_QUESTIONS + 10
MAX_REJECTED = 25


def get_current_user():
    """The signed-in user, from the Supabase bearer token.

    Mirrors app.py's helper rather than importing it, so the blueprint has no
    import cycle back to the application module.
    """
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return get_user_from_token(auth_header[7:]), auth_header[7:]
    return None, None


def _is_authenticated() -> bool:
    """Exempts signed-in users from the guest day-cap, as /api/delegate/ask does."""
    user, _ = get_current_user()
    return user is not None


def _limit(spec, **kwargs):
    """Rate-limit decorator that honours the GAME_RATE_LIMITS switch.

    When limits are off this returns `limiter.exempt` rather than a no-op: a
    route with no limit at all still inherits Flask-Limiter's 200/day + 50/hour
    defaults, which are tighter than anything here. Turning limits "off" by
    removing the decorator would quietly make the game *more* restricted.
    """
    if not GAME_RATE_LIMITS:
        return limiter.exempt
    return limiter.limit(spec, **kwargs)


def _difficulty(payload) -> str:
    value = (payload.get('difficulty') or DEFAULT_TIER).lower()
    return value if value in TIERS else DEFAULT_TIER


def _engine_or_error(difficulty):
    engine = profiles.get_engine(difficulty)
    if engine is None:
        # The matrix is missing or the database is down. Say so plainly rather
        # than serving a game with no candidates.
        return None, (jsonify({
            'error': 'We couldn\'t load the competitors right now. Please try again shortly.'
        }), 503)
    return engine, None


def _public(row: dict, confidence: float | None = None) -> dict:
    """The shape of a cuber sent to the browser.

    Note what is absent: the attrs blob. In Mode 1 that would hand the player
    every answer at once, and in PVP it would leak the opponent's cuber.
    """
    out = {
        'wca_id': row['wca_id'],
        'name': row['name'],
        # Same apostrophe restoration the questions use — this string lands on
        # the result card and in the PVP feed.
        'country': country_label(row['country_id']),
        'continent': (row['continent_id'] or '').lstrip('_'),
        'wca_url': f'https://www.worldcubeassociation.org/persons/{row["wca_id"]}',
    }
    if confidence is not None:
        out['confidence'] = round(confidence, 4)
    return out


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def _page(template, **extra):
    return render_template(template,
                           supabase_url=SUPABASE_URL,
                           supabase_anon_key=SUPABASE_ANON_KEY,
                           **extra)


@game_bp.route('/game')
@limiter.exempt
def game_index():
    return _page('game/index.html')


def _event_weight_scale():
    """The event-weight table shown on /game/cubers, read off the live schema.

    Built from EVENT_TIERS rather than written out in the template: the weights
    are the most likely part of the fame score to be retuned, and a hand-copied
    list would go on describing the old ranking without anything catching it.
    """
    by_tier: dict[int, list[str]] = {}
    for event_id, tier in EVENT_TIERS.items():
        by_tier.setdefault(tier, []).append(event_label(event_id))
    return [{'tier': tier, 'weight': EVENT_TIER_WEIGHT[tier], 'events': events}
            for tier, events in sorted(by_tier.items())]


@game_bp.route('/game/cubers')
@limiter.exempt
def game_cubers():
    """Who each difficulty includes, and why they rank where they do.

    Renders without touching the database — the tier explanation is the thing
    the link promises, and it should not wait on a matrix read or vanish when
    one fails. The ranking arrives from /api/game/cubers once the page is up.
    """
    return _page('game/cubers.html', event_scale=_event_weight_scale())


@game_bp.route('/game/akinator')
@limiter.exempt
def game_akinator():
    return _page('game/akinator.html')


@game_bp.route('/game/solo')
@limiter.exempt
def game_solo():
    return _page('game/solo.html')


@game_bp.route('/game/pvp')
@limiter.exempt
def game_pvp():
    # Told up front rather than on submit — the match tables are created by
    # hand, so a deploy can land before they exist.
    return _page('game/pvp.html', pvp_ready=pvp.is_ready())


# ---------------------------------------------------------------------------
# Mode 2 — the app guesses. Zero model calls.
# ---------------------------------------------------------------------------

@contextmanager
def _time_turn_stage(stage, difficulty, candidates=0, answers=0):
    """Log even on a worker abort, separating CPU work from elapsed time."""
    started, cpu_started = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        logger.info(
            'akinator stage=%s difficulty=%s candidates=%d answers=%d wall_ms=%.1f cpu_ms=%.1f',
            stage, difficulty, candidates, answers,
            (time.perf_counter() - started) * 1000,
            (time.process_time() - cpu_started) * 1000,
        )


@game_bp.route('/api/game/akinator/turn', methods=['POST'])
@_limit(GAME_LIMIT_FREE)
def akinator_turn():
    """Given the answers so far, return the next question or a guess.

    Fully stateless: the secret lives in the player's head, so there is nothing
    for the server to remember and nothing to hide.
    """
    payload = request.get_json(silent=True) or {}
    difficulty = _difficulty(payload)

    with _time_turn_stage('load', difficulty):
        engine, error = _engine_or_error(difficulty)
    if error:
        return error

    raw_log = payload.get('answers') or []
    if len(raw_log) > MAX_LOG_ENTRIES:
        return jsonify({'error': 'This game has reached its question limit. Please start a new game.'}), 400
    rejected = [str(x) for x in (payload.get('rejected') or [])][:MAX_REJECTED]

    answers = decode_log(raw_log)
    with _time_turn_stage('replay', difficulty, len(engine.rows), len(answers)):
        belief = engine.replay(answers, rejected)

    if engine.should_guess(belief, answers):
        return jsonify(_guess_response(engine, belief, len(answers)))

    # The engine excludes questions the log already settles — see
    # Engine.implied(). It takes the log rather than an asked-set precisely so
    # that exclusion can't be confused with the question count above.
    with _time_turn_stage('pick', difficulty, len(engine.rows), len(answers)):
        pred = engine.pick_question(belief, answers)
    if pred is None:
        # No question left that separates anyone — guessing is all that remains.
        return jsonify(_guess_response(engine, belief, len(answers)))

    return jsonify({
        'done': False,
        'question': {'id': pred.as_id(), 'text': pred.render()},
        'asked': len(answers),
        'max_questions': MAX_QUESTIONS,
        'answers': list(ANSWERS),
        'remaining': len(engine._live(belief)),
    })


def _guess_response(engine, belief, asked_count) -> dict:
    top = engine.top_candidates(belief, 6)
    if not top:
        return {'done': True, 'guess': None,
                'error': 'I couldn\'t find a competitor that fits your answers. Try a new game or a wider difficulty.'}
    best, confidence = top[0]
    return {
        'done': True,
        # Below the confidence bar the search has stalled with several
        # near-identical candidates standing, which is common on the larger
        # pools. Naming one of them outright is just a confident wrong answer,
        # so the client offers the shortlist instead.
        'confident': engine.is_confident(belief),
        'guess': _public(best, confidence),
        'alternatives': [_public(r, c) for r, c in top[1:]],
        'asked': asked_count,
    }


# ---------------------------------------------------------------------------
# Mode 1 — the player guesses. One model call per typed question.
# ---------------------------------------------------------------------------

@game_bp.route('/api/game/solo/start', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def solo_start():
    """Pick a secret cuber and seal it into a token for the browser to carry."""
    payload = request.get_json(silent=True) or {}
    difficulty = _difficulty(payload)

    # Not routed through the engine: on Everyone the secret is drawn from every
    # competitor in the database, including the ~245k the engine never loads.
    # This mode only ever needs the one hidden row, not a candidate set.
    secret = profiles.random_profile(difficulty)
    if secret is None:
        return jsonify({
            'error': 'We couldn\'t load the competitors right now. Please try again shortly.'
        }), 503

    logger.info('solo game started (%s): %s', difficulty, secret['wca_id'])

    engine = profiles.get_engine(difficulty)
    return jsonify({
        'token': tokens.seal({'wca_id': secret['wca_id'], 'difficulty': difficulty}),
        'difficulty': difficulty,
        'pool_size': len(engine.rows) if engine else None,
        'max_questions': MAX_QUESTIONS,
    })


@game_bp.route('/api/game/solo/ask', methods=['POST'])
@_limit(GAME_LIMIT_ASK)
@_limit(f"{MAX_GUEST_GAME_QUESTIONS} per day", exempt_when=_is_authenticated)
@usage_source("web")
def solo_ask():
    """Answer one free-text question about the sealed secret cuber."""
    payload = request.get_json(silent=True) or {}

    try:
        sealed = tokens.unseal(payload.get('token'))
    except tokens.TokenError as e:
        return jsonify({'error': str(e)}), 400

    secret = profiles.find_by_wca_id(sealed.get('wca_id'))
    if secret is None:
        # The matrix was rebuilt under an in-flight game and this cuber no
        # longer qualifies. Nothing to answer about.
        return jsonify({'error': 'This game has expired. Start a new one.'}), 409

    asked = (payload.get('question') or '').strip()
    pred, parse_error = question_parser.parse(asked)
    if pred is None:
        return jsonify({'error': parse_error, 'unanswerable': True}), 200

    # Echo the player's own wording, not pred.render(). A negated mapping —
    # "are they retired?" becomes `is_active is False` — answers correctly but
    # renders as the attribute's positive phrasing ("Are they still actively
    # competing?"), which would show the player a yes against the opposite
    # question. `interpreted` carries the canonical reading for transparency.
    return jsonify({
        'question': asked,
        'interpreted': pred.render(),
        'negated': pred.op == 'is' and pred.value is False,
        'answer': answer_for(secret['attrs'], pred),
        'predicate_id': pred.as_id(),
    })


@game_bp.route('/api/game/solo/guess', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def solo_guess():
    """Check a guess against the sealed secret."""
    payload = request.get_json(silent=True) or {}

    try:
        sealed = tokens.unseal(payload.get('token'))
    except tokens.TokenError as e:
        return jsonify({'error': str(e)}), 400

    secret = profiles.find_by_wca_id(sealed.get('wca_id'))
    if secret is None:
        return jsonify({'error': 'This game has expired. Start a new one.'}), 409

    guess_id = (payload.get('wca_id') or '').strip()
    correct = guess_id == secret['wca_id']

    response = {'correct': correct}
    if correct or payload.get('reveal'):
        # Only ever reveal on a win or an explicit give-up, never on a miss.
        response['secret'] = _public(secret)
    return jsonify(response)


@game_bp.route('/api/game/search')
@limiter.exempt
def game_search():
    """Name autocomplete for the guess box.

    On Everyone the search reaches the whole database, since the hidden
    competitor may be one the engine never loads.
    """
    everyone = request.args.get('difficulty') == 'everyone'
    hits = profiles.search_by_name(request.args.get('q', ''), include_all=everyone)
    return jsonify({'results': [_public(r) for r in hits]})


# ---------------------------------------------------------------------------
# Mode 3 — head to head
# ---------------------------------------------------------------------------

def _pvp_user():
    """PVP requires a signed-in user: matches are keyed on auth.users."""
    if not pvp.is_ready():
        return None, (jsonify({
            'error': 'Head to head is unavailable right now. Try one of the other game modes.'
        }), 503)
    user, _token = get_current_user()
    if user is None:
        return None, (jsonify({'error': 'Sign in to play head-to-head.'}), 401)
    return user, None


def _pvp(handler):
    """Run a pvp call, turning rule violations into 400s the player can read."""
    try:
        return jsonify(handler())
    except pvp.PvpError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error('pvp call failed: %s', e)
        return jsonify({'error': 'We couldn\'t update the match. Please try again.'}), 500


@game_bp.route('/api/game/pvp/create', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def pvp_create():
    user, error = _pvp_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    return _pvp(lambda: pvp.create_match(
        user.id, _difficulty(payload), (payload.get('wca_id') or '').strip()
    ))


@game_bp.route('/api/game/pvp/join', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def pvp_join():
    user, error = _pvp_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    return _pvp(lambda: pvp.join_match(
        user.id, payload.get('join_code', ''), (payload.get('wca_id') or '').strip()
    ))


@game_bp.route('/api/game/pvp/<match_id>/state')
@_limit(GAME_LIMIT_FREE)
def pvp_state(match_id):
    user, error = _pvp_user()
    if error:
        return error
    return _pvp(lambda: pvp.state(match_id, user.id))


@game_bp.route('/api/game/pvp/<match_id>/ask', methods=['POST'])
@_limit(GAME_LIMIT_ASK)
@_limit(f"{MAX_GUEST_GAME_QUESTIONS} per day", exempt_when=_is_authenticated)
@usage_source("web")
def pvp_ask(match_id):
    user, error = _pvp_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}

    pred, parse_error = question_parser.parse(payload.get('question', ''))
    if pred is None:
        # An unmappable question doesn't reach the opponent's cuber, so it
        # costs no turn — the same treatment as in solo.
        return jsonify({'error': parse_error, 'unanswerable': True}), 200

    return _pvp(lambda: pvp.ask(match_id, user.id, pred))


@game_bp.route('/api/game/pvp/<match_id>/guess', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def pvp_guess(match_id):
    user, error = _pvp_user()
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    return _pvp(lambda: pvp.guess(
        match_id, user.id, (payload.get('wca_id') or '').strip()
    ))


@game_bp.route('/api/game/pvp/<match_id>/resign', methods=['POST'])
@_limit(GAME_LIMIT_CHEAP)
def pvp_resign(match_id):
    user, error = _pvp_user()
    if error:
        return error
    return _pvp(lambda: pvp.resign(match_id, user.id))


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

@game_bp.route('/api/game/cubers')
@limiter.exempt
def cuber_index():
    """The ranked fame index for /game/cubers.

    Sent as one payload rather than paged: the table sorts and filters in the
    browser, and a page-per-scroll would mean a round trip every time someone
    changed a filter to answer a question they could otherwise answer at once.
    """
    index = fame_index.get_index()
    if not index['rows']:
        return jsonify({
            'error': 'We couldn\'t load the rankings right now. Please try again shortly.'
        }), 503

    response = jsonify(index)
    # The matrix is rebuilt weekly and the server-side cache holds for an hour,
    # so a browser copy of the same age costs nothing and saves re-sending ~1,000
    # rows to anyone who opens the page twice.
    response.headers['Cache-Control'] = 'public, max-age=3600'
    return response


@game_bp.route('/api/game/question/<path:pred_id>')
@limiter.exempt
def question_text(pred_id):
    """Render a predicate id back to its question text.

    Lets the client show a readable history without keeping a parallel copy of
    the attribute schema in JavaScript.
    """
    try:
        return jsonify({'text': predicate_from_id(pred_id).render()})
    except (ValueError, KeyError):
        return jsonify({'error': 'This question is no longer available. Please start a new game.'}), 404
