"""Tests for the game's token sealing and question-parser validation.

Neither of these needs the Anthropic API: the token tests are pure crypto, and
the parser tests exercise `_to_predicate`, which is the validation layer that
stands between an untrusted model response and the game engine.

    python -m pytest tests/test_game_api.py -v
"""
import hashlib
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (GAME_LIMIT_ASK, GAME_LIMIT_CHEAP, GAME_LIMIT_FREE,
                    MAX_GUEST_GAME_QUESTIONS)
from services.game import tokens
from services.game.attributes import (Attribute, BY_KEY, EVENT_GROUPS,
                                      Predicate)
from services.game.question_parser import (_SCHEMA_VERSION, _coerce_event_id,
                                           _normalize)


@pytest.fixture(scope='module')
def parser():
    """A parser instance without touching the network.

    QuestionParser's constructor only builds the prompt and cache structures —
    the Anthropic client is created but never called by these tests.
    """
    from services.game.question_parser import QuestionParser
    return QuestionParser()


# ---------------------------------------------------------------------------
# Sealed tokens
# ---------------------------------------------------------------------------

def test_seal_roundtrip():
    token = tokens.seal({'wca_id': '2012PARK03', 'difficulty': 'easy'})
    payload = tokens.unseal(token)
    assert payload['wca_id'] == '2012PARK03'
    assert payload['difficulty'] == 'easy'


def test_token_does_not_leak_the_secret_in_plaintext():
    """The whole point: a player holding the token must not be able to read it.

    A Flask session cookie would fail this test — it is signed, not encrypted.
    """
    token = tokens.seal({'wca_id': '2012PARK03'})
    assert '2012PARK03' not in token

    import base64
    # Nor should it survive a naive base64 decode of any token segment.
    for chunk in token.split('.'):
        padded = chunk + '=' * (-len(chunk) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded)
        except Exception:
            continue
        assert b'2012PARK03' not in decoded


def test_tampered_token_is_rejected():
    token = tokens.seal({'wca_id': '2012PARK03'})
    # Flip a character in the ciphertext body.
    tampered = token[:-6] + ('A' if token[-6] != 'A' else 'B') + token[-5:]
    with pytest.raises(tokens.TokenError):
        tokens.unseal(tampered)


@pytest.mark.parametrize('bad', ['', None, 'not-a-token', 'a.b.c', 123, {}])
def test_malformed_tokens_are_rejected(bad):
    with pytest.raises(tokens.TokenError):
        tokens.unseal(bad)


def test_expired_token_is_rejected(monkeypatch):
    token = tokens.seal({'wca_id': '2012PARK03'})
    # Fernet stamps its own timestamp, so age the clock rather than the token.
    real_time = time.time
    monkeypatch.setattr(
        time, 'time', lambda: real_time() + tokens.TOKEN_TTL_SECONDS + 60
    )
    with pytest.raises(tokens.TokenError):
        tokens.unseal(token)


# ---------------------------------------------------------------------------
# Question normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('a,b', [
    ('Are they American?', 'are they american'),
    ('ARE THEY AMERICAN', '  are   they american!!  '),
    ("Do they have a world record?", 'do they have a world record'),
])
def test_normalization_collapses_equivalent_phrasings(a, b):
    """Cache keys must not be split by case, spacing, or punctuation."""
    assert _normalize(a) == _normalize(b)


def test_normalization_keeps_distinct_questions_distinct():
    assert _normalize('are they american') != _normalize('are they australian')


# ---------------------------------------------------------------------------
# Model output validation — the untrusted boundary
# ---------------------------------------------------------------------------

def test_valid_payloads_become_predicates(parser):
    cases = [
        ({'attribute': 'has_wr', 'op': 'is', 'value': 'true'}, 'has_wr', True),
        ({'attribute': 'has_wr', 'op': 'is', 'value': 'false'}, 'has_wr', False),
        ({'attribute': 'continent_id', 'op': 'eq', 'value': '_Europe'},
         'continent_id', '_Europe'),
        ({'attribute': 'comp_count', 'op': 'gte', 'value': '40'}, 'comp_count', 40),
    ]
    for payload, key, value in cases:
        pred = parser._to_predicate(payload)
        assert pred is not None, payload
        assert pred.key == key and pred.value == value


def test_hallucinated_attribute_is_rejected(parser):
    assert parser._to_predicate(
        {'attribute': 'favourite_pizza', 'op': 'is', 'value': 'true'}
    ) is None


def test_wrong_op_for_attribute_kind_is_rejected(parser):
    """A boolean attribute compared with 'gte' would evaluate nonsensically."""
    assert parser._to_predicate(
        {'attribute': 'has_wr', 'op': 'gte', 'value': '1'}
    ) is None
    assert parser._to_predicate(
        {'attribute': 'comp_count', 'op': 'is', 'value': 'true'}
    ) is None


def test_category_outside_the_declared_values_is_rejected(parser):
    assert parser._to_predicate(
        {'attribute': 'continent_id', 'op': 'eq', 'value': 'Atlantis'}
    ) is None


def test_continent_missing_its_underscore_is_recovered(parser):
    """The model often writes 'Europe'; WCA ids carry a leading underscore."""
    pred = parser._to_predicate(
        {'attribute': 'continent_id', 'op': 'eq', 'value': 'Europe'}
    )
    assert pred is not None and pred.value == '_Europe'


def test_numeric_value_snaps_to_a_declared_threshold(parser):
    """An off-list threshold would create a question the engine can never reask."""
    pred = parser._to_predicate(
        {'attribute': 'comp_count', 'op': 'gte', 'value': '37'}
    )
    assert pred.value in BY_KEY['comp_count'].thresholds


def test_non_numeric_value_for_numeric_attribute_is_rejected(parser):
    assert parser._to_predicate(
        {'attribute': 'comp_count', 'op': 'gte', 'value': 'lots'}
    ) is None


@pytest.mark.parametrize('payload', [
    {}, {'attribute': None, 'op': 'is', 'value': 'true'},
    {'attribute': 'has_wr', 'op': None, 'value': 'true'},
    {'attribute': '', 'op': '', 'value': ''},
])
def test_empty_or_declining_payloads_yield_no_predicate(parser, payload):
    assert parser._to_predicate(payload) is None


# ---------------------------------------------------------------------------
# Event names, groups, and set-valued attributes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('written,expected', [
    ('minx', 'minx'), ('Megaminx', 'minx'), ('MEGAMINX', 'minx'),
    ('Fewest Moves', '333fm'), ('fmc', '333fm'),
    ('3x3x3', '333'), ('333', '333'), ('oh', '333oh'),
    ('Square-1', 'sq1'), ('sq1', 'sq1'), ('Multi-Blind', '333mbf'),
])
def test_event_names_resolve_to_wca_ids(written, expected):
    assert _coerce_event_id(written) == expected


def test_unknown_event_does_not_resolve():
    assert _coerce_event_id('Rubiks Tower') is None
    assert _coerce_event_id('') is None


def test_multi_attribute_accepts_either_op(parser):
    """The value decides the op, because the model reaches for 'eq' often.

    Rejecting the mismatch used to decline "is the person ranked top 100 in
    megaminx" outright, even though the intent is unambiguous.
    """
    for op in ('has', 'eq', 'has_group'):
        pred = parser._to_predicate(
            {'attribute': 'top100_events', 'op': op, 'value': 'minx'})
        assert pred == Predicate('top100_events', 'has', 'minx'), op


def test_multi_attribute_recognises_event_groups(parser):
    for group in EVENT_GROUPS:
        pred = parser._to_predicate(
            {'attribute': 'top100_events', 'op': 'eq', 'value': group})
        assert pred == Predicate('top100_events', 'has_group', group)


def test_multi_attribute_rejects_a_non_event_value(parser):
    assert parser._to_predicate(
        {'attribute': 'top100_events', 'op': 'has', 'value': 'Mongolia'}) is None


def test_main_event_display_name_is_coerced_to_an_id(parser):
    """main_event declares no value list, so an uncoerced display name would
    match nobody and quietly answer 'no' to every question."""
    pred = parser._to_predicate(
        {'attribute': 'main_event', 'op': 'eq', 'value': 'Megaminx'})
    assert pred == Predicate('main_event', 'eq', 'minx')


def test_main_event_rejects_an_unrecognised_event(parser):
    assert parser._to_predicate(
        {'attribute': 'main_event', 'op': 'eq', 'value': 'Rubiks Tower'}) is None


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def test_disabling_game_limits_exempts_rather_than_no_ops(monkeypatch):
    """Off must mean exempt, not "no decorator".

    A route carrying no limit still inherits Flask-Limiter's 200/day + 50/hour
    defaults, which are tighter than every per-endpoint limit the game sets. A
    plain identity decorator would therefore make GAME_RATE_LIMITS=false *more*
    restrictive than leaving limits on — the opposite of what it says.
    """
    import blueprints.game as bp

    # `==`, not `is`: limiter.exempt is a bound method, so every attribute
    # access builds a new object that compares equal but not identical.
    monkeypatch.setattr(bp, 'GAME_RATE_LIMITS', False)
    assert bp._limit('5 per minute') == bp.limiter.exempt

    monkeypatch.setattr(bp, 'GAME_RATE_LIMITS', True)
    decorator = bp._limit('5 per minute')
    assert decorator != bp.limiter.exempt
    assert callable(decorator)


def test_limit_tiers_are_ordered_by_what_they_cost():
    """The endpoints that reach Claude must not be looser than the free ones."""
    def per_minute(spec):
        return int(spec.split()[0])

    assert per_minute(GAME_LIMIT_ASK) <= per_minute(GAME_LIMIT_CHEAP)
    assert per_minute(GAME_LIMIT_CHEAP) <= per_minute(GAME_LIMIT_FREE)


def test_guest_day_cap_is_present_and_finite():
    """This cap is the only thing stopping an anonymous visitor spending budget."""
    assert isinstance(MAX_GUEST_GAME_QUESTIONS, int)
    assert 0 < MAX_GUEST_GAME_QUESTIONS < 100_000


# ---------------------------------------------------------------------------
# WCA profile photos on the result card
# ---------------------------------------------------------------------------

WCA_API_HOST = 'https://www.worldcubeassociation.org'


@pytest.fixture(scope='module')
def client():
    import app as flask_app
    flask_app.app.config['TESTING'] = True
    return flask_app.app.test_client()


def test_csp_allows_wca_avatars_to_load(client):
    """Photos come from avatars.worldcubeassociation.org, the API from www.

    Both are third-party, so a tightened CSP would break them silently — the
    image just never appears and nothing is logged.
    """
    csp = client.get('/game/akinator').headers['Content-Security-Policy']

    img = next(d for d in csp.split(';') if d.strip().startswith('img-src'))
    assert 'https:' in img, f'img-src blocks the avatar host: {img.strip()!r}'

    connect = next(d for d in csp.split(';') if d.strip().startswith('connect-src'))
    assert WCA_API_HOST in connect, (
        f'connect-src blocks the WCA API, so the photo can never be looked '
        f'up: {connect.strip()!r}'
    )


@pytest.mark.parametrize('page', ['/game/akinator', '/game/solo', '/game/pvp'])
def test_result_screens_mount_a_profile_photo(client, page):
    html = client.get(page).get_data(as_text=True)
    assert 'avatar-slot' in html, f'{page} has nowhere to put the photo'
    assert 'mountAvatar(' in html, f'{page} never requests the photo'


def test_avatar_helper_skips_competitors_without_a_photo(client):
    """`is_default` marks a generic silhouette, which is worse than nothing.

    Roughly 40% of the Everyone pool has no photo, so this branch runs more
    often than not.
    """
    html = client.get('/game/akinator').get_data(as_text=True)
    assert 'avatar.is_default' in html, (
        'the helper does not check is_default — default silhouettes will show'
    )


# ---------------------------------------------------------------------------
# Cache versioning
# ---------------------------------------------------------------------------

def test_schema_version_is_stable_and_short():
    assert isinstance(_SCHEMA_VERSION, str) and len(_SCHEMA_VERSION) == 12


def test_schema_version_tracks_the_attribute_set(monkeypatch):
    """Adding or retyping an attribute must produce a different stamp.

    This is what retires cached declines: a question the old schema refused
    becomes answerable the moment an attribute lands, and the stale "I can't
    answer that" would otherwise persist invisibly.
    """
    import services.game.question_parser as qp
    recompute = lambda: hashlib.sha256(  # noqa: E731
        repr([(a.key, a.kind, a.thresholds, a.values, a.event_valued)
              for a in qp.ATTRIBUTES]).encode('utf-8')
    ).hexdigest()[:12]

    assert recompute() == _SCHEMA_VERSION

    extra = Attribute(key='hypothetical', kind='bool', question='New?')
    monkeypatch.setattr(qp, 'ATTRIBUTES', qp.ATTRIBUTES + (extra,))
    assert recompute() != _SCHEMA_VERSION


def test_cached_entry_from_an_old_schema_is_ignored(parser, monkeypatch):
    """A cache hit stamped with a different schema must not be served."""
    served = {'v': 'deadbeefcafe', 'error': 'I cannot answer that.'}
    monkeypatch.setattr(parser, '_cache_get', lambda key: served)
    # No API key path is exercised: with the client disabled the parse falls
    # through past the cache and reports unavailability rather than the stale
    # cached error, proving the entry was skipped.
    monkeypatch.setattr(parser, 'client', None)
    pred, error = parser.parse('are they top 100 in megaminx')
    assert pred is None
    assert error != served['error']


def test_cached_entry_with_the_current_stamp_is_served(parser, monkeypatch):
    entry = {'v': _SCHEMA_VERSION, 'attribute': 'has_wr', 'op': 'is', 'value': 'true'}
    monkeypatch.setattr(parser, '_cache_get', lambda key: entry)
    monkeypatch.setattr(parser, 'client', None)  # must not be needed
    pred, error = parser.parse('do they have a world record')
    assert error is None
    assert pred == Predicate('has_wr', 'is', True)
