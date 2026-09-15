"""Sealed game tokens — how the app keeps a secret from the player holding it.

In Mode 1 the app picks a cuber and the player asks questions to find them. The
secret has to survive between requests, but the web app runs two Gunicorn
workers with no sticky sessions and the Fly machine suspends when idle, so
server-side memory is not an option.

The token goes to the browser instead, encrypted. A Flask session cookie would
not do: Flask signs its sessions but does not encrypt them, so anyone who can
base64-decode a cookie could read the answer and win every game. Fernet gives
authenticated encryption, so the payload is both unreadable and untamperable.

`cryptography` is already a dependency (it backs PyMySQL's auth), so this adds
nothing to the install.
"""
import base64
import hashlib
import json
import logging
import time

from cryptography.fernet import Fernet, InvalidToken

from config import SECRET_KEY

logger = logging.getLogger(__name__)

# Games are short. This bounds how long a stolen or stale token stays usable,
# and stops an abandoned game from being resumed days later.
TOKEN_TTL_SECONDS = 6 * 60 * 60


class TokenError(Exception):
    """Raised when a token is missing, corrupt, tampered with, or expired."""


def _fernet() -> Fernet:
    """Derive a Fernet key from the app secret.

    SECRET_KEY is arbitrary text, while Fernet needs exactly 32 url-safe
    base64 bytes, so it goes through SHA-256 first. That also means rotating
    SECRET_KEY invalidates in-flight games, which is the correct behaviour.
    """
    if not SECRET_KEY:
        raise TokenError('SECRET_KEY is not configured')
    digest = hashlib.sha256(SECRET_KEY.encode('utf-8')).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def seal(payload: dict) -> str:
    """Encrypt a game payload for the browser to hold."""
    body = dict(payload, _iat=int(time.time()))
    return _fernet().encrypt(json.dumps(body).encode('utf-8')).decode('ascii')


def unseal(token: str) -> dict:
    """Decrypt and validate a token. Raises TokenError on anything suspect."""
    if not token or not isinstance(token, str):
        raise TokenError('missing token')
    try:
        raw = _fernet().decrypt(token.encode('ascii'), ttl=TOKEN_TTL_SECONDS)
    except InvalidToken:
        # Covers tampering, a wrong key, and expiry alike. Deliberately not
        # distinguished in the message — a player probing the endpoint learns
        # nothing about which of those happened.
        raise TokenError('invalid or expired game token')
    except (UnicodeEncodeError, ValueError):
        raise TokenError('malformed game token')

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise TokenError('malformed game token')
    if not isinstance(payload, dict):
        raise TokenError('malformed game token')
    return payload
