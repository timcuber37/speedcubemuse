"""Flask extensions, constructed here so blueprints can import them.

The limiter lives outside app.py because Flask-Limiter only registers a limit
when `@limiter.limit(...)` decorates a view *before* the route is registered.
Applying it afterwards to `app.view_functions[...]` silently does nothing — the
call returns a wrapper that never replaces the registered view. Blueprints
therefore need the limiter at import time, and importing it from app.py would
be a cycle.

Note the in-memory storage: counters are per-worker and reset when the Fly
machine suspends. That is a deliberate tradeoff inherited from the original
setup — it keeps the app dependency-free at the cost of limits being
approximate. Anything that must be enforced exactly belongs in the database.
"""
import os

from flask import request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Fly sets FLY_APP_NAME in every machine it runs.
_ON_FLY = bool(os.getenv('FLY_APP_NAME'))


def client_ip() -> str:
    """The visitor's IP address, which every rate limit is keyed on.

    On Fly, requests arrive from Fly's proxy, so the socket address is the
    proxy's and would put every visitor in one shared bucket. The proxy passes
    the real address in Fly-Client-IP, which clients can't override. Off Fly
    the header would come straight from the client, so it's only read on Fly.
    """
    if _ON_FLY:
        ip = request.headers.get('Fly-Client-IP')
        if ip:
            return ip
    return get_remote_address()


limiter = Limiter(
    key_func=client_ip,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)
