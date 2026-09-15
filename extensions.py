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
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)
