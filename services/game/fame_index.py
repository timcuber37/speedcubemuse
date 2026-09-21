"""The ranked fame index behind "see who each difficulty includes".

The difficulty tiers are rank slices of one number — the fame score that
`scripts/build_cuber_profiles.py` computes — and that number was invisible to
players. Someone who wondered why a cuber they know well sits in Hard rather
than Easy had nothing to look at.

This builds the index from the same resident rows the game plays over rather
than from a separate snapshot, so the page cannot drift from the pools it
describes: the weekly rebuild moves a competitor between tiers and the page
moves them with it.

Only the elite tiers (1-3) are ranked. Tier 4 — the rest of the Everyone pool —
is admitted by competition count, not by fame, so ranking it would be ordering
rows on a number that played no part in putting them there.
"""
import logging
import threading
import time

from .attributes import (DEFAULT_EVENT_TIER, EVENT_TIERS, country_label,
                         event_label)
from . import profiles

logger = logging.getLogger(__name__)

# The whole elite pool is listed, not a top slice. Cutting at the Normal
# boundary would be the natural-looking choice and is wrong for this page: tiers
# 1 and 2 are exactly the first 1,000 rows, so a top-1,000 table would answer
# "who does Hard include?" — the one difficulty a reader is most likely to be
# asking about — with an empty list.
#
# A guard rather than a limit. The pool is everyone with a continental record or
# a current world top 100, which is ~1,800 and grows by tens per rebuild; if the
# admission rule is ever widened this keeps the page from quietly turning into a
# multi-megabyte download.
MAX_ROWS = 5000

# The elite pool is only rebuilt weekly, and `profiles` already caches the rows
# it is derived from for an hour. Matching that TTL keeps the two from expiring
# out of step, which would rebuild this payload off half-stale rows.
_CACHE_TTL_SECONDS = 3600

_cache: dict | None = None
_cache_expires_at: float = 0.0
_cache_lock = threading.Lock()


def _row(rank: int, profile: dict) -> dict:
    """One table row: the identity, the score, and why the score is what it is.

    Zero and False fields are left out rather than sent. Across 1,000 rows the
    absent keys are most of them — few competitors hold a continental record and
    fewer a world one — and the client already reads a missing key as "no badge",
    so spelling them out would add roughly a third to the payload to say nothing.
    """
    attrs = profile['attrs']
    event = attrs.get('main_event')

    row = {
        'rank': rank,
        'id': profile['wca_id'],
        'name': profile['name'],
        # The same apostrophe restoration the questions and result cards use —
        # 'Cote d_Ivoire' reaches a reader as written otherwise.
        'country': country_label(profile['country_id']),
        'continent': (profile['continent_id'] or '').lstrip('_'),
        'fame': profile['fame'],
        'tier': profile['tier'],
    }

    if event:
        row['event'] = event_label(event)
        row['etier'] = EVENT_TIERS.get(event, DEFAULT_EVENT_TIER)

    # Singles and averages are separate records in the WCA tables and separate
    # attributes here, but a reader counting someone's world records counts both.
    wr = int(attrs.get('wr_single_count') or 0) + int(attrs.get('wr_average_count') or 0)
    for key, value in (('wr', wr),
                       ('cr', int(attrs.get('cr_count') or 0)),
                       ('top100', int(attrs.get('top100_event_count') or 0)),
                       ('podium', bool(attrs.get('has_worlds_podium'))),
                       ('active', bool(attrs.get('is_active')))):
        if value:
            row[key] = value

    return row


def _build() -> dict:
    """Rank the elite pool.

    Returns empty rows if the matrix is unreadable; callers surface that as an
    outage rather than as "nobody qualifies".
    """
    pool = profiles.load_profiles(max(profiles.TIERS['hard']))
    if not pool:
        return {'rows': [], 'pool_size': 0, 'tier_sizes': {}}

    # `load_profiles` reads `tier <= 3`, so the rows are already the elite pool.
    # Sorting by fame reproduces the ranking the builder sliced the tiers on —
    # the tie-break on wca_id is there only so the order is stable between
    # workers, which each sort their own copy.
    ranked = sorted(pool, key=lambda p: (-p['fame'], p['wca_id']))

    tier_sizes: dict[int, int] = {}
    for profile in ranked:
        tier_sizes[profile['tier']] = tier_sizes.get(profile['tier'], 0) + 1

    return {
        'rows': [_row(i, p) for i, p in enumerate(ranked[:MAX_ROWS], start=1)],
        'pool_size': len(ranked),
        # Keyed by tier number so the page can state the cuts from the data
        # instead of repeating 300 and 1,000 in prose that nothing keeps honest.
        'tier_sizes': {str(t): n for t, n in sorted(tier_sizes.items())},
    }


def get_index() -> dict:
    """The cached index payload. Empty `rows` means the matrix is unavailable."""
    global _cache, _cache_expires_at

    now = time.monotonic()
    if _cache is not None and now < _cache_expires_at:
        return _cache

    with _cache_lock:
        # Another thread may have rebuilt it while we waited for the lock.
        if _cache is not None and time.monotonic() < _cache_expires_at:
            return _cache

        try:
            built = _build()
        except Exception as e:
            logger.warning('could not build the fame index: %s', e)
            built = _cache or {'rows': [], 'pool_size': 0, 'tier_sizes': {}}

        _cache = built
        _cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        return _cache
