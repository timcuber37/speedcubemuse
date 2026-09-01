"""Database-backed site metadata (WCA export date + row-count stats).

The weekly refresh job writes these values into a `site_meta` table right
after it reloads the WCA export; the web app reads them at render time. That
keeps the numbers shown on the site in sync with the database without
redeploying — previously they were regex-patched into the templates, so a
refresh meant a commit and a push.

`site_meta` is safe from the reload: update_database.py only drops tables that
appear in the WCA TSV export, and this one never does.
"""
import json
import logging
import ssl
import threading
import time
from datetime import datetime

import certifi
import pymysql

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL

logger = logging.getLogger(__name__)

META_KEY = 'db_stats'

# Order the stat cards appear in on the About page.
STAT_ORDER = ['Competitors', 'Results', 'Competitions', 'Events']

# Last known good values, baked in so the pages still render if the database
# is unreachable. Refreshed automatically in the DB; only update these if you
# want the offline fallback to be less stale.
FALLBACK = {
    'export_date': '2026-08-02',
    'stats': {
        'Competitors': 293935,
        'Results': 6764347,
        'Competitions': 18280,
        'Events': 17,
    },
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS `site_meta` (
  `key`        VARCHAR(64) NOT NULL PRIMARY KEY,
  `value`      TEXT NOT NULL,
  `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
)
"""

# The app re-reads at most once an hour; the export only changes weekly.
_CACHE_TTL_SECONDS = 3600
_cache_value = None
_cache_expires_at = 0.0
_cache_lock = threading.Lock()


def format_export_date(export_date: str) -> str:
    """'2026-08-02' -> 'August 2, 2026'. Returns the input unchanged if unparseable."""
    try:
        dt = datetime.strptime(export_date, '%Y-%m-%d')
    except (ValueError, TypeError):
        return export_date
    # Built by hand rather than with %-d/%#d, which differ across platforms.
    return f"{dt.strftime('%B')} {dt.day}, {dt.year}"


# ---------------------------------------------------------------------------
# Connection-taking helpers — used by scripts/update_database.py, which already
# has its own connection open.
# ---------------------------------------------------------------------------

def ensure_table(conn) -> None:
    cur = conn.cursor()
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()
    cur.close()


def read_meta(conn) -> dict | None:
    """Return the stored metadata dict, or None if absent/unreadable."""
    cur = conn.cursor()
    try:
        cur.execute('SELECT `value` FROM `site_meta` WHERE `key` = %s', (META_KEY,))
        row = cur.fetchone()
    except pymysql.err.ProgrammingError:
        return None  # table not created yet
    finally:
        cur.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (json.JSONDecodeError, TypeError):
        logger.warning("site_meta.%s is not valid JSON — ignoring", META_KEY)
        return None


def write_meta(conn, stats: dict, export_date: str) -> None:
    """Upsert the stats + export date the site renders from."""
    ensure_table(conn)
    payload = json.dumps({'export_date': export_date, 'stats': stats})
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO `site_meta` (`key`, `value`) VALUES (%s, %s) '
        'ON DUPLICATE KEY UPDATE `value` = VALUES(`value`)',
        (META_KEY, payload),
    )
    conn.commit()
    cur.close()


# ---------------------------------------------------------------------------
# App-side read path
# ---------------------------------------------------------------------------

def _connect():
    kwargs = {}
    if DB_SSL:
        # certifi rather than the OS trust store, matching WCAService — some
        # hosts carry a stale root that breaks TiDB Cloud cert verification.
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        # Short timeouts: this runs inline in a page render, so a database
        # hiccup should fall back to FALLBACK fast rather than hang the page.
        connect_timeout=5,
        read_timeout=10,
        **kwargs,
    )


def _build(meta: dict) -> dict:
    """Shape stored metadata into what the templates expect."""
    stats = meta.get('stats') or {}
    export_date = meta.get('export_date') or ''
    # Follow STAT_ORDER, then append any labels added later that aren't in it.
    labels = [l for l in STAT_ORDER if l in stats]
    labels += [l for l in stats if l not in STAT_ORDER]
    return {
        'export_date': export_date,
        'export_date_display': format_export_date(export_date),
        'stats': [(label, f'{stats[label]:,}') for label in labels],
    }


def get_site_meta(force_refresh: bool = False) -> dict:
    """Stats + export date for the templates, cached for an hour per process.

    Never raises: falls back to the last known good values if the database is
    unreachable, so a DB outage can't take the public pages down with it.
    """
    global _cache_value, _cache_expires_at

    now = time.monotonic()
    if not force_refresh and _cache_value is not None and now < _cache_expires_at:
        return _cache_value

    with _cache_lock:
        # Another thread may have refreshed while we waited for the lock.
        now = time.monotonic()
        if not force_refresh and _cache_value is not None and now < _cache_expires_at:
            return _cache_value

        meta = None
        try:
            conn = _connect()
            try:
                meta = read_meta(conn)
            finally:
                conn.close()
        except Exception as e:
            logger.warning("Could not read site_meta, using fallback values: %s", e)

        if meta is None:
            meta = FALLBACK

        _cache_value = _build(meta)
        _cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        return _cache_value
