"""Reload TiDB Serverless with the latest WCA data export.

Downloads the WCA TSV export (data) and SQL export (schema), then
drops/recreates and bulk-loads every table. Skips if the database is
already on the latest export date.

On success it records the export date and row-count stats in the `site_meta`
table, which is what the web app renders on its home and About pages — so a
refresh reaches the site with no commit or redeploy. This also means the
"already up to date" check works on a fresh CI runner, where the local
.last_export_date file doesn't exist.

Usage:
    python scripts/update_database.py               # skip if already up to date
    python scripts/update_database.py --force       # reload regardless
    python scripts/update_database.py --patch-repo  # also rewrite README stats
"""
import argparse
import io
import logging
import os
import re
import ssl
import sys
import zipfile
from pathlib import Path

import certifi
import pymysql
import requests
from dotenv import load_dotenv

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))
load_dotenv()

from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL
from services import site_meta

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

WCA_EXPORT_API = "https://www.worldcubeassociation.org/api/v0/export/public"
STATE_FILE = _HERE / ".last_export_date"

# Rows per INSERT batch — large enough to be fast, small enough to avoid timeouts
BATCH_SIZE = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_connection() -> pymysql.Connection:
    kwargs = {}
    if DB_SSL:
        # Verify against the certifi CA bundle rather than the OS trust store:
        # on Windows the system store can carry a stale/expired root that makes
        # TiDB Cloud's valid cert fail verification. certifi is what requests
        # already uses successfully for the export download above.
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    return pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        autocommit=False,
        **kwargs,
    )


def download(url: str, label: str) -> bytes:
    log.info("Downloading %s ...", label)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get('content-length', 0))
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        buf.extend(chunk)
        if total:
            pct = len(buf) * 100 // total
            print(f"\r  {pct:3d}%  {len(buf) // 1_048_576} / {total // 1_048_576} MB",
                  end='', flush=True)
    print()
    log.info("  Downloaded %d MB", len(buf) // 1_048_576)
    return bytes(buf)


def parse_create_statements(sql_text: str) -> dict[str, str]:
    """Return {table_name: CREATE TABLE ...;} from a MySQL dump."""
    pattern = re.compile(
        r'(CREATE TABLE `([^`]+)`\s*\(.*?\)\s*(?:ENGINE[^;]*)?;)',
        re.DOTALL | re.IGNORECASE,
    )
    return {m.group(2): m.group(1) for m in pattern.finditer(sql_text)}


def table_name_from_tsv(filename: str) -> str:
    """'WCA_export_Results.tsv' → 'Results'"""
    name = Path(filename).name          # strip any directory prefix
    name = re.sub(r'^WCA_export_', '', name)
    return name.removesuffix('.tsv')


# ---------------------------------------------------------------------------
# Batch flusher — falls back to row-by-row on error so one bad row can't
# abort an entire table load, and logs exactly what caused the failure.
# ---------------------------------------------------------------------------

def _insert_individually(conn, insert_sql: str, batch: list, table: str) -> None:
    """Insert a batch one row at a time, reconnecting on connection loss so a
    dropped connection (or an oversized multi-row packet) can't abort the load."""
    cur = conn.cursor()
    for row_data in batch:
        try:
            cur.execute(insert_sql, row_data)
        except (pymysql.err.OperationalError, pymysql.err.InterfaceError):
            # Connection went away — reconnect and retry this single row once.
            conn.ping(reconnect=True)
            cur = conn.cursor()
            try:
                cur.execute(insert_sql, row_data)
            except pymysql.err.DatabaseError as e:
                log.warning("  Skipping bad row in %s: %s | values: %s", table, e, row_data)
        except pymysql.err.DatabaseError as e:
            log.warning("  Skipping bad row in %s: %s | values: %s", table, e, row_data)
    conn.commit()
    cur.close()


def _flush(conn, insert_sql: str, batch: list, table: str) -> None:
    # Fresh cursor each call: after a reconnect the previous cursor is bound to a
    # dead socket and would raise InterfaceError on every subsequent batch.
    cur = conn.cursor()
    try:
        cur.executemany(insert_sql, batch)
        conn.commit()
    except (pymysql.err.OperationalError, pymysql.err.InterfaceError) as e:
        # Connection dropped mid-query (TiDB idle/timeout, network blip, or an
        # oversized packet). Reconnect and retry row-by-row with smaller packets.
        log.warning("  Connection lost while loading %s (%s) — reconnecting", table, e)
        conn.ping(reconnect=True)
        _insert_individually(conn, insert_sql, batch, table)
    except pymysql.err.DatabaseError:
        # Bad data somewhere in the batch — isolate the offending row(s).
        try:
            conn.rollback()
        except Exception:
            pass
        _insert_individually(conn, insert_sql, batch, table)
    finally:
        try:
            cur.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Per-table loader
# ---------------------------------------------------------------------------

def load_table(conn: pymysql.Connection, table: str,
               tsv_bytes: bytes, create_sql: str | None) -> None:
    cur = conn.cursor()

    if create_sql:
        log.info("Recreating table: %s", table)
        cur.execute(f"DROP TABLE IF EXISTS `{table}`")
        cur.execute(create_sql)
    else:
        log.info("Truncating table (no schema found): %s", table)
        try:
            cur.execute(f"TRUNCATE TABLE `{table}`")
        except pymysql.err.ProgrammingError:
            log.warning("  Table %s does not exist and no schema available — skipping", table)
            cur.close()
            return

    # Parse TSV with plain tab-splitting — WCA TSV files are unquoted, and using
    # csv.reader's default quoting causes column misalignment when names contain
    # double-quote characters (e.g. John "Johnny" Smith).
    wrapper = io.TextIOWrapper(io.BytesIO(tsv_bytes), encoding='utf-8-sig')
    lines = iter(wrapper)

    columns = next(lines).rstrip('\r\n').split('\t')
    col_list = ', '.join(f'`{c}`' for c in columns)
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f'INSERT INTO `{table}` ({col_list}) VALUES ({placeholders})'
    n_cols = len(columns)

    batch: list[list] = []
    total = 0
    skipped = 0

    for line in lines:
        row = line.rstrip('\r\n').split('\t')
        if len(row) != n_cols:
            skipped += 1
            continue
        # v1 exports used \N for NULL; v2 exports use the literal string 'NULL'
        batch.append([None if v in (r'\N', 'NULL') else v for v in row])
        total += 1
        if len(batch) >= BATCH_SIZE:
            _flush(conn, insert_sql, batch, table)
            batch = []
            if total % 500_000 == 0:
                log.info("  %s: %d rows loaded...", table, total)

    if batch:
        _flush(conn, insert_sql, batch, table)

    if skipped:
        log.warning("  %s: skipped %d malformed rows", table, skipped)
    log.info("  %s: done (%d rows)", table, total)
    cur.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def last_loaded_export_date() -> str:
    """Export date currently in the database.

    Read from `site_meta` first so the check works anywhere with database
    credentials — a CI runner has no .last_export_date file, and without this
    every scheduled run would do a full reload even with no new export. The
    local file is the fallback for a database written before site_meta existed.
    """
    try:
        conn = get_connection()
        try:
            meta = site_meta.read_meta(conn)
        finally:
            conn.close()
        if meta and meta.get('export_date'):
            return meta['export_date']
    except Exception as e:
        log.warning("Could not read last export date from site_meta: %s", e)

    return STATE_FILE.read_text().strip() if STATE_FILE.exists() else ''


def run(force: bool = False, patch_repo: bool = False) -> None:
    # Check for a newer export
    log.info("Fetching export metadata from WCA API...")
    meta_resp = requests.get(WCA_EXPORT_API, timeout=30)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    # The API may return 'export_date' as ISO datetime — take the date part
    export_date = (meta.get('export_date') or '')[:10]
    last_loaded = last_loaded_export_date()

    log.info("Latest export: %s  |  Last loaded: %s", export_date, last_loaded or 'never')

    if not force and export_date and export_date == last_loaded:
        log.info("Already up to date. Run with --force to reload anyway.")
        return

    # Resolve download URLs (field names may vary; try common variants)
    def _find_url(keys: list[str]) -> str | None:
        for k in keys:
            if meta.get(k):
                return meta[k]
        return None

    tsv_url = _find_url(['tsv_url', 'tsv_zip_url', 'tsv_download_url'])
    sql_url = _find_url(['sql_url', 'sql_zip_url', 'sql_download_url'])

    if not tsv_url:
        log.error("Could not find TSV download URL in API response: %s", list(meta.keys()))
        sys.exit(1)

    # Download
    tsv_zip = download(tsv_url, 'TSV export')
    sql_zip = download(sql_url, 'SQL export') if sql_url else None

    # Parse schema from SQL dump
    create_stmts: dict[str, str] = {}
    if sql_zip:
        with zipfile.ZipFile(io.BytesIO(sql_zip)) as z:
            sql_names = [n for n in z.namelist() if n.endswith('.sql')]
            if sql_names:
                sql_text = z.read(sql_names[0]).decode('utf-8', errors='replace')
                create_stmts = parse_create_statements(sql_text)
                log.info("Parsed %d CREATE TABLE statements from SQL dump", len(create_stmts))
            else:
                log.warning("No .sql file found inside SQL zip")

    # Load each TSV into TiDB
    conn = get_connection()
    try:
        with zipfile.ZipFile(io.BytesIO(tsv_zip)) as z:
            tsv_files = sorted(n for n in z.namelist() if n.endswith('.tsv'))
            log.info("Found %d TSV files", len(tsv_files))
            for fname in tsv_files:
                table = table_name_from_tsv(fname)
                if not table:
                    continue
                load_table(conn, table, z.read(fname), create_stmts.get(table))
    finally:
        conn.close()

    # Record the loaded export date
    STATE_FILE.write_text(export_date)
    log.info("Database updated to export date: %s", export_date)

    # Query stats and publish them to the site
    print_and_update_stats(export_date, patch_repo=patch_repo)


def print_and_update_stats(export_date: str, patch_repo: bool = False) -> None:
    """Query row counts from TiDB and publish them to `site_meta`."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        stats = {}
        for sql, label in [
            ('SELECT COUNT(*) FROM `persons`',                    'Competitors'),
            ('SELECT COUNT(*) FROM `results`',                    'Results'),
            ('SELECT COUNT(*) FROM `competitions`',               'Competitions'),
            ('SELECT COUNT(*) FROM `events` WHERE `rank` < 900',  'Events'),
        ]:
            cur.execute(sql)
            stats[label] = cur.fetchone()[0]
        cur.close()

        # Publish on the same connection — this is what the site renders from.
        site_meta.write_meta(conn, stats, export_date)
        log.info("site_meta updated with latest stats and export date: %s", export_date)
    finally:
        conn.close()

    print('\n' + '=' * 40)
    print('  Database stats')
    print('=' * 40)
    for label, count in stats.items():
        print(f'  {label:<15} {count:>10,}')
    print('=' * 40 + '\n')

    if patch_repo:
        _patch_readme(stats, export_date)


def _patch_readme(stats: dict, export_date: str) -> None:
    """Update the stat table and export date in README.md."""
    readme_path = _HERE.parent / 'README.md'
    if not readme_path.exists():
        log.warning("README.md not found at %s — skipping auto-update", readme_path)
        return

    text = readme_path.read_text(encoding='utf-8')

    # Update each markdown table row, e.g. "| Competitors | 281,645 |"
    for label, count in stats.items():
        text = re.sub(
            r'(\|\s*' + re.escape(label) + r'\s*\|\s*)[^|]+(\|)',
            rf'\g<1>{count:,} \g<2>',
            text,
        )

    # Update the export date, e.g. "(May 21, 2026)"
    formatted_date = site_meta.format_export_date(export_date)

    # In the README the date sits after the markdown link: "](url) (May 21, 2026)"
    # Anchor to the closing paren of the URL so we don't accidentally match the URL itself.
    text = re.sub(
        r'(export/results\)\s*\()[^)]+(\))',
        rf'\g<1>{formatted_date}\g<2>',
        text,
    )

    readme_path.write_text(text, encoding='utf-8')
    log.info("README.md updated with latest stats and export date: %s", formatted_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update TiDB with latest WCA data export')
    parser.add_argument('--force', action='store_true',
                        help='Reload even if already up to date')
    parser.add_argument('--patch-repo', action='store_true',
                        help='Also rewrite the stats table in README.md (local runs only; '
                             'the site itself reads from site_meta)')
    args = parser.parse_args()
    run(force=args.force, patch_repo=args.patch_repo)
