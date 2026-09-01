"""Reload TiDB Serverless with the latest WCA data export.

Downloads the WCA TSV export (data) and SQL export (schema), then
drops/recreates and bulk-loads every table. Skips if the database is
already on the latest export date.

On success it records the export date and row-count stats in the `site_meta`
table, which is what the web app renders on its home and About pages — so a
refresh reaches the site with no commit or redeploy. This also means the
"already up to date" check works on a fresh CI runner, where the local
.last_export_date file doesn't exist.

Tables are bulk-loaded with LOAD DATA LOCAL INFILE, measured at ~8x the speed of
the batched INSERTs it replaced (200k rows: 48.3s -> 5.7s). The tradeoff is that MySQL's LOCAL protocol can't
abort mid-stream, so the server coerces a value its column type rejects (a
non-numeric int becomes 0) instead of raising. Rows with the wrong column count
are still dropped during staging, and --no-fast-load restores the strict
row-by-row path, which is also the automatic fallback if LOAD DATA errors.

Usage:
    python scripts/update_database.py                 # skip if already up to date
    python scripts/update_database.py --force         # reload regardless
    python scripts/update_database.py --patch-repo    # also rewrite README stats
    python scripts/update_database.py --no-fast-load  # strict row-by-row load
"""
import argparse
import io
import logging
import os
import re
import ssl
import sys
import tempfile
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

# Size of one LOAD DATA statement. TiDB runs each as a single transaction, so
# these bound the server-side memory it needs and how long the client holds the
# socket streaming into it.
#
# Both limits are needed, because they catch different shapes of table. Bytes
# alone is not enough: TiDB's memory use tracks row count, not payload size, and
# a narrow table like result_attempts (int, tinyint, bigint) packs ~2.5M rows
# into 64 MB where a wide one like results packs ~760k. The wide case loaded
# fine; the narrow one exceeded tidb_server_memory_limit and was cancelled.
LOAD_CHUNK_BYTES = 64 * 1024 * 1024
LOAD_CHUNK_ROWS = 500_000

# Divisor applied to both limits when a chunk fails and the table is retried.
LOAD_RETRY_SHRINK = 4


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
        # Required for the LOAD DATA LOCAL INFILE fast path. The server side is
        # already enabled (@@local_infile = 1 on TiDB Serverless); this opts the
        # client in. Harmless when the fast path isn't used.
        local_infile=True,
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

def _stage_tsv(zf: zipfile.ZipFile, fname: str, tmpdir: Path,
               max_bytes: int | None = None,
               max_rows: int | None = None) -> tuple[list[Path], list[str], int, int]:
    """Stream one TSV member out of the zip onto disk, ready for LOAD DATA.

    Returns (parts, columns, rows, skipped). Everything is done in bytes and
    streamed a line at a time: the export's biggest table is well over a
    gigabyte uncompressed, and reading it whole is what used to dominate this
    script's memory use.

    The staged files strip the header, normalise CRLF to LF, and drop rows whose
    column count is wrong. That last check matters more than it used to: the
    row-by-row path rejected a short row outright, whereas LOAD DATA would
    quietly pad it with NULLs.

    Output is split into parts bounded by both max_bytes and max_rows, because
    TiDB runs each LOAD DATA as a single transaction (tidb_dml_batch_size = 0)
    and a whole big table at once exhausts its memory limits. See the notes on
    LOAD_CHUNK_BYTES for why one limit alone doesn't cover it.
    """
    # Resolved here rather than as default arguments so the limits stay
    # overridable at runtime (defaults would freeze them at import time).
    max_bytes = LOAD_CHUNK_BYTES if max_bytes is None else max_bytes
    max_rows = LOAD_CHUNK_ROWS if max_rows is None else max_rows

    base = Path(fname).name
    parts: list[Path] = []
    rows = skipped = 0

    def _next_part():
        path = tmpdir / f'{base}.{len(parts):03d}.staged'
        parts.append(path)
        return open(path, 'wb', buffering=1 << 20)

    with zf.open(fname) as src:
        header = src.readline()
        if header.startswith(b'\xef\xbb\xbf'):   # utf-8-sig BOM
            header = header[3:]
        columns = header.rstrip(b'\r\n').decode('utf-8').split('\t')
        n_tabs = len(columns) - 1

        dst = _next_part()
        written = 0
        part_rows = 0
        try:
            # Plain tab-splitting, not csv.reader — WCA TSV files are unquoted,
            # and csv's default quoting misaligns columns when a name contains a
            # double quote (e.g. John "Johnny" Smith).
            for line in src:
                line = line.rstrip(b'\r\n')
                if not line:
                    continue
                if line.count(b'\t') != n_tabs:
                    skipped += 1
                    continue
                if written >= max_bytes or part_rows >= max_rows:
                    dst.close()
                    dst = _next_part()
                    written = part_rows = 0
                dst.write(line)
                dst.write(b'\n')
                written += len(line) + 1
                part_rows += 1
                rows += 1
        finally:
            dst.close()

    return parts, columns, rows, skipped


def _load_via_infile(conn: pymysql.Connection, table: str,
                     path: Path, columns: list[str]) -> int:
    """Bulk-load a staged TSV with LOAD DATA LOCAL INFILE. Returns rows loaded."""
    variables = ', '.join(f'@v{i}' for i in range(len(columns)))
    # Every field is read into a user variable first so NULLs can be recognised:
    # v1 exports wrote \N, v2 exports write the literal string 'NULL'. ESCAPED BY ''
    # keeps backslashes inside names intact, which means \N has to be matched here
    # rather than left to the server's own escape processing.
    set_clause = ', '.join(
        f"`{c}` = NULLIF(NULLIF(@v{i}, 'NULL'), '\\\\N')"
        for i, c in enumerate(columns)
    )
    sql = (
        f"LOAD DATA LOCAL INFILE {conn.escape(str(path))} "
        f"INTO TABLE `{table}` CHARACTER SET utf8mb4 "
        f"FIELDS TERMINATED BY '\\t' ESCAPED BY '' "
        f"LINES TERMINATED BY '\\n' "
        f"({variables}) SET {set_clause}"
    )
    cur = conn.cursor()
    try:
        cur.execute(sql)
        conn.commit()
        return cur.rowcount
    finally:
        cur.close()


def _load_all_parts(conn: pymysql.Connection, table: str,
                    parts: list[Path], columns: list[str]) -> int:
    total = 0
    for i, part in enumerate(parts, 1):
        total += _load_via_infile(conn, table, part, columns)
        if len(parts) > 1:
            log.info("  %s: chunk %d/%d — %d rows loaded", table, i, len(parts), total)
    return total


def _insert_from_files(conn: pymysql.Connection, table: str,
                       parts: list[Path], columns: list[str]) -> int:
    """Row-by-row fallback: batched INSERTs from the staged TSV. Returns rows loaded.

    Roughly 8x slower than LOAD DATA, but it reports per-row errors, so a value
    the column type rejects is skipped and logged rather than coerced.
    """
    col_list = ', '.join(f'`{c}`' for c in columns)
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f'INSERT INTO `{table}` ({col_list}) VALUES ({placeholders})'

    batch: list[list] = []
    total = 0
    for part in parts:
        with open(part, 'rb') as f:
            for line in f:
                row = line.rstrip(b'\n').decode('utf-8', errors='replace').split('\t')
                batch.append([None if v in (r'\N', 'NULL') else v for v in row])
                total += 1
                if len(batch) >= BATCH_SIZE:
                    _flush(conn, insert_sql, batch, table)
                    batch = []
                    if total % 500_000 == 0:
                        log.info("  %s: %d rows loaded...", table, total)

    if batch:
        _flush(conn, insert_sql, batch, table)
    return total


def _reset_table(conn: pymysql.Connection, table: str) -> None:
    """Reconnect if the socket died, then empty the table so a retry can't double up.

    A LOAD DATA that exceeds the server's limits can be killed mid-transfer,
    which leaves the connection unusable — every later statement would fail
    until it is re-established. Reconnecting on the same object keeps the
    caller's handle valid for the rest of the run.
    """
    if not conn.open:
        conn.connect()
    else:
        try:
            conn.rollback()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            conn.connect()

    cur = conn.cursor()
    try:
        cur.execute(f"TRUNCATE TABLE `{table}`")
        conn.commit()
    finally:
        cur.close()


def load_table(conn: pymysql.Connection, table: str, zf: zipfile.ZipFile,
               fname: str, create_sql: str | None, tmpdir: Path,
               fast_load: bool = True) -> None:
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
    cur.close()

    parts, columns, rows, skipped = _stage_tsv(zf, fname, tmpdir)
    try:
        loaded = None
        # Two goes at the fast path before falling back. The retry re-stages
        # into smaller chunks, since the usual reason for failure is a chunk the
        # server hasn't got the memory for, and that is a property of the table
        # rather than bad luck. Catch Exception rather than DatabaseError — when
        # the server kills the connection mid-transfer, pymysql's own error
        # handling raises AttributeError on the dead socket.
        for attempt in (1, 2) if fast_load else ():
            try:
                loaded = _load_all_parts(conn, table, parts, columns)
                break
            except Exception as e:
                log.warning("  LOAD DATA failed for %s (attempt %d/2): %s: %s",
                            table, attempt, type(e).__name__, e)
                # A failed chunk may have committed rows, so start the table over.
                _reset_table(conn, table)
                if attempt == 1:
                    for part in parts:
                        part.unlink(missing_ok=True)
                    log.info("  %s: re-staging with %dx smaller chunks",
                             table, LOAD_RETRY_SHRINK)
                    parts, columns, rows, skipped = _stage_tsv(
                        zf, fname, tmpdir,
                        max_bytes=LOAD_CHUNK_BYTES // LOAD_RETRY_SHRINK,
                        max_rows=LOAD_CHUNK_ROWS // LOAD_RETRY_SHRINK,
                    )

        if loaded is None:
            if fast_load:
                log.warning("  %s: falling back to row-by-row INSERT", table)
            loaded = _insert_from_files(conn, table, parts, columns)
    finally:
        for part in parts:
            part.unlink(missing_ok=True)

    if skipped:
        log.warning("  %s: skipped %d malformed rows", table, skipped)
    if loaded != rows:
        log.warning("  %s: loaded %d rows but staged %d", table, loaded, rows)
    log.info("  %s: done (%d rows)", table, loaded)


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


def run(force: bool = False, patch_repo: bool = False, fast_load: bool = True) -> None:
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

    # Load each TSV into TiDB. Tables are staged to disk one at a time and the
    # staged file is deleted straight after, so peak disk use is the largest
    # single table rather than the whole export.
    log.info("Bulk load mode: %s", 'LOAD DATA LOCAL INFILE' if fast_load else 'batched INSERT')
    conn = get_connection()
    try:
        with tempfile.TemporaryDirectory(prefix='wca-export-') as tmp:
            tmpdir = Path(tmp)
            with zipfile.ZipFile(io.BytesIO(tsv_zip)) as z:
                tsv_files = sorted(n for n in z.namelist() if n.endswith('.tsv'))
                log.info("Found %d TSV files", len(tsv_files))
                for fname in tsv_files:
                    table = table_name_from_tsv(fname)
                    if not table:
                        continue
                    load_table(conn, table, z, fname, create_stmts.get(table),
                               tmpdir, fast_load=fast_load)
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
    parser.add_argument('--no-fast-load', action='store_true',
                        help='Load with batched INSERTs instead of LOAD DATA LOCAL INFILE: '
                             'about 8x slower, but rejects and logs bad rows individually '
                             'rather than letting the server coerce them')
    args = parser.parse_args()
    run(force=args.force, patch_repo=args.patch_repo, fast_load=not args.no_fast_load)
