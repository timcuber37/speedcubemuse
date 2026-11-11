"""Downloading and reading the WCA results export.

Shared by update_database.py, which loads the export into TiDB, and
build_cuber_profiles.py, which computes the Guess the Cuber matrix straight from
the export files. Reading the files directly keeps the heavy aggregation on the
machine running the job. Run inside TiDB, the same work cost the most of any
query the refresh made and failed against its memory limit.
"""
import io
import json
import logging
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path

import requests

log = logging.getLogger(__name__)

WCA_EXPORT_API = "https://www.worldcubeassociation.org/api/v0/export/public"


def fetch_metadata() -> dict:
    """The export API's description of the latest export."""
    resp = requests.get(WCA_EXPORT_API, timeout=30)
    resp.raise_for_status()
    return resp.json()


def export_date(meta: dict) -> str:
    """The API may return 'export_date' as an ISO datetime; take the date part."""
    return (meta.get('export_date') or '')[:10]


def find_url(meta: dict, keys: list[str]) -> str | None:
    """Download URLs have appeared under several field names."""
    for k in keys:
        if meta.get(k):
            return meta[k]
    return None


def tsv_url(meta: dict) -> str | None:
    return find_url(meta, ['tsv_url', 'tsv_zip_url', 'tsv_download_url'])


def sql_url(meta: dict) -> str | None:
    return find_url(meta, ['sql_url', 'sql_zip_url', 'sql_download_url'])


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


def table_name_from_tsv(filename: str) -> str:
    """'WCA_export_Results.tsv' -> 'Results', 'results.tsv' -> 'results'."""
    name = Path(filename).name          # strip any directory prefix
    name = re.sub(r'^WCA_export_', '', name)
    return name.removesuffix('.tsv')


# v1 exports wrote \N for NULL, v2 exports write the literal string 'NULL'.
_NULLS = (b'NULL', b'\\N')


class ExportReader:
    """The TSV export's tables, read row by row from the zip.

    The parsing rules match update_database.py's loader, so a table read here
    holds the same rows the database does. Fields are unquoted and split on tabs
    (a name can contain a double quote). The header may start with a BOM. Rows
    with the wrong number of fields are dropped.
    """

    def __init__(self, source: bytes | str | Path):
        self._zip = zipfile.ZipFile(
            io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)
        self._members = {
            table_name_from_tsv(n).lower(): n
            for n in self._zip.namelist() if n.endswith('.tsv')
        }

    def export_date(self) -> str | None:
        """The export's own date from metadata.json, e.g. '2026-11-11'."""
        try:
            return export_date(json.loads(self._zip.read('metadata.json'))) or None
        except (KeyError, ValueError):
            return None

    def rows(self, table: str, columns: tuple[str, ...]) -> Iterator[tuple]:
        """Yield the named columns of every row as strings, with NULL as None."""
        member = self._members.get(table.lower())
        if member is None:
            raise KeyError(f'{table}.tsv is not in the export')
        with self._zip.open(member) as src:
            header = src.readline()
            if header.startswith(b'\xef\xbb\xbf'):
                header = header[3:]
            names = header.rstrip(b'\r\n').decode('utf-8').split('\t')
            try:
                wanted = [names.index(c) for c in columns]
            except ValueError as e:
                raise KeyError(f'{table}.tsv has no column {e}') from None
            width = len(names)
            for line in src:
                fields = line.rstrip(b'\r\n').split(b'\t')
                if len(fields) != width:
                    continue   # blank or malformed, exactly as the loader skips them
                yield tuple(
                    None if (v := fields[i]) in _NULLS else v.decode('utf-8', 'replace')
                    for i in wanted
                )
