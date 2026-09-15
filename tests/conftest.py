"""pytest configuration — skip all DB tests if the database is unreachable."""
# Test modules that need a live database. The game suites read `cuber_profiles`,
# which lives in the same TiDB instance as the WCA export.
DB_BACKED_MODULES = ('test_database', 'test_game_profiles', 'test_game_engine')

import ssl
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "db: mark test as requiring a live database connection"
    )


def pytest_collection_modifyitems(config, items):
    """Skip every DB-backed test module if the database is unreachable."""
    try:
        import certifi
        import pymysql
        from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL

        kwargs = {}
        if DB_SSL:
            kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
        c = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME,
            connect_timeout=5, **kwargs,
        )
        c.close()
    except Exception as exc:
        skip = pytest.mark.skip(reason=f"Database unreachable: {exc}")
        for item in items:
            if any(mod in str(item.fspath) for mod in DB_BACKED_MODULES):
                item.add_marker(skip)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if exitstatus == 0:
        terminalreporter.write_line(
            "All database validation tests passed. The database is up to date and consistent.",
            green=True, bold=True,
        )
