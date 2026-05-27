"""pytest configuration — skip all DB tests if the database is unreachable."""
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
    """Skip every test in test_database.py if the DB is unreachable."""
    try:
        import pymysql
        from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL

        kwargs = {}
        if DB_SSL:
            kwargs['ssl'] = ssl.create_default_context()
        c = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME,
            connect_timeout=5, **kwargs,
        )
        c.close()
    except Exception as exc:
        skip = pytest.mark.skip(reason=f"Database unreachable: {exc}")
        for item in items:
            if "test_database" in str(item.fspath):
                item.add_marker(skip)
