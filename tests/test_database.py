"""Integration tests to validate TiDB database contents after a WCA data export load.

Run with:
    python -m pytest tests/test_database.py -v

Requires a populated database (run scripts/update_database.py first).
"""
import re
import ssl
import sys
from pathlib import Path

import pymysql
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def conn():
    kwargs = {}
    if DB_SSL:
        kwargs['ssl'] = ssl.create_default_context()
    connection = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        charset='utf8mb4',
        autocommit=True,
        **kwargs,
    )
    yield connection
    connection.close()


def query_one(conn, sql, args=None):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()


def query_all(conn, sql, args=None):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

EXPECTED_TABLES = [
    'persons',
    'results',
    'result_attempts',
    'competitions',
    'events',
    'ranks_single',
    'ranks_average',
    'countries',
    'championships',
    'continents',
    'scrambles',
    'eligible_country_iso2s_for_championship',
]

@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_exists(conn, table):
    row = query_one(conn, "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = %s AND table_name = %s", (DB_NAME, table))
    assert row[0] == 1, f"Table '{table}' does not exist"


# ---------------------------------------------------------------------------
# Minimum row counts (sanity-check that data was actually loaded)
# ---------------------------------------------------------------------------

MIN_ROW_COUNTS = {
    'persons': 200_000,
    'results': 5_000_000,
    'competitions': 15_000,
    'events': 17,
    'ranks_single': 100_000,
    'ranks_average': 100_000,
    'countries': 150,
    'result_attempts': 20_000_000,
}

@pytest.mark.parametrize("table,minimum", MIN_ROW_COUNTS.items())
def test_minimum_row_count(conn, table, minimum):
    row = query_one(conn, f"SELECT COUNT(*) FROM `{table}`")
    assert row[0] >= minimum, f"Table '{table}' has {row[0]:,} rows (expected >= {minimum:,})"


# ---------------------------------------------------------------------------
# Persons table integrity
# ---------------------------------------------------------------------------

def test_persons_no_null_wca_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM persons WHERE wca_id IS NULL OR wca_id = ''")
    assert row[0] == 0, f"{row[0]} persons have a NULL or empty wca_id"


def test_persons_no_null_name(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM persons WHERE name IS NULL OR name = ''")
    assert row[0] == 0, f"{row[0]} persons have a NULL or empty name"


def test_persons_wca_id_format(conn):
    """WCA IDs follow the pattern YYYY + 4 uppercase letters + 2 digits (e.g. 2003PHAM01)."""
    row = query_one(conn, "SELECT COUNT(*) FROM persons WHERE wca_id NOT REGEXP '^[0-9]{4}[A-Z]{4}[0-9]{2}$'")
    assert row[0] == 0, f"{row[0]} persons have malformed wca_id values"


def test_persons_no_null_country(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM persons WHERE country_id IS NULL OR country_id = ''")
    assert row[0] == 0, f"{row[0]} persons have a NULL or empty country_id"


def test_persons_gender_values(conn):
    """Gender must be 'm', 'f', 'o', or NULL."""
    row = query_one(conn, "SELECT COUNT(*) FROM persons WHERE gender NOT IN ('m', 'f', 'o') AND gender IS NOT NULL")
    assert row[0] == 0, f"{row[0]} persons have unexpected gender values"


# ---------------------------------------------------------------------------
# Results table integrity
# ---------------------------------------------------------------------------

def test_results_no_null_person_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM results WHERE person_id IS NULL OR person_id = ''")
    assert row[0] == 0, f"{row[0]} results have a NULL or empty person_id"


def test_results_no_null_competition_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM results WHERE competition_id IS NULL OR competition_id = ''")
    assert row[0] == 0, f"{row[0]} results have a NULL or empty competition_id"


def test_results_no_null_event_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM results WHERE event_id IS NULL OR event_id = ''")
    assert row[0] == 0, f"{row[0]} results have a NULL or empty event_id"


def test_results_best_values_valid(conn):
    """best must be > 0, -1 (DNF), or -2 (DNS). 0 means no best, also acceptable."""
    row = query_one(conn, "SELECT COUNT(*) FROM results WHERE best < -2")
    assert row[0] == 0, f"{row[0]} results have invalid best values (< -2)"


def test_results_average_values_valid(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM results WHERE average < -2")
    assert row[0] == 0, f"{row[0]} results have invalid average values (< -2)"


# ---------------------------------------------------------------------------
# Result attempts integrity
# ---------------------------------------------------------------------------

def test_result_attempts_no_orphans(conn):
    """Every attempt must reference an existing result."""
    row = query_one(conn, """
        SELECT COUNT(*) FROM result_attempts ra
        LEFT JOIN results r ON ra.result_id = r.id
        WHERE r.id IS NULL
    """)
    assert row[0] == 0, f"{row[0]} result_attempts reference non-existent result IDs"


def test_result_attempts_attempt_number_range(conn):
    """Attempt numbers should be 1–5."""
    row = query_one(conn, "SELECT COUNT(*) FROM result_attempts WHERE attempt_number < 1 OR attempt_number > 5")
    assert row[0] == 0, f"{row[0]} result_attempts have out-of-range attempt_number"


def test_result_attempts_value_valid(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM result_attempts WHERE value < -2")
    assert row[0] == 0, f"{row[0]} result_attempts have invalid time values (< -2)"


# ---------------------------------------------------------------------------
# Events table — known event IDs must exist
# ---------------------------------------------------------------------------

MAIN_EVENT_IDS = ['333', '222', '444', '555', '666', '777', '333bf', '333fm', '333oh',
                  'clock', 'minx', 'pyram', 'skewb', 'sq1', '444bf', '555bf', '333mbf']

@pytest.mark.parametrize("event_id", MAIN_EVENT_IDS)
def test_event_exists(conn, event_id):
    row = query_one(conn, "SELECT COUNT(*) FROM events WHERE id = %s", (event_id,))
    assert row[0] == 1, f"Event '{event_id}' not found in events table"


# ---------------------------------------------------------------------------
# Rankings integrity — world rank 1 must exist for main events
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("event_id", MAIN_EVENT_IDS)
def test_rank_single_world_record_exists(conn, event_id):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_single WHERE event_id = %s AND world_rank = 1", (event_id,))
    assert row[0] >= 1, f"No world rank 1 entry in ranks_single for event '{event_id}'"


@pytest.mark.parametrize("event_id", ['333', '222', '444', '555', '666', '777',
                                       '333oh', 'clock', 'minx', 'pyram', 'skewb', 'sq1'])
def test_rank_average_world_record_exists(conn, event_id):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_average WHERE event_id = %s AND world_rank = 1", (event_id,))
    assert row[0] >= 1, f"No world rank 1 entry in ranks_average for event '{event_id}'"


def test_ranks_single_no_null_person_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_single WHERE person_id IS NULL OR person_id = ''")
    assert row[0] == 0, f"{row[0]} ranks_single rows have NULL or empty person_id"


def test_ranks_average_no_null_person_id(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_average WHERE person_id IS NULL OR person_id = ''")
    assert row[0] == 0, f"{row[0]} ranks_average rows have NULL or empty person_id"


def test_ranks_single_best_positive(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_single WHERE best <= 0")
    assert row[0] == 0, f"{row[0]} ranks_single rows have non-positive best values"


def test_ranks_average_best_positive(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM ranks_average WHERE best <= 0")
    assert row[0] == 0, f"{row[0]} ranks_average rows have non-positive best values"


# ---------------------------------------------------------------------------
# Cross-table consistency
# ---------------------------------------------------------------------------

def test_results_person_ids_exist_in_persons(conn):
    """Sampled check: results should reference valid persons."""
    row = query_one(conn, """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT person_id FROM results LIMIT 10000
        ) sub
        LEFT JOIN persons p ON sub.person_id = p.wca_id
        WHERE p.wca_id IS NULL
    """)
    assert row[0] == 0, f"{row[0]} sampled result person_ids don't exist in persons"


def test_ranks_persons_cross_reference(conn):
    """Sampled check: ranks_single should reference valid persons."""
    row = query_one(conn, """
        SELECT COUNT(*) FROM (
            SELECT DISTINCT person_id FROM ranks_single LIMIT 5000
        ) sub
        LEFT JOIN persons p ON sub.person_id = p.wca_id
        WHERE p.wca_id IS NULL
    """)
    assert row[0] == 0, f"{row[0]} sampled ranks_single person_ids don't exist in persons"


def test_competitions_no_null_name(conn):
    row = query_one(conn, "SELECT COUNT(*) FROM competitions WHERE name IS NULL OR name = ''")
    assert row[0] == 0, f"{row[0]} competitions have NULL or empty name"


def test_competitions_year_range(conn):
    """WCA competitions started in 2003; future year cap is generous."""
    row = query_one(conn, "SELECT COUNT(*) FROM competitions WHERE year < 2003 OR year > 2030")
    assert row[0] == 0, f"{row[0]} competitions have years outside the expected 2003–2030 range"


# ---------------------------------------------------------------------------
# Spot-check: known world records (validate time values are reasonable)
# ---------------------------------------------------------------------------

def test_3x3_wr_single_is_sub_5_seconds(conn):
    """Current 3x3 WR single is under 5 seconds (500 centiseconds)."""
    row = query_one(conn, "SELECT best FROM ranks_single WHERE event_id = '333' AND world_rank = 1")
    assert row is not None, "No 3x3 WR single found"
    assert row[0] < 500, f"3x3 WR single {row[0]}cs seems too slow (expected < 500cs / 5.00s)"
    assert row[0] > 0, "3x3 WR single should be a positive time"


def test_3x3_wr_average_is_sub_6_seconds(conn):
    row = query_one(conn, "SELECT best FROM ranks_average WHERE event_id = '333' AND world_rank = 1")
    assert row is not None, "No 3x3 WR average found"
    assert row[0] < 600, f"3x3 WR average {row[0]}cs seems too slow (expected < 600cs / 6.00s)"
    assert row[0] > 0, "3x3 WR average should be a positive time"
