"""Offline relational tests: execute production comparison SQL over small fixtures."""
import random
import sqlite3
from pathlib import Path

from flask import Flask
import pymysql
import pytest

from services import nemesis
from services.nemesis import NemesisService, compare_records


class Cursor:
    """Use SQLite to exercise the actual parameterized SQL without a live export."""
    def __init__(self, conn):
        self.cur = conn.cursor()

    def execute(self, sql, args=()):
        self.cur.execute(sql.replace('%s', '?'), args)

    def fetchone(self):
        row = self.cur.fetchone()
        return dict(row) if row else None

    def fetchall(self):
        return [dict(row) for row in self.cur.fetchall()]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cur.close()


class Connection:
    def __init__(self, conn):
        self.conn = conn

    def cursor(self):
        return Cursor(self.conn)

    def close(self):
        pass  # each request borrows the same fixture, which pytest owns


@pytest.fixture
def database(monkeypatch):
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE persons (wca_id TEXT, sub_id INTEGER, name TEXT, country_id TEXT);
        CREATE TABLE events (id TEXT, name TEXT, rank INTEGER);
        CREATE TABLE ranks_single (person_id TEXT, event_id TEXT, best INTEGER);
        CREATE TABLE ranks_average (person_id TEXT, event_id TEXT, best INTEGER);
        INSERT INTO events VALUES ('333', '3x3x3 Cube', 10), ('222', '2x2x2 Cube', 20),
            ('333fm', 'Fewest Moves', 30), ('333mbf', 'Multi-Blind', 40),
            ('fto', 'Upcoming FTO', 810), ('333ft', 'With Feet', 900);
    ''')
    conn.executemany('INSERT INTO persons VALUES (?, 1, ?, ?)', [
        (f'2020TEST{i:02d}', f'Competitor {i}', 'USA') for i in range(9)
    ])
    conn.execute("INSERT INTO persons VALUES ('2020TEST00', 2, 'Old identity', 'Canada')")
    # Target, better everywhere, tie on average, only 3x3, worse everywhere,
    # better singles/worse averages, extra event not held by target, no valid
    # results, only a discontinued event.
    records = [
        (0, '333', 1000, 1200), (0, '222', 400, 600),
        (1, '333', 900, 1100), (1, '222', 300, 500),
        (2, '333', 900, 1200), (2, '222', 300, 500),
        (3, '333', 1100, 1300),
        (4, '333', 1100, 1300), (4, '222', 500, 700),
        (5, '333', 900, 1300), (5, '222', 300, 700),
        (6, '333', 1100, 1300), (6, '222', 500, 700), (6, '333fm', 30, 3000),
        (7, '333', -1, 0), (8, '333ft', 500, 600),
        # Target's discontinued PB cannot block a nemesis.
        (0, '333ft', 1, 1), (0, 'fto', 1, 1),
    ]
    for person, event, single, average in records:
        conn.execute('INSERT INTO ranks_single VALUES (?, ?, ?)', (f'2020TEST{person:02d}', event, single))
        conn.execute('INSERT INTO ranks_average VALUES (?, ?, ?)', (f'2020TEST{person:02d}', event, average))
    monkeypatch.setattr(nemesis, '_connect', lambda: Connection(conn))
    yield conn
    conn.close()


def ids(payload):
    return [person['wca_id'] for person in payload['competitors']]


def test_overview_counts_and_lists_agree_with_asymmetric_event_histories(database):
    service = NemesisService()
    incoming = service.overview('2020test00')
    assert incoming['counts'] == {'nemeses': 1, 'nemesized': 2}
    assert ids(incoming) == ['2020TEST01']
    outgoing = service.overview('2020TEST00', direction='nemesized')
    assert ids(outgoing) == ['2020TEST03', '2020TEST04']
    assert incoming['competitor']['name'] == 'Competitor 0'
    # The stronger competitor lacks the target's retired result and still wins.
    assert service.compare('2020TEST01', '2020TEST00')['left_nemesizes_right']


def test_modes_require_only_selected_results(database):
    service = NemesisService()
    singles = service.overview('2020TEST00', mode='single')
    assert ids(singles) == ['2020TEST01', '2020TEST02', '2020TEST05']
    assert singles['counts'] == {'nemeses': 3, 'nemesized': 2}
    averages = service.overview('2020TEST00', mode='average')
    assert averages['counts'] == {'nemeses': 1, 'nemesized': 3}


@pytest.mark.parametrize('person', ['2020TEST07', '2020TEST08'])
def test_no_valid_current_results_cannot_create_vacuous_relationships(database, person):
    payload = NemesisService().overview(person)
    assert payload['counts'] == {'nemeses': 0, 'nemesized': 0}
    assert not payload['has_results']
    assert not NemesisService().compare('2020TEST00', person)['left_nemesizes_right']


def test_counts_survive_pagination_and_an_empty_page(database, monkeypatch):
    monkeypatch.setattr(nemesis, 'PAGE_SIZE', 1)
    service = NemesisService()
    pages = [service.overview('2020TEST00', direction='nemesized', page=i) for i in (1, 2, 3)]
    assert [ids(page) for page in pages] == [['2020TEST03'], ['2020TEST04'], []]
    assert all(page['counts'] == {'nemeses': 1, 'nemesized': 2} for page in pages)


def test_sql_matches_independent_oracle_across_random_profiles(database):
    rng = random.Random(37)
    profiles = {}
    for i in range(40):
        person = f'2021RAND{i:02d}'
        profiles[person] = {}
        database.execute('INSERT INTO persons VALUES (?, 1, ?, ?)', (person, person, 'Japan'))
        for event in ('333', '222', '333fm', '333mbf'):
            for kind in ('single', 'average'):
                if event == '333mbf' and kind == 'average':
                    continue
                if rng.random() < .65:
                    value = rng.choice([100, 200, 300, 400])
                    profiles[person][event, kind] = value
                    database.execute(f'INSERT INTO ranks_{kind} VALUES (?, ?, ?)', (person, event, value))
    # Include the deterministic fixtures too, so counts cover the entire pool.
    for kind in ('single', 'average'):
        for row in database.execute(f'SELECT * FROM ranks_{kind}'):
            if row['event_id'] in ('333ft', 'fto') or row['best'] <= 0:
                continue
            profiles.setdefault(row['person_id'], {})[row['event_id'], kind] = row['best']
    for mode in nemesis.MODES:
        selected = {person: {key: value for key, value in values.items()
                            if mode == 'both' or key[1] == mode}
                    for person, values in profiles.items()}
        for target in list(profiles)[:12]:
            actual = NemesisService().overview(target, mode=mode)
            a = selected[target]
            incoming, outgoing = [], []
            for person, b in selected.items():
                if person == target:
                    continue
                # Independent definition; do not use the production dominates().
                if a and all(b.get(key, float('inf')) < value for key, value in a.items()):
                    incoming.append(person)
                if b and all(a.get(key, float('inf')) < value for key, value in b.items()):
                    outgoing.append(person)
            assert actual['counts'] == {'nemeses': len(incoming), 'nemesized': len(outgoing)}
            assert ids(actual) == sorted(incoming)[:nemesis.PAGE_SIZE]
            for opponent in list(profiles)[:4]:
                comparison = NemesisService().compare(target, opponent, mode)
                assert comparison['right_nemesizes_left'] == (opponent in incoming)
                assert comparison['left_nemesizes_right'] == (opponent in outgoing)


def test_search_current_identity_unicode_and_literal_wildcards(database):
    database.execute("UPDATE persons SET name = 'Zoë (张)' WHERE wca_id = '2020TEST00' AND sub_id = 1")
    service = NemesisService()
    assert service.search('张)')[0]['name'] == 'Zoë (张)'
    assert service.search('2020test00')[0]['wca_id'] == '2020TEST00'
    assert service.search('%_') == []
    assert service.search("' OR 1=1 --") == []
    assert service.search('Old identity') == []
    assert service.search('x') == []


def test_comparison_formats_fmc_multiblind_and_long_times():
    events = [{'id': e, 'name': e} for e in ('333fm', '333mbf', '333', '222')]
    a = {('333fm', 'single'): 20, ('333fm', 'average'): 2250,
         ('333mbf', 'single'): 580360004, ('333', 'single'): 6100}
    b = {('333fm', 'single'): 21, ('333fm', 'average'): 2250,
         ('333mbf', 'single'): 590360004, ('222', 'single'): 300}
    result = compare_records(a, b, events, 'both')
    assert result['events'][0]['single']['left'] == '20 moves'
    assert result['events'][0]['single']['percentage_difference'] == pytest.approx(100 / 21)
    assert result['events'][0]['average']['percentage_difference'] == 0
    assert result['events'][0]['average']['left'] == '22.50 moves'
    assert result['events'][1]['single']['left'] == '45/49 1:00:00'
    assert result['events'][1]['single']['percentage_difference'] is None
    assert result['events'][2]['single']['left'] == '1:01.00'
    assert result['events'][2]['single']['winner'] is None
    assert result['scores'] == {'left': 2, 'right': 0, 'ties': 1, 'unmatched': 2}
    assert not result['left_nemesizes_right']
    assert not result['right_nemesizes_left']


@pytest.mark.parametrize('left,right,expected', [
    (1000, 2000, 50), (3000, 2000, -50), (1000, 1000, 0),
    (None, 1000, None), (1000, None, None), (None, None, None),
])
def test_percentages_are_relative_to_opponent_and_missing_results_are_unranked(left, right, expected):
    key = ('333', 'single')
    result = compare_records({key: left} if left else {}, {key: right} if right else {},
                             [{'id': '333', 'name': '3x3x3 Cube'}], 'both')
    percentage = result['events'][0]['single']['percentage_difference']
    assert percentage == expected


def test_self_comparison_is_all_ties(database):
    result = NemesisService().compare('2020TEST00', '2020TEST00')
    assert result['scores'] == {'left': 0, 'right': 0, 'ties': 4, 'unmatched': 0}
    assert not result['left_nemesizes_right']
    assert not result['right_nemesizes_left']


def test_multiblind_averages_are_excluded_from_comparisons_and_counts(database):
    before = NemesisService().compare('2020TEST00', '2020TEST01')
    counts = NemesisService().overview('2020TEST00')['counts']
    # Even a malformed import cannot introduce a nonexistent result type.
    database.executemany('INSERT INTO ranks_average VALUES (?, ?, ?)', [
        ('2020TEST00', '333mbf', 1), ('2020TEST01', '333mbf', 2),
        ('2020TEST03', '333mbf', 3),
    ])
    after = NemesisService().compare('2020TEST00', '2020TEST01')
    multiblind = next(event for event in after['events'] if event['event_id'] == '333mbf')
    assert 'single' in multiblind
    assert 'average' not in multiblind
    assert after == before
    assert NemesisService().overview('2020TEST00')['counts'] == counts


def test_single_only_competitor_has_no_average_relationships(database):
    database.execute("DELETE FROM ranks_average WHERE person_id = '2020TEST00'")
    payload = NemesisService().overview('2020TEST00', mode='average')
    assert not payload['has_results']
    assert payload['counts'] == {'nemeses': 0, 'nemesized': 0}
    assert NemesisService().overview('2020TEST00', mode='single')['has_results']


def test_overview_cache_expires_and_stays_bounded(database, monkeypatch):
    clock = [0]
    monkeypatch.setattr(nemesis.time, 'monotonic', lambda: clock[0])
    service = NemesisService()
    first = service.overview('2020TEST00')
    database.execute("DELETE FROM ranks_average WHERE person_id = '2020TEST01'")
    assert service.overview('2020TEST00') == first
    clock[0] = 301
    assert service.overview('2020TEST00')['counts']['nemeses'] == 0
    for page in range(1, 131):
        service.overview('2020TEST00', page=page)
    assert len(service._cache) == 128


@pytest.fixture
def client(database, monkeypatch):
    from blueprints import nemesis as bp
    from services.site_meta import _build, FALLBACK
    monkeypatch.setattr(bp, 'service', NemesisService())
    monkeypatch.setattr(bp, 'get_site_meta', lambda: _build(FALLBACK))
    app = Flask(__name__, template_folder=str(Path(__file__).resolve().parents[1] / 'templates'))
    app.config['TESTING'] = True
    app.register_blueprint(bp.nemesis_bp)
    return app.test_client()


def test_page_and_api_end_to_end(client):
    page = client.get('/nemesis?id=2020TEST00')
    assert page.status_code == 200
    assert 'nemesis.js' in page.text and 'nemesis.css' in page.text
    assert 'both singles and averages' in page.text
    assert client.get('/api/nemesis/search?q=2020TEST00').json['competitors'][0]['wca_id'] == '2020TEST00'
    assert client.get('/api/nemesis/2020TEST00').json['counts'] == {'nemeses': 1, 'nemesized': 2}
    comparison = client.get('/api/nemesis/compare?left=2020TEST00&right=2020TEST01')
    assert comparison.status_code == 200
    assert comparison.json['right_nemesizes_left']


@pytest.mark.parametrize('path', [
    '/api/nemesis/not-an-id',
    '/api/nemesis/2020TEST00?direction=invalid', '/api/nemesis/2020TEST00?page=0',
    '/api/nemesis/2020TEST00?page=-1', '/api/nemesis/2020TEST00?page=1.5',
    '/api/nemesis/2020TEST00?page=100001', '/api/nemesis/search?q=' + 'x' * 101,
    '/api/nemesis/compare?left=2020TEST00',
])
def test_invalid_inputs_are_json_errors(client, path):
    response = client.get(path)
    assert response.status_code == 400
    assert response.json['error']


def test_missing_competitor_is_404(client):
    assert client.get('/api/nemesis/2020NONE01').status_code == 404


@pytest.mark.parametrize('old_mode', ['single', 'average'])
def test_public_comparisons_include_both_results_even_with_old_modes(client, old_mode):
    overview = client.get(f'/api/nemesis/2020TEST00?mode={old_mode}')
    assert overview.json['mode'] == 'both'
    assert overview.json['counts'] == {'nemeses': 1, 'nemesized': 2}
    comparison = client.get(f'/api/nemesis/compare?left=2020TEST00&right=2020TEST02&mode={old_mode}')
    assert comparison.json['mode'] == 'both'
    assert not comparison.json['right_nemesizes_left']  # Tied average blocks the faster singles.


def test_database_outage_is_not_a_false_zero_or_cached(client, monkeypatch):
    connect = nemesis._connect
    def fail():
        raise pymysql.OperationalError(2003, 'private connection detail')
    monkeypatch.setattr(nemesis, '_connect', fail)
    response = client.get('/api/nemesis/2020TEST00')
    assert response.status_code == 503
    assert 'private connection detail' not in response.text
    assert 'counts' not in response.json
    monkeypatch.setattr(nemesis, '_connect', connect)
    assert client.get('/api/nemesis/2020TEST00').json['counts']['nemeses'] == 1
