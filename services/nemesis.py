"""Personal-best comparisons over the current official WCA ranking tables.

A dominates B when A has a strictly better result for every ranked result B
has in the selected mode. B must have at least one result. Missing results and
ties cannot satisfy a comparison; retired events and DNF/DNS are excluded.
No model calls, new tables, or resident copy of the ranking pool are needed.
"""
from collections import OrderedDict
from contextlib import closing
import re
import ssl
import threading
import time

import certifi
import pymysql

from config import (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSL,
                    QUERY_TIMEOUT_SECONDS)
from services.wca_api import WCAService

MODES = ('both', 'single', 'average')
DIRECTIONS = ('nemeses', 'nemesized')
PAGE_SIZE = 25
_formatter = WCAService()


def normalize_id(value):
    value = value.strip().upper()
    if not re.fullmatch(r'[0-9]{4}[A-Z]{4}[0-9]{2}', value):
        raise ValueError('Enter a WCA ID such as 2009ZEMD01, or search by name.')
    return value


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError('Choose both results, singles, or averages.')
    return mode


def _connect():
    kwargs = {}
    if DB_SSL:
        kwargs['ssl'] = ssl.create_default_context(cafile=certifi.where())
    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD,
        database=DB_NAME, charset='utf8mb4', autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5, read_timeout=QUERY_TIMEOUT_SECONDS + 5,
        **kwargs,
    )
    try:
        with conn.cursor() as cur:
            cur.execute('SET SESSION max_execution_time = %s',
                        (QUERY_TIMEOUT_SECONDS * 1000,))
    except Exception:
        conn.close()
        raise
    return conn


def _rank_sql(mode):
    """Only table names chosen here enter SQL; visitor input is bound separately."""
    validate_mode(mode)
    kinds = ('single', 'average') if mode == 'both' else (mode,)
    return ' UNION ALL '.join(
        f"SELECT r.person_id, r.event_id, '{kind}' AS kind, r.best "
        f"FROM ranks_{kind} r JOIN events e ON e.id = r.event_id "
        "WHERE e.`rank` < 800 AND r.best > 0"
        + (" AND r.event_id <> '333mbf'" if kind == 'average' else '')
        for kind in kinds
    )


def relation_query(mode, records, event_ids):
    """Count both directions and return only one page of competitor IDs.

    Grouping happens in the database. Comparisons match event and result kind;
    a candidate's extra results do not prevent them dominating the
    target, but DO matter when checking whether the target dominates them.
    Count rows are separate from page rows so an empty or out-of-range page
    still returns the exact totals. This SQL also runs against SQLite fixtures.
    """
    validate_mode(mode)
    # Aggregate each table at storage before combining singles and averages.
    # A join over every ranking row makes TiDB transfer the full ranking pool
    # to its SQL node. CASE comparisons let it push partial sums to storage.
    # Only the shape is generated here; event IDs and PB values stay bound.
    kinds = ('single', 'average') if mode == 'both' else (mode,)
    parts, args = [], []
    for kind in kinds:
        comparisons = [(event, best) for (event, result_kind), best in records.items()
                       if result_kind == kind]
        expressions = []
        for operator in ('<', '>'):
            branches = []
            for event, best in comparisons:
                branches.append(f'WHEN %s THEN CASE WHEN best {operator} %s THEN 1 ELSE 0 END')
                args.extend((event, best))
            expressions.append('CASE event_id ' + ' '.join(branches) + ' ELSE 0 END'
                               if branches else '0')
        placeholders = ', '.join('%s' for _ in event_ids)
        average_filter = " AND event_id <> '333mbf'" if kind == 'average' else ''
        parts.append(f"""
            SELECT person_id, COUNT(*) AS result_count,
                   SUM({expressions[0]}) AS better, SUM({expressions[1]}) AS worse
              FROM ranks_{kind} WHERE best > 0 AND event_id IN ({placeholders}){average_filter}
             GROUP BY person_id
        """)
        args.extend(event_ids)
    return f"""
        WITH per_kind AS ({' UNION ALL '.join(parts)}),
        comparisons AS (
            SELECT person_id, SUM(result_count) AS result_count,
                   SUM(better) AS better, SUM(worse) AS worse
              FROM per_kind WHERE person_id <> %s GROUP BY person_id
        ),
        relations AS (
            SELECT person_id, 'nemeses' AS direction FROM comparisons
             WHERE better = %s
            UNION ALL
            SELECT person_id, 'nemesized' AS direction FROM comparisons
             WHERE worse = result_count
        ),
        paged AS (
            SELECT person_id, direction FROM relations WHERE direction = %s
             ORDER BY person_id LIMIT %s OFFSET %s
        )
        SELECT 'count' AS row_type, direction, COUNT(*) AS total,
               NULL AS wca_id, NULL AS name, NULL AS country_id
          FROM relations GROUP BY direction
        UNION ALL
        SELECT 'person' AS row_type, n.direction, NULL AS total,
               p.wca_id, p.name, p.country_id
          FROM paged n JOIN persons p ON p.wca_id = n.person_id AND p.sub_id = 1
         ORDER BY row_type, wca_id
    """, args


def _person(cur, wca_id):
    cur.execute('SELECT wca_id, name, country_id FROM persons '
                'WHERE wca_id = %s AND sub_id = 1', (wca_id,))
    person = cur.fetchone()
    if not person:
        raise LookupError('No competitor was found for that WCA ID.')
    return dict(person)


def _events(cur):
    # 800-series ranks are upcoming events (e.g. FTO before January 2027);
    # 900-series ranks are discontinued events. Neither is currently official.
    cur.execute('SELECT id, name FROM events WHERE `rank` < 800 ORDER BY `rank`, id')
    return [dict(row) for row in cur.fetchall()]


def _records(cur, wca_id, mode='both'):
    cur.execute(f'SELECT event_id, kind, best FROM ({_rank_sql(mode)}) r '
                'WHERE person_id = %s', (wca_id,))
    return {(row['event_id'], row['kind']): int(row['best']) for row in cur.fetchall()}


def dominates(left, right):
    return bool(right) and all(key in left and left[key] < value
                               for key, value in right.items())


def _result(value, event_id, kind):
    if value is None:
        return None
    # FMC singles are moves; averages/means are moves times 100.
    if event_id == '333fm':
        return f'{value} moves' if kind == 'single' else f'{value / 100:.2f} moves'
    return _formatter._format_result_value(value, event_id)


def _percentage_difference(left, right, event_id):
    # Encoded Multi-Blind values combine points and time, so their numeric
    # ratio is not a meaningful percentage. Missing PBs are unranked too.
    if left is None or right is None or event_id == '333mbf':
        return None
    return 100 * (right - left) / right


def compare_records(left, right, events, mode):
    validate_mode(mode)
    kinds = ('single', 'average') if mode == 'both' else (mode,)
    scores = {'left': 0, 'right': 0, 'ties': 0, 'unmatched': 0}
    rows = []
    for event in events:
        row = {'event_id': event['id'], 'event_name': event['name']}
        for kind in kinds:
            if event['id'] == '333mbf' and kind == 'average':
                continue
            key = (event['id'], kind)
            a, b = left.get(key), right.get(key)
            if a is None or b is None:
                winner = None
                if a is not None or b is not None:
                    scores['unmatched'] += 1
            else:
                winner = 'tie' if a == b else ('left' if a < b else 'right')
                scores['ties' if winner == 'tie' else winner] += 1
            row[kind] = {'left': _result(a, event['id'], kind),
                         'right': _result(b, event['id'], kind), 'winner': winner,
                         'percentage_difference': _percentage_difference(a, b, event['id'])}
        rows.append(row)
    selected_left = {k: v for k, v in left.items() if k[1] in kinds and k != ('333mbf', 'average')}
    selected_right = {k: v for k, v in right.items() if k[1] in kinds and k != ('333mbf', 'average')}
    return {'events': rows, 'scores': scores,
            'left_nemesizes_right': dominates(selected_left, selected_right),
            'right_nemesizes_left': dominates(selected_right, selected_left)}


class NemesisService:
    def __init__(self):
        # A short, bounded cache prevents paging/repeated links from repeatedly
        # scanning ranks. Each entry holds at most PAGE_SIZE people, not the pool.
        self._cache = OrderedDict()
        self._cache_lock = threading.Lock()

    def search(self, query):
        query = query.strip()
        if len(query) < 2:
            return []
        if len(query) > 100:
            raise ValueError('Search with 100 characters or fewer.')
        escaped = query.replace('=', '==').replace('%', '=%').replace('_', '=_')
        with closing(_connect()) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT wca_id, name, country_id FROM persons WHERE sub_id = 1 "
                "AND (wca_id LIKE %s ESCAPE '=' OR name LIKE %s ESCAPE '=') "
                "ORDER BY CASE WHEN wca_id = %s THEN 0 ELSE 1 END, name, wca_id LIMIT 10",
                (escaped.upper() + '%', '%' + escaped + '%', query.upper()),
            )
            return [dict(row) for row in cur.fetchall()]

    def overview(self, wca_id, mode='both', direction='nemeses', page=1):
        wca_id = normalize_id(wca_id)
        validate_mode(mode)
        if direction not in DIRECTIONS:
            raise ValueError('Choose nemeses or competitors nemesized.')
        if not 1 <= page <= 100000:
            raise ValueError('Page must be between 1 and 100000.')
        key = (wca_id, mode, direction, page)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached and cached[0] > time.monotonic():
                self._cache.move_to_end(key)
                return cached[1]
        with closing(_connect()) as conn, conn.cursor() as cur:
            person = _person(cur, wca_id)
            records = _records(cur, wca_id, mode)
            counts = dict.fromkeys(DIRECTIONS, 0)
            people = []
            if records:
                sql, target_args = relation_query(mode, records,
                                                  [event['id'] for event in _events(cur)])
                cur.execute(sql,
                            (*target_args, wca_id, len(records), direction,
                             PAGE_SIZE, (page - 1) * PAGE_SIZE))
                for row in cur.fetchall():
                    if row['row_type'] == 'count':
                        counts[row['direction']] = int(row['total'])
                    else:
                        people.append({k: row[k] for k in ('wca_id', 'name', 'country_id')})
        payload = {'competitor': person, 'mode': mode, 'direction': direction,
                   'counts': counts, 'competitors': people, 'page': page,
                   'page_size': PAGE_SIZE, 'has_results': bool(records),
                   'result_count': len(records)}
        with self._cache_lock:
            self._cache[key] = (time.monotonic() + 300, payload)
            self._cache.move_to_end(key)
            while len(self._cache) > 128:
                self._cache.popitem(last=False)
        return payload

    def compare(self, left_id, right_id, mode='both'):
        left_id, right_id = normalize_id(left_id), normalize_id(right_id)
        validate_mode(mode)
        with closing(_connect()) as conn, conn.cursor() as cur:
            left, right = _person(cur, left_id), _person(cur, right_id)
            events = _events(cur)
            data = compare_records(_records(cur, left_id), _records(cur, right_id), events, mode)
        return {'left': left, 'right': right, 'mode': mode, **data}
