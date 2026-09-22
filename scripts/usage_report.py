"""Compare API operations using the shared database or exported application logs."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymysql

from services.api_usage import TOKEN_FIELDS, connect_usage_db

GROUP_FIELDS = ('operation', 'source', 'model', 'provider', 'day')


def utc_datetime(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def database_events(since, until):
    connection = connect_usage_db()
    try:
        with connection.cursor(pymysql.cursors.SSCursor) as cursor:
            try:
                cursor.execute(
                    'SELECT `event` FROM api_usage_events '
                    'WHERE occurred_at >= %s AND occurred_at < %s',
                    (since.replace(tzinfo=None), until.replace(tzinfo=None)))
            except pymysql.err.ProgrammingError as error:
                if error.args[0] == 1146:
                    raise RuntimeError('No usage table yet. It is created when the deployed app records its first API call.') from None
                raise
            for (payload,) in cursor:
                yield json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    finally:
        connection.close()


def log_events(paths):
    """Read plain JSON or app/Fly log prefixes; deduplicate repeated exports."""
    seen = set()
    for path in paths:
        with open(path, encoding='utf-8') as stream:
            for line in stream:
                if 'api_usage ' in line:
                    line = line.split('api_usage ', 1)[1]
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(event, dict) or event.get('type') != 'api_usage':
                    continue
                event_id = event.get('event_id')
                if event_id and event_id not in seen:
                    seen.add(event_id)
                    yield event


def summarize(events, since, until, group_by, source=None):
    groups = {}
    for event in events:
        occurred = utc_datetime(event['occurred_at'])
        if not since <= occurred < until or (source and event.get('source') != source):
            continue
        dimensions = {field: (occurred.date().isoformat() if field == 'day' else
                              event.get(field, 'unknown')) for field in group_by}
        key = tuple(dimensions.values())
        if key not in groups:
            groups[key] = {
                **dimensions, 'calls': 0, 'result_cache_hits': 0, 'errors': 0,
                'unpriced_calls': 0, 'known_estimated_usd': Decimal(0),
                'priced_calls': 0, 'duration_ms': 0,
                **dict.fromkeys(TOKEN_FIELDS, 0),
            }
        row = groups[key]
        if event['status'] == 'cache_hit':
            row['result_cache_hits'] += 1
            continue
        row['calls'] += 1
        row['errors'] += event['status'] == 'error'
        row['duration_ms'] += event.get('duration_ms', 0)
        for field in TOKEN_FIELDS:
            row[field] += event.get(field) or 0
        cost = event.get('estimated_cost_usd')
        if cost is None:
            row['unpriced_calls'] += 1
        else:
            row['known_estimated_usd'] += Decimal(str(cost))
            row['priced_calls'] += 1

    rows = sorted(groups.values(), key=lambda row: row['known_estimated_usd'], reverse=True)
    for row in rows:
        cost = row['known_estimated_usd']
        row['known_estimated_usd'] = str(cost) if row['priced_calls'] or not row['calls'] else None
        row['avg_usd_per_call'] = (str(cost / row['calls']) if row['calls'] and not row['unpriced_calls'] else None)
        row['avg_ms_per_call'] = round(row.pop('duration_ms') / row['calls']) if row['calls'] else None
        total_inputs = row['input_tokens'] + row['cache_creation_input_tokens'] + row['cache_read_input_tokens']
        row['prompt_cache_read_pct'] = round(100 * row['cache_read_input_tokens'] / total_inputs, 1) if total_inputs else None
        total = row['calls'] + row['result_cache_hits']
        row['result_cache_hit_pct'] = round(100 * row['result_cache_hits'] / total, 1) if total else None
    return rows


def print_table(rows, group_by):
    headers = [*group_by, 'Calls', 'Result hits', 'Est. USD', 'Avg/call', 'Cache read %', 'Errors', 'Unpriced']
    values = []
    for row in rows:
        cost = row['known_estimated_usd']
        average = row['avg_usd_per_call']
        values.append([*[str(row[field]) for field in group_by],
                       str(row['calls']), str(row['result_cache_hits']),
                       (f'{Decimal(cost):.6f}' + ('*' if row['unpriced_calls'] else '')) if cost is not None else 'unknown',
                       f'{Decimal(average):.6f}' if average is not None else '—',
                       str(row['prompt_cache_read_pct']) if row['prompt_cache_read_pct'] is not None else '—',
                       str(row['errors']), str(row['unpriced_calls'])])
    widths = [max(len(value) for value in column) for column in zip(headers, *values)]
    for line in [headers, ['-' * width for width in widths], *values]:
        print('  '.join(value.ljust(width) for value, width in zip(line, widths)))
    total = sum((Decimal(row['known_estimated_usd'] or '0') for row in rows), Decimal(0))
    unknown = sum(row['unpriced_calls'] for row in rows)
    print(f'\nKnown estimated total: ${total:.6f}; calls with unknown cost: {unknown}.')
    print('USD at recorded public rates, before credits/discounts. * = partial total.')
    print('Cache read % measures input tokens; result hits avoided an API call.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--days', type=int, default=30, help='Rolling days to include (default: 30)')
    parser.add_argument('--since', help='Inclusive ISO date/time in UTC; overrides --days')
    parser.add_argument('--until', help='Exclusive ISO date/time in UTC (default: now)')
    parser.add_argument('--group-by', default='operation', help='Comma-separated: ' + ', '.join(GROUP_FIELDS))
    parser.add_argument('--source', choices=['web', 'discord', 'maintenance', 'unattributed'])
    parser.add_argument('--format', choices=['table', 'csv', 'json'], default='table')
    parser.add_argument('--logs', nargs='+', metavar='FILE', help='Read exported logs instead of the database')
    args = parser.parse_args(argv)
    group_by = args.group_by.split(',')
    if not group_by or len(set(group_by)) != len(group_by) or any(field not in GROUP_FIELDS for field in group_by):
        parser.error('--group-by must contain unique fields from ' + ', '.join(GROUP_FIELDS))
    if args.days <= 0:
        parser.error('--days must be positive')
    try:
        until = utc_datetime(args.until) if args.until else datetime.now(timezone.utc)
        since = utc_datetime(args.since) if args.since else until - timedelta(days=args.days)
    except ValueError:
        parser.error('--since and --until must be ISO dates or timestamps')
    if since >= until:
        parser.error('--since must precede --until')
    try:
        events = log_events(args.logs) if args.logs else database_events(since, until)
        rows = summarize(events, since, until, group_by, args.source)
    except (OSError, ValueError, RuntimeError, pymysql.MySQLError) as error:
        # Connection errors can contain infrastructure details; keep them private.
        message = str(error) if isinstance(error, RuntimeError) else type(error).__name__
        print(f'Unable to read API usage: {message}. Check DB settings or use --logs.', file=sys.stderr)
        return 1
    if args.format == 'json':
        print(json.dumps({'since': since.isoformat(), 'until': until.isoformat(),
                          'cost_basis': 'public_rate_estimate_before_credits', 'rows': rows}, indent=2))
    elif args.format == 'csv':
        if rows:
            writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    elif rows:
        print(f'API usage: {since.isoformat()} to {until.isoformat()} (exclusive)\n')
        print_table(rows, group_by)
    else:
        print('No recorded API usage in this period. Tracking begins after deployment.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
