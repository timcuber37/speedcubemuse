"""Offline tests for building cuber_profiles from the WCA TSV export.

The build used to run as SQL inside TiDB. These pin the semantics it kept when
it moved to Python: NULL handling, the joins, retired events, the Worlds rules,
and how update_database.py decides between reloading tables and rebuilding the
matrix. A tiny synthetic export stands in for the real 386 MB one.
"""
import io
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))

import build_cuber_profiles as build  # noqa: E402
import update_database  # noqa: E402
from wca_export import ExportReader  # noqa: E402

from services import site_meta  # noqa: E402

# Column orders as the real v2 export writes them.
RESULTS = ('id', 'pos', 'best', 'average', 'competition_id', 'round_type_id',
           'event_id', 'person_name', 'person_id', 'format_id',
           'regional_single_record', 'regional_average_record', 'person_country_id')
RANKS = ('best', 'person_id', 'event_id', 'world_rank', 'continent_rank', 'country_rank')


def result(pid, comp, event, best=1000, average=1100, pos=5, round_type='1',
           single_record='NULL', average_record='NULL'):
    return ('1', str(pos), str(best), str(average), comp, round_type, event,
            'Name', pid, 'a', single_record, average_record, 'X')


def export_zip(*, results=(), ranks_single=(), ranks_average=(), persons=None,
               export_date='2026-11-11 00:30:20 UTC', extra_lines=None) -> bytes:
    tables = {
        'results': (RESULTS, results),
        'ranks_single': (RANKS, ranks_single),
        'ranks_average': (RANKS, ranks_average),
        'persons': (('name', 'gender', 'wca_id', 'sub_id', 'country_id'), persons or [
            ('Ana Lee', 'f', '2010LEEA01', '1', 'USA'),
            ('Old Name', 'f', '2010LEEA01', '2', 'Canada'),
            ('Bo Kim', ' M ', '2012KIMB01', '1', 'Korea'),
            ('Cy Diaz', 'm', '2020DIAZ01', '1', 'Atlantis'),
        ]),
        'competitions': (('id', 'name', 'country_id', 'year', 'month', 'day'), [
            ('Open2012', 'Open 2012', 'USA', '2012', '1', '1'),
            ('Open2019', 'Open 2019', 'Canada', '2019', '1', '1'),
            ('Open2026', 'Open 2026', 'XN', '2026', '1', '1'),
            ('WC2019', 'Worlds 2019', 'Australia', '2019', '7', '1'),
            ('WC2019Side', 'Not Worlds', 'Australia', '2019', '7', '1'),
        ]),
        'countries': (('id', 'iso2', 'name', 'continent_id'), [
            ('USA', 'US', 'USA', '_North America'),
            ('Korea', 'KR', 'Korea', '_Asia'),
        ]),
        'events': (('id', 'format', 'name', 'rank'), [
            ('333', 'time', '3x3', '10'), ('444', 'time', '4x4', '30'),
            ('minx', 'time', 'Megaminx', '120'), ('333ft', 'time', 'Feet', '996'),
            # Live in the export but not official: not in CURRENT_EVENTS.
            ('fto', 'time', 'Face Turning Octahedron', '810'),
        ]),
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w') as z:
        if export_date:
            z.writestr('metadata.json', json.dumps({'export_date': export_date}))
        for name, (header, rows) in tables.items():
            lines = ['\t'.join(header)] + ['\t'.join(r) for r in rows]
            lines += (extra_lines or {}).get(name, [])
            z.writestr(f'WCA_export_{name}.tsv', '﻿' + '\n'.join(lines) + '\n')
    return buf.getvalue()


def built(**tables) -> dict[str, dict]:
    profiles = build.run(export=export_zip(**tables), dry_run=True)
    return {p['wca_id']: p for p in profiles}


# ---------------------------------------------------------------------------
# Reading the export
# ---------------------------------------------------------------------------

def test_reader_maps_nulls_skips_malformed_rows_and_strips_the_bom():
    data = export_zip(results=[result('2010LEEA01', 'Open2012', '333')],
                      extra_lines={'results': ['too\tfew\tfields', '']})
    rows = list(ExportReader(data).rows('results', ('person_id', 'regional_single_record')))
    assert rows == [('2010LEEA01', None)]
    assert ExportReader(data).export_date() == '2026-11-11'


def test_reader_rejects_missing_tables_and_columns():
    reader = ExportReader(export_zip())
    with pytest.raises(KeyError):
        list(reader.rows('scrambles', ('id',)))
    with pytest.raises(KeyError):
        list(reader.rows('results', ('no_such_column',)))


def test_undated_export_uses_the_fallback_date(monkeypatch):
    published = []
    monkeypatch.setattr(build, 'get_connection', Mock())
    monkeypatch.setattr(build, 'write_profiles', Mock())
    monkeypatch.setattr(site_meta, 'publish_game_profiles_generation',
                        lambda conn, generation: published.append(generation))
    build.run(export=export_zip(export_date=None), export_date='2026-11-11')
    assert published[0].startswith('2026-11-11/')


# ---------------------------------------------------------------------------
# The attribute families
# ---------------------------------------------------------------------------

def test_records_count_both_columns_and_tolerate_nulls():
    p = built(results=[
        result('2010LEEA01', 'Open2012', '333', single_record='WR', average_record='NAR'),
        result('2010LEEA01', 'Open2019', '444', single_record='NR', average_record='\\N'),
        result('2010LEEA01', 'Open2019', 'minx', single_record='ER', average_record='WR'),
    ])['2010LEEA01']['attrs']
    assert (p['wr_single_count'], p['wr_average_count'], p['cr_count'], p['nr_count']) == (1, 1, 2, 1)
    assert p['has_wr'] and p['has_cr_or_better'] and p['has_nr_or_better']


def test_record_holders_and_current_top100_are_the_elite():
    profiles = built(
        results=[result('2010LEEA01', 'Open2012', '333', single_record='AsR'),
                 result('2012KIMB01', 'Open2012', '333'),
                 result('2020DIAZ01', 'Open2012', '333')],
        ranks_single=[('500', '2012KIMB01', '444', '100', '10', '1'),
                      # A top 100 in a retired event is not a current standing.
                      ('900', '2020DIAZ01', '333ft', '3', '1', '1')],
    )
    assert profiles['2010LEEA01']['tier'] in (1, 2, 3)
    assert profiles['2012KIMB01']['tier'] in (1, 2, 3)
    assert profiles['2020DIAZ01']['tier'] == build.TIER_LONG_TAIL


def test_career_counts_distinct_values_and_follows_the_competition_join():
    p = built(results=[
        result('2010LEEA01', 'Open2012', '333'),
        result('2010LEEA01', 'Open2012', '444'),
        result('2010LEEA01', 'Open2026', '333'),
        # No such competition: the SQL joined competitions, so this row counts
        # toward records but not toward the career numbers.
        result('2010LEEA01', 'Ghost2015', 'minx', single_record='NR'),
    ])['2010LEEA01']['attrs']
    assert p['total_results'] == 3
    assert p['comp_count'] == 2
    assert p['events_competed'] == 2
    assert (p['first_year'], p['last_year'], p['years_active']) == (2012, 2026, 15)
    assert p['countries_competed_in'] == 2   # USA and the multi-country 'XN'
    assert p['nr_count'] == 1


def test_worlds_needs_a_real_championship_id_and_a_final_podium():
    profiles = built(results=[
        result('2010LEEA01', 'WC2019', '333', pos=2, round_type='f'),
        result('2012KIMB01', 'WC2019', '333', pos=2, round_type='1'),   # not a final
        result('2012KIMB01', 'WC2019', '444', pos=1, round_type='c', best='-1'),  # DNF
        result('2020DIAZ01', 'WC2019Side', '333', pos=1, round_type='f'),
    ])
    assert profiles['2010LEEA01']['attrs']['has_worlds_podium'] is True
    assert profiles['2012KIMB01']['attrs']['competed_at_worlds'] is True
    assert profiles['2012KIMB01']['attrs']['has_worlds_podium'] is False
    assert profiles['2020DIAZ01']['attrs']['competed_at_worlds'] is False


def test_rankings_take_the_best_of_single_and_average_and_skip_retired_events():
    p = built(
        results=[result('2010LEEA01', 'Open2019', '333')],
        ranks_single=[('560', '2010LEEA01', '333', '743', '202', '178'),
                      ('3945', '2010LEEA01', 'minx', '511', '120', '1'),
                      ('900', '2010LEEA01', '333ft', '2', '1', '1')],
        ranks_average=[('651', '2010LEEA01', '333', '294', '78', '67'),
                       ('4145', '2010LEEA01', 'minx', '313', '72', '58')],
    )['2010LEEA01']['attrs']
    # The case that started this: best rank 294 in 3x3 average beats Megaminx.
    assert p['main_event'] == '333' and p['main_event_group'] is None
    assert p['top100_events'] == [] and p['top10_any'] is False
    assert p['currently_nr_or_better'] is True   # country rank 1 in Megaminx single
    assert p['sub6_333_single'] is True and p['sub10_333_average'] is True


def test_identity_uses_the_current_sub_id_and_the_country_continent():
    profiles = built(results=[result('2010LEEA01', 'Open2012', '333'),
                              result('2012KIMB01', 'Open2012', '333'),
                              result('2020DIAZ01', 'Open2012', '333')])
    assert profiles['2010LEEA01']['country_id'] == 'USA'
    assert profiles['2010LEEA01']['continent_id'] == '_North America'
    assert profiles['2012KIMB01']['attrs']['gender'] == 'm'
    assert profiles['2020DIAZ01']['continent_id'] is None   # LEFT JOIN, no country row


def test_current_events_are_official_and_not_retired():
    assert build.read_current_events(ExportReader(export_zip())) == {'333', '444', 'minx'}


def test_competed_all_events_needs_every_current_event_not_just_as_many():
    profiles = built(results=[
        result('2010LEEA01', 'Open2012', '333'),
        result('2010LEEA01', 'Open2012', '444'),
        result('2010LEEA01', 'Open2019', 'minx'),
        # Three distinct events too, but one is retired and Megaminx is missing.
        result('2012KIMB01', 'Open2012', '333'),
        result('2012KIMB01', 'Open2012', '444'),
        result('2012KIMB01', 'Open2012', '333ft'),
    ])
    assert profiles['2010LEEA01']['attrs']['competed_all_events'] is True   # no FTO needed
    assert profiles['2012KIMB01']['attrs']['events_competed'] == 3
    assert profiles['2012KIMB01']['attrs']['competed_all_events'] is False


def test_people_without_results_are_not_profiled():
    assert built(results=[result('2010LEEA01', 'Open2012', '333')]).keys() == {'2010LEEA01'}


# ---------------------------------------------------------------------------
# Refresh decisions
# ---------------------------------------------------------------------------

@pytest.fixture
def refresh(monkeypatch):
    calls = Mock()
    monkeypatch.setattr(update_database.wca_export, 'fetch_metadata',
                        lambda: {'export_date': '2026-11-11T00:30:20Z', 'tsv_url': 'u'})
    monkeypatch.setattr(update_database, 'download', calls.download)
    monkeypatch.setattr(update_database, 'reload_tables', calls.reload_tables)
    monkeypatch.setattr(update_database.build_cuber_profiles, 'run', calls.build)
    state = {'tables': '2026-11-11', 'profiles': '2026-11-11'}
    monkeypatch.setattr(update_database, 'last_loaded_export_date', lambda: state['tables'])
    monkeypatch.setattr(update_database, 'last_profiles_export_date', lambda: state['profiles'])
    return calls, state


def test_nothing_to_do_downloads_nothing(refresh):
    calls, _ = refresh
    update_database.run()
    calls.download.assert_not_called()
    calls.build.assert_not_called()


def test_stale_profiles_are_rebuilt_without_reloading_tables(refresh):
    """The Nov 1 state: tables current, matrix left behind by a failed rebuild."""
    calls, state = refresh
    state['profiles'] = None
    update_database.run()
    calls.reload_tables.assert_not_called()
    calls.build.assert_called_once()
    assert calls.build.call_args.kwargs['export_date'] == '2026-11-11'


def test_a_new_export_reloads_and_rebuilds(refresh):
    calls, state = refresh
    state.update(tables='2026-10-31', profiles='2026-10-31')
    update_database.run()
    calls.reload_tables.assert_called_once()
    calls.build.assert_called_once()


def test_a_failed_rebuild_fails_the_run(refresh):
    calls, state = refresh
    state['profiles'] = '2026-10-01'
    calls.build.side_effect = RuntimeError('memory limit')
    with pytest.raises(RuntimeError):
        update_database.run()


def test_skip_profiles_ignores_a_stale_matrix(refresh):
    calls, state = refresh
    state['profiles'] = None
    update_database.run(build_profiles=False)
    calls.download.assert_not_called()


# ---------------------------------------------------------------------------
# The generation marker
# ---------------------------------------------------------------------------

def _conn_returning(value):
    cur = Mock()
    cur.fetchone.return_value = None if value is None else (value,)
    conn = Mock()
    conn.cursor.return_value = cur
    return conn


def test_marker_carries_the_export_date_and_stays_unique():
    a, b = (site_meta.game_profiles_generation('2026-11-11') for _ in range(2))
    assert a != b
    assert site_meta.read_game_profiles_export_date(_conn_returning(a)) == '2026-11-11'


@pytest.mark.parametrize('value', [None, '819d7e6383504850a4ad9aeaaef331fc'])
def test_old_or_missing_markers_read_as_unknown(value):
    assert site_meta.read_game_profiles_export_date(_conn_returning(value)) is None


# ---------------------------------------------------------------------------
# Writing: staged, then swapped in atomically
# ---------------------------------------------------------------------------

class RecordingCursor:
    def __init__(self, staged_rows=None):
        self.statements = []
        self.staged_rows = staged_rows

    def execute(self, sql, args=None):
        self.statements.append(' '.join(sql.split()))

    def executemany(self, sql, rows):
        self.statements.append(' '.join(sql.split()))
        self.inserted = getattr(self, 'inserted', 0) + len(rows)

    def fetchone(self):
        return (self.inserted if self.staged_rows is None else self.staged_rows,)

    def close(self):
        pass


def _write(profile_count, staged_rows=None):
    cur = RecordingCursor(staged_rows)
    conn = Mock()
    conn.cursor.return_value = cur
    profiles = [{'wca_id': f'2020TEST{i:02d}', 'name': 'T', 'country_id': 'USA',
                 'continent_id': '_North America', 'fame': 1, 'tier': 4, 'attrs': {}}
                for i in range(profile_count)]
    build.write_profiles(conn, profiles)
    return cur.statements


def test_rows_go_to_staging_and_one_rename_swaps_them_in():
    statements = _write(1200)
    inserts = [s for s in statements if s.startswith('INSERT')]
    assert len(inserts) == 3   # batches of 500
    assert all(s.startswith('INSERT INTO `cuber_profiles_next`') for s in inserts)
    assert 'RENAME TABLE `cuber_profiles` TO `cuber_profiles_old`, '\
        '`cuber_profiles_next` TO `cuber_profiles`' in statements
    # Nothing ever empties or writes the live table directly.
    assert not any('TRUNCATE' in s or s.startswith('INSERT INTO `cuber_profiles` ')
                   or s.startswith('DELETE') for s in statements)
    rename = next(i for i, s in enumerate(statements) if s.startswith('RENAME'))
    assert max(i for i, s in enumerate(statements) if s.startswith('INSERT')) < rename
    assert statements[rename + 1] == 'DROP TABLE IF EXISTS `cuber_profiles_old`'
    assert 'DROP TABLE IF EXISTS `_gc_pool`' in statements[rename:]


def test_leftovers_from_a_failed_run_are_cleared_first():
    statements = _write(3)
    assert statements[:2] == ['DROP TABLE IF EXISTS `cuber_profiles_next`',
                              'DROP TABLE IF EXISTS `cuber_profiles_old`']


def test_a_short_staging_table_is_never_swapped_in():
    cur = RecordingCursor(staged_rows=2)
    conn = Mock()
    conn.cursor.return_value = cur
    profiles = [{'wca_id': f'x{i}', 'name': 'T', 'country_id': None, 'continent_id': None,
                 'fame': 1, 'tier': 5, 'attrs': {}} for i in range(3)]
    with pytest.raises(RuntimeError, match='expected 3'):
        build.write_profiles(conn, profiles)
    assert not any(s.startswith('RENAME') for s in cur.statements)
