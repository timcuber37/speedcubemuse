"""Offline checks for recovering competitor loads after database outages."""
from unittest.mock import Mock

import pymysql
import pytest

from services.game import profiles

_ORIGINAL_READ_GENERATION = profiles._read_generation


def row(wca_id='test', tier=1):
    return {'wca_id': wca_id, 'name': 'Test Cuber', 'tier': tier,
            'fame': 1, 'attrs': {'has_wr': True}}


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch):
    monkeypatch.setattr(profiles, '_cache_rows', {})
    monkeypatch.setattr(profiles, '_cache_expires_at', {})
    monkeypatch.setattr(profiles, '_cache_loaded_at', {})
    monkeypatch.setattr(profiles, '_cache_generation', None)
    monkeypatch.setattr(profiles, '_cache_engines', {})
    monkeypatch.setattr(profiles, '_tail_cache', {})
    monkeypatch.setattr(profiles, '_all_ids', [])
    monkeypatch.setattr(profiles, '_name_index', None)
    monkeypatch.setattr(profiles, '_ids_generation', None)
    monkeypatch.setattr(profiles, '_ids_expires_at', 0.0)
    monkeypatch.setattr(profiles, '_ids_loaded_at', 0.0)
    monkeypatch.setattr(profiles, '_search_cache', {})
    monkeypatch.setattr(profiles, '_read_generation', lambda: None)
    clock = [1000.0]
    monkeypatch.setattr(profiles.time, 'monotonic', lambda: clock[0])
    return clock


def test_quota_failure_retries_promptly_after_database_is_restored(monkeypatch, isolated_cache):
    rows = [row()]
    read = Mock(side_effect=[pymysql.err.OperationalError(1105, 'usage quota exhausted'), rows])
    monkeypatch.setattr(profiles, '_read_all', read)

    assert profiles.get_engine('everyone') is None
    assert profiles._cache_rows == {}
    isolated_cache[0] += profiles._CACHE_RETRY_SECONDS - 1
    assert profiles.get_engine('everyone') is None
    assert read.call_count == 1  # Back off instead of hammering a blocked DB.

    isolated_cache[0] += 1
    engine = profiles.get_engine('everyone')
    assert engine is not None
    assert engine.rows == rows
    assert read.call_count == 2
    assert profiles._cache_expires_at[4] == isolated_cache[0] + profiles._CACHE_TTL_SECONDS


@pytest.mark.parametrize('failure', [[], pymysql.err.OperationalError(1105, 'quota')])
def test_failed_refresh_preserves_snapshot_and_engine(monkeypatch, isolated_cache, failure):
    original, updated = [row('original')], [row('updated')]
    read = Mock(side_effect=[original, failure, updated])
    monkeypatch.setattr(profiles, '_read_all', read)
    original_engine = profiles.get_engine('easy')
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS

    assert profiles.get_engine('easy') is original_engine
    assert profiles._cache_rows[1] is original
    assert read.call_count == 2
    isolated_cache[0] += profiles._CACHE_RETRY_SECONDS
    new_engine = profiles.get_engine('easy')
    assert new_engine is not original_engine
    assert new_engine.rows == updated


def test_empty_cold_load_is_retried_instead_of_cached_for_an_hour(monkeypatch, isolated_cache):
    read = Mock(side_effect=[[], [row()]])
    monkeypatch.setattr(profiles, '_read_all', read)
    assert profiles.load_profiles(4) == []
    isolated_cache[0] += profiles._CACHE_RETRY_SECONDS
    assert profiles.load_profiles(4)
    assert read.call_count == 2


def test_failed_large_pool_does_not_hide_cached_elite_names(monkeypatch):
    rows = [row()]
    monkeypatch.setattr(profiles, '_read_all', Mock(side_effect=[rows, OSError('unavailable')]))
    assert profiles.load_profiles(3) is rows
    assert profiles.load_profiles(4) == []
    assert profiles.search_by_name('Test') == rows


def test_successful_snapshot_keeps_normal_cache_lifetime(monkeypatch, isolated_cache):
    rows = [row()]
    read = Mock(return_value=rows)
    monkeypatch.setattr(profiles, '_read_all', read)
    assert profiles.load_profiles(4) is rows
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS - 1
    assert profiles.load_profiles(4) is rows
    read.assert_called_once_with(4)


def test_explicit_refresh_can_bypass_failure_cooldown(monkeypatch):
    read = Mock(side_effect=[OSError('down'), [row()]])
    monkeypatch.setattr(profiles, '_read_all', read)
    assert profiles.load_profiles(4) == []
    assert profiles.load_profiles(4, force_refresh=True)
    assert read.call_count == 2


def test_live_turn_recovers_from_503_without_a_process_restart(monkeypatch, isolated_cache):
    from app import app
    rows = [row('one'), row('two')]
    rows[1]['attrs']['has_wr'] = False
    read = Mock(side_effect=[OSError('quota exceeded'), rows])
    monkeypatch.setattr(profiles, '_read_all', read)
    with app.test_client() as client:
        failed = client.post('/api/game/akinator/turn', json={'difficulty': 'everyone'})
        assert failed.status_code == 503
        isolated_cache[0] += profiles._CACHE_RETRY_SECONDS
        recovered = client.post('/api/game/akinator/turn', json={'difficulty': 'everyone'})
    assert recovered.status_code == 200
    assert recovered.json['done'] is False
    assert recovered.json['remaining'] == 2


def test_unchanged_generation_skips_hourly_bulk_reads(monkeypatch, isolated_cache):
    read = Mock(return_value=[row()])
    monkeypatch.setattr(profiles, '_read_all', read)
    monkeypatch.setattr(profiles, '_read_generation', lambda: 'weekly-build-a')

    engine = profiles.get_engine('everyone')
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS + 1
    assert profiles.get_engine('everyone') is engine
    read.assert_called_once_with(4)


def test_metadata_outage_keeps_loaded_engine_without_bulk_retry(monkeypatch, isolated_cache):
    generation = Mock(side_effect=['build-a', OSError('temporarily unavailable'),
                                   'build-a'])
    read = Mock(return_value=[row()])
    monkeypatch.setattr(profiles, '_read_generation', generation)
    monkeypatch.setattr(profiles, '_read_all', read)

    engine = profiles.get_engine('everyone')
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS + 1
    assert profiles.get_engine('everyone') is engine
    isolated_cache[0] += profiles._CACHE_RETRY_SECONDS
    assert profiles.get_engine('everyone') is engine
    read.assert_called_once_with(4)


def test_new_generation_reloads_and_invalidates_old_engine(monkeypatch, isolated_cache):
    generation = ['weekly-build-a']
    read = Mock(side_effect=[[row('old')], [row('new')]])
    monkeypatch.setattr(profiles, '_read_generation', lambda: generation[0])
    monkeypatch.setattr(profiles, '_read_all', read)

    old_engine = profiles.get_engine('everyone')
    generation[0] = 'weekly-build-b'
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS + 1
    new_engine = profiles.get_engine('everyone')
    assert new_engine is not old_engine
    assert new_engine.rows[0]['wca_id'] == 'new'
    assert read.call_count == 2


def test_periodic_full_read_recovers_from_an_unchanged_generation(monkeypatch, isolated_cache):
    read = Mock(side_effect=[[row('old')], [row('updated')]])
    monkeypatch.setattr(profiles, '_read_generation', lambda: 'unchanged-marker')
    monkeypatch.setattr(profiles, '_read_all', read)

    assert profiles.get_engine('everyone').rows[0]['wca_id'] == 'old'
    isolated_cache[0] += profiles._FULL_REFRESH_SECONDS + 1
    assert profiles.get_engine('everyone').rows[0]['wca_id'] == 'updated'
    assert read.call_count == 2


def test_preexisting_database_uses_first_profile_as_generation(monkeypatch):
    query = Mock(side_effect=[[], [('1982PETR01', '2026-09-22 12:00:00')]])
    monkeypatch.setattr(profiles, '_read_generation', _ORIGINAL_READ_GENERATION)
    monkeypatch.setattr(profiles, '_query', query)

    assert profiles._read_generation() == 'legacy:1982PETR01:2026-09-22 12:00:00'
    assert query.call_count == 2
    assert 'ORDER BY wca_id LIMIT 1' in query.call_args.args[0]


def test_larger_cached_pool_serves_smaller_difficulties(monkeypatch):
    read = Mock(return_value=[row('easy', 1), row('hard', 3), row('everyone', 4)])
    monkeypatch.setattr(profiles, '_read_all', read)

    profiles.load_profiles(4)
    assert [r['wca_id'] for r in profiles.load_profiles(1)] == ['easy']
    assert [r['wca_id'] for r in profiles.load_profiles(3)] == ['easy', 'hard']
    read.assert_called_once_with(4)


def test_everyone_secret_uses_one_id_scan_then_indexed_lookup(monkeypatch):
    calls = []
    def query(sql, params):
        calls.append(sql)
        assert 'WHERE wca_id = %s' in sql
        return [('two', 'Cuber Two', 'US', '_North America', 1, 5, '{}')]

    directory = Mock(return_value=profiles._CompactNameIndex([
        ('one', 'Cuber One', 1), ('two', 'Cuber Two', 2)]))
    monkeypatch.setattr(profiles, '_read_directory', directory)
    monkeypatch.setattr(profiles, '_query', query)
    monkeypatch.setattr(profiles.random, 'choice', lambda ids: ids[-1])

    assert profiles.random_profile('everyone')['wca_id'] == 'two'
    assert profiles.random_profile('everyone')['wca_id'] == 'two'
    directory.assert_called_once()
    assert len(calls) == 1
    assert all('OFFSET' not in sql for sql in calls)


def test_name_directory_refreshes_after_rebuild(monkeypatch, isolated_cache):
    generation = ['a']
    def read_directory():
        if generation[0] == 'a':
            return profiles._CompactNameIndex([('one', 'Cuber One', 1)])
        return profiles._CompactNameIndex([('two', 'Cuber Two', 2)])
    directory = Mock(side_effect=read_directory)
    monkeypatch.setattr(profiles, '_read_directory', directory)
    monkeypatch.setattr(profiles, '_read_generation', lambda: generation[0])

    assert profiles._profile_ids() == ['one']
    generation[0] = 'b'
    isolated_cache[0] += profiles._CACHE_TTL_SECONDS + 1
    assert profiles._profile_ids() == ['two']
    assert directory.call_count == 2


def test_cold_directory_failure_backs_off_before_rescanning(monkeypatch, isolated_cache):
    directory = Mock(side_effect=[OSError('throttled'),
                                  profiles._CompactNameIndex([('one', 'Cuber One', 1)])])
    query = Mock(return_value=[('one', 'Cuber One', 'US', '_North America', 1)])
    monkeypatch.setattr(profiles, '_read_directory', directory)
    monkeypatch.setattr(profiles, '_query', query)

    assert profiles.search_by_name('cuber', include_all=True) == []
    assert profiles.search_by_name('cuber o', include_all=True) == []
    assert directory.call_count == 1  # No full-table rescan per keystroke.

    isolated_cache[0] += profiles._CACHE_RETRY_SECONDS
    assert profiles.search_by_name('cuber', include_all=True)[0]['wca_id'] == 'one'
    assert directory.call_count == 2


def test_solo_start_does_not_load_full_pool(monkeypatch):
    from app import app
    from blueprints import game
    monkeypatch.setattr(profiles, 'random_profile', lambda difficulty: row())
    full_load = Mock(side_effect=AssertionError('solo does not need the pool'))
    monkeypatch.setattr(profiles, 'get_engine', full_load)
    monkeypatch.setattr(game.tokens, 'seal', lambda value: 'offline-token')
    with app.test_client() as client:
        response = client.post('/api/game/solo/start', json={'difficulty': 'everyone'})
    assert response.status_code == 200
    assert 'pool_size' not in response.json
    full_load.assert_not_called()


def test_autocomplete_caches_db_results_without_attrs(monkeypatch):
    directory = Mock(return_value=profiles._CompactNameIndex([
        ('one', 'Cuber One', 7)]))
    query = Mock(return_value=[('one', 'Cuber One', 'US', '_North America', 7)])
    monkeypatch.setattr(profiles, '_read_directory', directory)
    monkeypatch.setattr(profiles, '_query', query)
    assert profiles.search_by_name('Cuber', include_all=True)[0]['wca_id'] == 'one'
    assert profiles.search_by_name('Cuber', include_all=True)[0]['wca_id'] == 'one'
    directory.assert_called_once()
    query.assert_called_once()
    assert 'WHERE wca_id IN (%s)' in query.call_args.args[0]
    assert all('LOWER(name)' not in call.args[0] for call in query.call_args_list)
    assert all('attrs' not in call.args[0] for call in query.call_args_list)


def test_name_index_preserves_substring_search_and_fame_order():
    index = profiles._CompactNameIndex([
        ('low', 'Mae Müller', 1),
        ('high', 'Zoë Müller', 9),
        ('other', 'Alan Zhang', 20),
    ])
    assert index.find('müll', 8) == ['high', 'low']
    assert index.find('zhang', 8) == ['other']
    assert index.find('lerzoë', 8) == []


def test_common_name_search_keeps_fame_order_after_index_fallback():
    rows = [(str(i), f'John {i}', i) for i in range(520)]
    index = profiles._CompactNameIndex(rows)
    assert index.find('john', 8) == [str(i) for i in range(519, 511, -1)]


def test_search_finds_last_names_without_matching_across_rows():
    rows = [
        ('one', 'Sofia García', 3),
        ('two', 'Gary Chen', 10),
        ('three', 'Aaron Garcia', 8),
        ('four', 'Mélanie García', 12),
    ]
    index = profiles._CompactNameIndex(rows)
    assert index.find('garcia', 8) == ['three']
    assert index.find('garcía', 8) == ['four', 'one']
    assert index.find('cíagary', 8) == []


def test_bulk_reads_stream_rows_without_buffering_a_second_copy(monkeypatch):
    class Cursor:
        def __init__(self):
            self.sql = ''

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def execute(self, sql, *_):
            self.sql = sql

        def __iter__(self):
            if 'WHERE tier' in self.sql:
                return iter([('one', 'Cuber One', 'US', '_North America',
                              7, 1, '{}')])
            return iter([('one', 'Cuber One', 7)])

        def fetchall(self):
            raise AssertionError('bulk reads must stream')

    cursor = Cursor()
    conn = Mock()
    conn.cursor.return_value = cursor
    monkeypatch.setattr(profiles, '_connect', lambda **kwargs: conn)

    assert profiles._read_all(1)[0]['wca_id'] == 'one'
    assert profiles._read_directory().find('cuber', 8) == ['one']
    assert all(call.args[0] is pymysql.cursors.SSCursor
               for call in conn.cursor.call_args_list)
    assert conn.close.call_count == 2
