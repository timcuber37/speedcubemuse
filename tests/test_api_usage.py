"""Offline accounting and integration checks; never call paid APIs or a DB.

API_USAGE_PERSIST=false python -m unittest discover -s tests -p test_api_usage.py
"""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from services import api_usage as usage
from scripts import usage_report as report


def response(model='claude-sonnet-5', text='{"sql":"SELECT 1","error":null}', **tokens):
    counts = {'input_tokens': 1000, 'output_tokens': 100,
              'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0, **tokens}
    return SimpleNamespace(model=model, usage=SimpleNamespace(**counts),
                           content=[SimpleNamespace(type='text', text=text)],
                           _request_id='req_test')


class AccountingTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.enterContext(patch.object(usage, 'API_USAGE_ENABLED', True))
        self.enterContext(patch.object(usage, '_emit', side_effect=self.events.append))

    def test_mixed_cache_ttls_are_charged_once_and_output_is_included(self):
        reply = response(cache_read_input_tokens=3000, cache_creation_input_tokens=6000,
                         cache_creation={'ephemeral_5m_input_tokens': 4000,
                                         'ephemeral_1h_input_tokens': 2000})
        with usage.APICall('stats.generate_sql', 'anthropic', 'claude-sonnet-5') as call:
            call.capture(reply)
        event = self.events[0]
        # 1000*2 + 3000*.2 + 4000*2.5 + 2000*4 + 100*10 per million.
        self.assertEqual(Decimal(event['estimated_cost_usd']), Decimal('0.0216'))
        self.assertEqual(event['cache_creation_input_tokens'], 6000)

    def test_snapshot_uses_served_model_price_and_aggregate_cache_fallback(self):
        reply = response(model='claude-haiku-4-5-20251001', cache_creation_input_tokens=1000)
        with usage.APICall('game.parse_question', 'anthropic', 'claude-sonnet-5') as call:
            call.capture(reply)
        event = self.events[0]
        self.assertEqual(Decimal(event['estimated_cost_usd']), Decimal('0.00275'))
        self.assertEqual(event['requested_model'], 'claude-sonnet-5')
        self.assertEqual(event['model'], reply.model)

    def test_voyage_uses_reported_tokens_including_all_reranked_documents(self):
        with usage.APICall('delegate.rerank', 'voyage', 'rerank-2-lite') as call:
            call.capture(SimpleNamespace(total_tokens=12345))
        self.assertEqual(Decimal(self.events[0]['estimated_cost_usd']), Decimal('0.0002469'))

    def test_unknown_model_or_missing_usage_is_not_reported_as_free(self):
        for reply in (response(model='claude-future'), SimpleNamespace(model='claude-sonnet-5'),
                      response(service_tier='priority')):
            with usage.APICall('test', 'anthropic', 'claude-sonnet-5') as call:
                call.capture(reply)
        self.assertTrue(all(event['estimated_cost_usd'] is None for event in self.events))

    def test_provider_failure_is_preserved_and_records_no_sensitive_message(self):
        error = RuntimeError('secret-key and private prompt')
        with self.assertRaises(RuntimeError) as caught:
            with usage.APICall('delegate.answer', 'anthropic', 'claude-sonnet-5'):
                raise error
        self.assertIs(caught.exception, error)
        event = self.events[0]
        self.assertEqual(event['status'], 'error')
        self.assertIsNone(event['estimated_cost_usd'])
        self.assertNotIn('secret-key', json.dumps(event))

    def test_meter_failure_does_not_break_success_or_mask_provider_error(self):
        with patch.object(usage, '_emit', side_effect=RuntimeError('storage down')):
            with usage.APICall('test', 'anthropic', 'claude-sonnet-5') as call:
                call.capture(response())
            with self.assertRaisesRegex(ValueError, 'provider failed'):
                with usage.APICall('test', 'anthropic', 'claude-sonnet-5'):
                    raise ValueError('provider failed')

    def test_disabled_tracking_does_not_emit(self):
        with patch.object(usage, 'API_USAGE_ENABLED', False):
            with usage.APICall('test', 'anthropic', 'claude-sonnet-5') as call:
                call.capture(response())
            usage.record_cache_hit('game.parse_question', 'claude-haiku-4-5')
        self.assertEqual(self.events, [])

    def test_full_queue_does_not_block_request_and_reports_log_fallback(self):
        writer = usage.UsageWriter()
        writer.thread = Mock()  # prevent worker start
        writer.queue = Mock()
        writer.queue.put_nowait.side_effect = usage.queue.Full
        with self.assertLogs(usage.logger, level='WARNING') as logs:
            writer.submit({'event_id': 'abc'})
        self.assertIn('remains in application logs', logs.output[0])


class ContextTests(unittest.IsolatedAsyncioTestCase):
    async def test_parallel_bot_requests_propagate_into_threads_without_leaking(self):
        events = []

        @usage.usage_source('discord')
        async def handle():
            usage.record_cache_hit('game.parse_question', 'claude-haiku-4-5')
            await asyncio.to_thread(usage.record_cache_hit, 'game.parse_question', 'claude-haiku-4-5')

        with patch.object(usage, '_emit', side_effect=events.append):
            await asyncio.gather(handle(), handle())
        ids = {event['request_id'] for event in events}
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(sum(e['request_id'] == identifier for e in events) == 2 for identifier in ids))
        self.assertTrue(all(e['source'] == 'discord' for e in events))
        self.assertIsNone(usage._context.get())


class IntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.events = []
        self.enterContext(patch.object(usage, '_emit', side_effect=self.events.append))

    async def test_sql_generation_repair_summary_are_attributed_separately(self):
        from services.nl_to_sql import NLToSQLService
        service = NLToSQLService()
        service.client = SimpleNamespace(messages=SimpleNamespace(create=AsyncMock(side_effect=[
            response(text='{"sql":"SELECT bad","error":null}'),
            response(), response(text='One result.'),
        ])))
        execute = AsyncMock(side_effect=[[{'error': True, 'message': 'unknown column'}], [{'result': 1}]])
        with usage.usage_scope('web'):
            _, results = await service.answer_question('Count results', execute, model='sonnet')
            summary = await service.summarize_results('Count results', results, model='sonnet')
        self.assertEqual(summary, 'One result.')
        self.assertEqual([e['operation'] for e in self.events],
                         ['stats.generate_sql', 'stats.repair_sql', 'stats.summarize'])
        self.assertEqual(len({e['request_id'] for e in self.events}), 1)

    async def test_delegate_records_all_four_billable_stages_on_followup(self):
        from services.rag import DelegateRAGService
        service = DelegateRAGService()
        service.anthropic = SimpleNamespace(messages=SimpleNamespace(create=Mock(side_effect=[
            response(model='claude-haiku-4-5-20251001', text='rewritten'), response(text='Answer [1a].')
        ])))
        service.voyage = SimpleNamespace(
            embed=Mock(return_value=SimpleNamespace(total_tokens=10, embeddings=[[0.1]])),
            rerank=Mock(return_value=SimpleNamespace(total_tokens=40, results=[SimpleNamespace(index=0)])))
        hit = {'regulation_id': '1a', 'kind': 'regulation', 'content': 'Source'}
        database = Mock()
        database.rpc.return_value.execute.return_value.data = [hit]
        with (patch('services.rag.get_supabase', return_value=database),
              patch.object(service, '_lookup_by_ids', return_value=[]),
              patch.object(service, '_expand', side_effect=lambda hits: hits),
              usage.usage_scope('web')):
            result = service.answer([{'role': 'user', 'content': 'prior question'}], 'Follow up?')
        self.assertEqual(result['answer'], 'Answer [1a].')
        self.assertEqual([e['operation'] for e in self.events],
                         ['delegate.rewrite', 'delegate.embed', 'delegate.rerank', 'delegate.answer'])

    async def test_game_cache_hit_avoids_model_and_records_zero_cost(self):
        from services.game.question_parser import QuestionParser, _SCHEMA_VERSION
        parser = QuestionParser()
        parser.client = Mock()
        with patch.object(parser, '_cache_get', return_value={
            'v': _SCHEMA_VERSION, 'attribute': 'has_wr', 'op': 'is', 'value': 'true'
        }):
            predicate, error = parser.parse('Do they have a world record?')
        self.assertIsNotNone(predicate)
        self.assertIsNone(error)
        parser.client.messages.create.assert_not_called()
        self.assertEqual(self.events[0]['status'], 'cache_hit')
        self.assertEqual(self.events[0]['estimated_cost_usd'], '0')


class PersistenceTests(unittest.TestCase):
    def test_log_record_survives_persistence_submission_failure(self):
        with (patch.object(usage, 'API_USAGE_ENABLED', True),
              patch.object(usage, 'API_USAGE_PERSIST', True),
              patch.object(usage._writer, 'submit', side_effect=OSError('unavailable')),
              self.assertLogs(usage.logger, level='INFO') as logs):
            with usage.APICall('stats.generate_sql', 'anthropic', 'claude-sonnet-5') as call:
                call.capture(response())
        self.assertIn('api_usage {', logs.output[0])
        self.assertIn('capture failed', logs.output[1])

    def test_batch_is_idempotent_utc_and_shared_across_sources(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        writer = usage.UsageWriter()
        events = [{'event_id': 'a', 'occurred_at': '2026-09-22T10:00:00-04:00', 'source': 'web'},
                  {'event_id': 'b', 'occurred_at': '2026-09-22T14:00:00+00:00', 'source': 'discord'}]
        with patch.object(usage, 'connect_usage_db', return_value=connection):
            writer._write_batch(events)
            writer._write_batch(events)
        cursor.execute.assert_called_once_with(usage.CREATE_TABLE_SQL)
        sql, rows = cursor.executemany.call_args.args
        self.assertIn('ON DUPLICATE KEY UPDATE', sql)
        self.assertEqual(rows[0][1], rows[1][1])
        self.assertIsNone(rows[0][1].tzinfo)
        self.assertEqual(rows[0][1].hour, 14)

    def test_background_failure_retries_then_finishes_without_losing_log_fallback(self):
        writer = usage.UsageWriter()
        writer.queue.put({'event_id': 'a'})
        writer.stopping.set()  # drain immediately, with no sleep in this test
        with patch.object(writer, '_write_batch', side_effect=OSError) as write:
            with self.assertLogs(usage.logger, level='WARNING') as logs:
                writer._run()
        self.assertEqual(write.call_count, 2)
        self.assertEqual(writer.queue.unfinished_tasks, 0)
        self.assertIn('remain in application logs', logs.output[0])


class ReportTests(unittest.TestCase):
    since = datetime(2026, 9, 1, tzinfo=timezone.utc)
    until = datetime(2026, 10, 1, tzinfo=timezone.utc)

    def event(self, **changes):
        return {'type': 'api_usage', 'event_id': 'a', 'occurred_at': '2026-09-22T00:00:00Z',
                'operation': 'game.parse_question', 'source': 'web', 'status': 'success',
                'input_tokens': 100, 'cache_read_input_tokens': 900,
                'duration_ms': 100, 'estimated_cost_usd': '0.002', **changes}

    def test_cache_hits_do_not_dilute_per_call_cost_or_token_cache_rate(self):
        rows = report.summarize([self.event(), self.event(status='cache_hit', estimated_cost_usd='0')],
                                self.since, self.until, ['operation'])
        self.assertEqual(rows[0]['calls'], 1)
        self.assertEqual(rows[0]['result_cache_hits'], 1)
        self.assertEqual(rows[0]['avg_usd_per_call'], '0.002')
        self.assertEqual(rows[0]['prompt_cache_read_pct'], 90)

    def test_unknown_cost_is_visible_and_average_is_not_understated(self):
        rows = report.summarize([self.event(), self.event(status='error', estimated_cost_usd=None)],
                                self.since, self.until, ['operation'])
        self.assertEqual(rows[0]['known_estimated_usd'], '0.002')
        self.assertEqual(rows[0]['unpriced_calls'], 1)
        self.assertIsNone(rows[0]['avg_usd_per_call'])

    def test_group_filter_utc_dates_and_exclusive_end(self):
        events = [self.event(), self.event(source='discord'),
                  self.event(occurred_at='2026-09-30T20:00:00-04:00')]
        rows = report.summarize(events, self.since, self.until, ['day', 'source'], source='web')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['day'], '2026-09-22')
        self.assertEqual(rows[0]['calls'], 1)

    def test_log_export_deduplication_and_json_cli_without_database(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'usage.log'
            line = 'INFO:services.api_usage:api_usage ' + json.dumps(self.event())
            path.write_text('unrelated log\n' + line + '\n' + line + '\n')
            with patch('sys.stdout', new_callable=io.StringIO) as output:
                code = report.main(['--logs', str(path), '--since', '2026-09-01',
                                    '--until', '2026-10-01', '--format', 'json'])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue())['rows'][0]['calls'], 1)

    def test_report_query_is_parameterized_and_missing_table_is_clear(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.execute.side_effect = usage.pymysql.err.ProgrammingError(1146, 'missing')
        with patch.object(report, 'connect_usage_db', return_value=connection):
            with self.assertRaisesRegex(RuntimeError, 'first API call'):
                list(report.database_events(self.since, self.until))
        sql, parameters = cursor.execute.call_args.args
        self.assertIn('occurred_at >= %s AND occurred_at < %s', sql)
        self.assertEqual(len(parameters), 2)
        connection.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
