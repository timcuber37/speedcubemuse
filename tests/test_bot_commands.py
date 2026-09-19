"""Offline bot checks: python -m unittest discover -s tests -p test_bot_commands.py."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from discord.ext import commands
from discord.ext.commands.view import StringView


class BotCommandsTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        bot_path = Path(__file__).resolve().parents[1] / 'delegate-bot' / 'bot.py'
        spec = importlib.util.spec_from_file_location('speedcubemuse_bot_test', bot_path)
        cls.module = importlib.util.module_from_spec(spec)
        # Import the real command definitions without constructing API clients.
        with (
            patch.object(sys, 'path', [str(bot_path.parent), *sys.path]),
            patch('services.nl_to_sql.NLToSQLService'),
            patch('services.wca_api.WCAService'),
            patch('services.rag.DelegateRAGService'),
        ):
            spec.loader.exec_module(cls.module)

    def setUp(self):
        self.answer = AsyncMock(return_value=('SELECT result', [{'result': '3.47'}]))
        self.format_results = Mock(return_value='result\n3.47')
        self.enterContext(patch.object(
            self.module, 'nl_to_sql_service', SimpleNamespace(answer_question=self.answer)))
        self.enterContext(patch.object(self.module, 'wca_service', SimpleNamespace(
            execute_query=AsyncMock(), format_results=self.format_results)))
        self.enterContext(patch.object(self.module, 'logger'))

    def make_slash_context(self):
        """Use discord.py's actual defer/send routing with mocked Discord transport."""
        self.reply = SimpleNamespace(edit=AsyncMock())
        response = SimpleNamespace(is_done=Mock(return_value=False), send_message=AsyncMock())

        async def defer(**kwargs):
            response.is_done.return_value = True

        response.defer = AsyncMock(side_effect=defer)
        self.interaction = SimpleNamespace(
            response=response,
            followup=SimpleNamespace(send=AsyncMock(return_value=self.reply)),
            original_response=AsyncMock(return_value=self.reply),
            is_expired=Mock(return_value=False),
        )
        return commands.Context(
            message=SimpleNamespace(_state=self.module.bot._connection, author='test-user'),
            bot=self.module.bot,
            view=StringView(''),
            interaction=self.interaction,
        )

    async def test_questions_are_registered_as_slash_commands(self):
        tree = self.module.bot.tree
        self.assertEqual({command.name for command in tree.get_commands()},
                         {'query', 'delegate', 'help', 'ping'})
        for name in ('query', 'delegate'):
            payload = tree.get_command(name).to_dict(tree)
            question = next(option for option in payload['options'] if option['name'] == 'question')
            self.assertTrue(question['required'])
            self.assertEqual(question['type'], 3)  # Discord's string option type
        for alias in ('q', 'ask'):
            self.assertIs(self.module.bot.get_command(alias), self.module.query_wca)

    async def test_query_acknowledges_before_lookup_and_returns_results(self):
        ctx = self.make_slash_context()

        async def lookup(question, execute):
            self.interaction.response.defer.assert_awaited_once()
            return 'SELECT result', [{'result': '3.47'}]

        self.answer.side_effect = lookup
        await self.module.query_wca(ctx, question='  Who has the fastest 3x3 single?  ')
        self.answer.assert_awaited_once_with(
            'Who has the fastest 3x3 single?', self.module.wca_service.execute_query)
        self.interaction.response.send_message.assert_not_awaited()
        self.interaction.followup.send.assert_awaited_once()
        self.reply.edit.assert_awaited_once_with(content='```\nresult\n3.47\n```')

    async def test_results_fit_discord_limit_without_losing_content(self):
        for length in (1900, 1993, 2000, 4001):
            with self.subTest(length=length):
                ctx = self.make_slash_context()
                formatted = 'x' * length
                self.format_results.return_value = formatted
                await self.module.query_wca(ctx, question='Show me the fastest results')
                messages = [self.reply.edit.await_args.kwargs['content']]
                messages.extend(call.kwargs['content'] for call in
                                self.interaction.followup.send.await_args_list[1:])
                self.assertTrue(all(len(message) <= 2000 for message in messages))
                self.assertEqual(''.join(message[4:-4] for message in messages), formatted)

    async def test_blank_question_does_not_start_lookup(self):
        ctx = self.make_slash_context()
        await self.module.query_wca(ctx, question='   ')
        self.answer.assert_not_awaited()
        self.interaction.response.defer.assert_not_awaited()
        self.interaction.response.send_message.assert_awaited_once()

    async def test_no_results_and_unrecognized_question_finish_the_response(self):
        for result in ((None, None), ('SELECT result', [])):
            with self.subTest(result=result):
                ctx = self.make_slash_context()
                self.answer.return_value = result
                await self.module.query_wca(ctx, question='Find results')
                self.reply.edit.assert_awaited_once()
                self.assertNotIn('SQL', self.reply.edit.await_args.kwargs['content'])
                self.format_results.assert_not_called()

    async def test_lookup_failure_gives_a_retry_message(self):
        ctx = self.make_slash_context()
        self.answer.side_effect = RuntimeError('internal connection details')
        await self.module.query_wca(ctx, question='Find results')
        message = self.reply.edit.await_args.kwargs['content']
        self.assertIn('Please try again', message)
        self.assertNotIn('internal connection details', message)

    async def test_help_describes_both_slash_questions(self):
        ctx = self.make_slash_context()
        await self.module.help_command(ctx)
        message = self.interaction.response.send_message.await_args.kwargs['content']
        self.assertIn('/query', message)
        self.assertIn('/delegate', message)
        self.assertNotIn('!wca', message)

    async def test_startup_syncs_the_global_command_tree(self):
        tree = self.module.bot.tree
        with (
            patch.object(self.module, 'DISCORD_GUILD_ID', None),
            patch.object(tree, 'sync', AsyncMock(return_value=tree.get_commands())) as sync,
        ):
            await self.module.bot.setup_hook()
            sync.assert_awaited_once_with()
