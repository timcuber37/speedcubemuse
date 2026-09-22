"""SpeedCubeMuse Discord bot: WCA results (/query) and rules (/delegate)."""
import asyncio
import logging
import sys
from pathlib import Path

# Local development and the shared Docker image keep services/ and config.py
# one level up. The optional standalone bot image puts everything in /app.
_repo_root = Path(__file__).resolve().parent.parent
if (_repo_root / 'services').is_dir() and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import discord
from discord import app_commands
from discord.ext import commands

from config import COMMAND_PREFIX, DISCORD_GUILD_ID, DISCORD_TOKEN, MAX_QUERY_RESULTS
from delegate import UserCooldown, build_answer_embed, build_history
from services.nl_to_sql import NLToSQLService
from services.rag import DelegateRAGService
from services.wca_api import WCAService
from services.api_usage import usage_source

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

intents = discord.Intents.default()
intents.message_content = True


class SpeedCubeMuseBot(commands.Bot):
    async def setup_hook(self):
        if DISCORD_GUILD_ID:
            guild = discord.Object(id=int(DISCORD_GUILD_ID))
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info("Synced %d slash command(s) to guild %s", len(synced), DISCORD_GUILD_ID)
        else:
            synced = await self.tree.sync()
            logger.info("Synced %d slash command(s) globally", len(synced))


bot = SpeedCubeMuseBot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

nl_to_sql_service = NLToSQLService()
wca_service = WCAService()
delegate_service = DelegateRAGService()

followup_cooldown = UserCooldown(rate=5, per_seconds=60.0)


@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    for guild in bot.guilds:
        logger.info(f'  - {guild.name} (id: {guild.id})')


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    ch = message.channel
    if (isinstance(ch, discord.Thread)
            and ch.owner_id == (bot.user.id if bot.user else None)
            and not message.content.startswith(COMMAND_PREFIX)):
        await handle_thread_followup(message)
        return

    await bot.process_commands(message)


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param.name}")
    else:
        logger.error(f"Error in command {ctx.command}: {error}", exc_info=error)
        await ctx.send(f"An error occurred: {str(error)}")


# ---------------- Ask a Delegate ----------------

@bot.tree.command(name="delegate", description="Ask a question about the WCA Regulations & Guidelines")
@app_commands.describe(question="Your question about the WCA Regulations or Guidelines")
@app_commands.checks.cooldown(5, 60.0, key=lambda i: i.user.id)
@usage_source("discord")
async def delegate(interaction: discord.Interaction, question: app_commands.Range[str, 5, 1000]):
    logger.info(f"/delegate invoked by {interaction.user} with question: {question[:80]}")
    await interaction.response.defer(thinking=True)

    try:
        result = await asyncio.to_thread(delegate_service.answer, [], question)
    except Exception as e:
        logger.error(f"Error answering /delegate question: {e}", exc_info=e)
        await interaction.followup.send("❌ Sorry, something went wrong answering that. Please try again.")
        return

    embed = build_answer_embed(result, question=question)
    msg = await interaction.followup.send(embed=embed, wait=True)

    # Open a follow-up thread only from a normal text channel; inside an
    # existing thread or DM the answer is delivered inline.
    if isinstance(interaction.channel, discord.TextChannel):
        try:
            # Create via the channel: followup.send() returns a WebhookMessage,
            # which lacks guild info so msg.create_thread() raises ValueError.
            thread = await interaction.channel.create_thread(
                name=question[:100], message=msg, auto_archive_duration=1440)
            await thread.send("💬 Ask follow-up questions in this thread — I'll keep the context.")
        except (discord.Forbidden, discord.HTTPException) as e:
            logger.warning(f"Could not create follow-up thread: {e}")


@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction,
                               error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        await interaction.response.send_message(
            f"⏳ Slow down — try again in {error.retry_after:.0f}s.", ephemeral=True)
        return
    logger.error(f"Error in slash command: {error}", exc_info=error)
    send = (interaction.followup.send if interaction.response.is_done()
            else interaction.response.send_message)
    try:
        await send("❌ An unexpected error occurred. Please try again.", ephemeral=True)
    except discord.HTTPException:
        pass


@usage_source("discord")
async def handle_thread_followup(message: discord.Message):
    """Answer a follow-up question in a bot-owned /delegate thread."""
    question = message.content.strip()
    if not question:
        return

    retry_after = followup_cooldown.check(message.author.id)
    if retry_after > 0:
        await message.reply(f"⏳ Slow down — try again in {retry_after:.0f}s.")
        return

    thread = message.channel
    logger.info(f"Thread follow-up from {message.author}: {question[:80]}")
    try:
        async with thread.typing():
            history = await build_history(thread, before=message, bot_user_id=bot.user.id)
            result = await asyncio.to_thread(
                delegate_service.answer, history, question[:2000])
        await message.reply(embed=build_answer_embed(result))
    except Exception as e:
        logger.error(f"Error answering thread follow-up: {e}", exc_info=e)
        await message.reply("❌ Sorry, something went wrong answering that. Please try again.")


# ---------------- WCA stats ----------------

@bot.hybrid_command(name='query', aliases=['q', 'ask'],
                    description='Ask about WCA competitors, records, and results')
@app_commands.describe(question='Your question about WCA competition results')
@usage_source("discord")
async def query_wca(ctx, *, question: str):
    """
    Query WCA statistics using natural language.

    Usage: /query question:<your question>
    Example: /query question:What is the world record for 3x3?
    """
    logger.info(f"Query command invoked by {ctx.author} with question: {question}")

    question = question.strip()
    if not question:
        await ctx.send("Please enter a question, such as: Who has the fastest 3x3 single?")
        return

    # Acknowledge slash commands before the AI and database calls begin.
    await ctx.defer()
    thinking_msg = await ctx.send("🤔 Looking up your question...")

    try:
        # Step 1: Translate to SQL and execute (retries once with the DB error on failure)
        logger.info(f"User question: {question}")
        sql_query, results = await nl_to_sql_service.answer_question(question, wca_service.execute_query)
        logger.info(f"Generated SQL: {sql_query}")

        if not sql_query:
            await thinking_msg.edit(content="❌ I couldn't understand that question. Try naming the event, competitor, or record you're interested in.")
            return

        if not results:
            await thinking_msg.edit(content="No matching results found. Try a different question.")
            return

        # Step 2: Format and send results
        formatted_results = wca_service.format_results(results, max_results=MAX_QUERY_RESULTS)

        # Leave room for code fences within Discord's 2000-character limit.
        chunks = [formatted_results[i:i+1900] for i in range(0, len(formatted_results), 1900)]
        for i, chunk in enumerate(chunks):
            if i == 0:
                await thinking_msg.edit(content=f"```\n{chunk}\n```")
            else:
                await ctx.send(f"```\n{chunk}\n```")

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=e)
        await thinking_msg.edit(content="❌ Sorry, I couldn't look up those results. Please try again.")


@bot.hybrid_command(name='help', aliases=['h'], description='See commands and example questions')
async def help_command(ctx):
    """Show help information."""
    help_text = """
**SpeedCubeMuse Bot Help**

**Commands:**
`/delegate <question>` - Ask about the WCA Regulations & Guidelines
  Answers cite the official regulations, and open a thread for follow-up questions.

`/query <question>` - Ask about competitors, records, and results
  Examples:
    - `/query question:What is the world record for 3x3?`
    - `/query question:Who has the most world records?`
    - `/query question:Show me the top 10 fastest times for 2x2`

`/help` - Show this help message
`/ping` - Check whether the bot is responding

**Tips:**
- Be specific in your questions
- You can ask about records, rankings, competitions, and more
- Choose a slash command, then enter your question in its question field
"""
    await ctx.send(help_text)


@bot.hybrid_command(name='ping', description='Check whether the bot is responding')
async def ping(ctx):
    """Check if the bot is responsive."""
    logger.info(f"Ping command invoked by {ctx.author}")
    await ctx.send(f'Pong! Latency: {round(bot.latency * 1000)}ms')


def main():
    """Main entry point for the bot."""
    if not DISCORD_TOKEN:
        logger.error("DISCORD_TOKEN not found in environment variables!")
        logger.error("Please create a .env file with your Discord token.")
        return

    try:
        bot.run(DISCORD_TOKEN)
    except discord.LoginFailure:
        logger.error("Invalid Discord token!")
    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=e)


if __name__ == "__main__":
    main()
