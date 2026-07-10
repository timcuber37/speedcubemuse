"""Main Discord bot file for WCA Statistics Bot."""
import discord
from discord.ext import commands
import asyncio
import logging
from config import DISCORD_TOKEN, COMMAND_PREFIX, MAX_QUERY_RESULTS
from services.nl_to_sql import NLToSQLService
from services.wca_api import WCAService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up Discord intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize bot (disable default help command to use our custom one)
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

# Initialize services
nl_to_sql_service = NLToSQLService()
wca_service = WCAService()


@bot.event
async def on_ready():
    """Called when the bot is ready and connected to Discord."""
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info(f'Bot is in {len(bot.guilds)} guild(s)')
    for guild in bot.guilds:
        logger.info(f'  - {guild.name} (id: {guild.id})')


@bot.event
async def on_message(message):
    """Log all messages for debugging."""
    if message.author == bot.user:
        return

    logger.info(f'Message received from {message.author}: {message.content[:50]}')

    # Important: Process commands after logging
    await bot.process_commands(message)


@bot.event
async def on_command_error(ctx, error):
    """Handle command errors."""
    if isinstance(error, commands.CommandNotFound):
        return  # Ignore unknown commands
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: {error.param.name}")
    else:
        logger.error(f"Error in command {ctx.command}: {error}", exc_info=error)
        await ctx.send(f"An error occurred: {str(error)}")


@bot.command(name='query', aliases=['q', 'ask'])
async def query_wca(ctx, *, question: str):
    """
    Query WCA statistics using natural language.

    Usage: !wca query <your question>
    Example: !wca query What is the world record for 3x3?
    """
    logger.info(f"Query command invoked by {ctx.author} with question: {question}")

    if not question:
        await ctx.send("Please provide a question! Usage: `!wca query <your question>`")
        return

    # Send "thinking" message
    thinking_msg = await ctx.send("🤔 Processing your question...")
    
    try:
        # Step 1: Translate to SQL and execute (retries once with the DB error on failure)
        logger.info(f"User question: {question}")
        sql_query, results = await nl_to_sql_service.answer_question(question, wca_service.execute_query)
        logger.info(f"Generated SQL: {sql_query}")

        if not sql_query:
            await thinking_msg.edit(content="❌ Could not generate a valid SQL query from your question.")
            return

        if not results:
            await thinking_msg.edit(content="❌ No results found for your query.")
            return
        
        # Step 2: Format and send results
        formatted_results = wca_service.format_results(results, max_results=MAX_QUERY_RESULTS)
        
        # Discord has a 2000 character limit per message
        if len(formatted_results) > 2000:
            # Split into multiple messages or use file upload
            chunks = [formatted_results[i:i+1900] for i in range(0, len(formatted_results), 1900)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await thinking_msg.edit(content=f"```\n{chunk}\n```")
                else:
                    await ctx.send(f"```\n{chunk}\n```")
        else:
            await thinking_msg.edit(content=f"```\n{formatted_results}\n```")
            
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=e)
        await thinking_msg.edit(content=f"❌ An error occurred: {str(e)}")


@bot.command(name='help', aliases=['h'])
async def help_command(ctx):
    """Show help information."""
    help_text = """
**WCA Statistics Bot Help**

**Commands:**
`!wca query <question>` - Ask a question about WCA statistics
  Examples:
    - `!wca query What is the world record for 3x3?`
    - `!wca query Who has the most world records?`
    - `!wca query Show me the top 10 fastest times for 2x2`

`!wca help` - Show this help message

**Tips:**
- Be specific in your questions
- You can ask about records, rankings, competitions, and more
- The bot translates your question to SQL and queries the WCA database
"""
    await ctx.send(help_text)


@bot.command(name='ping')
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

