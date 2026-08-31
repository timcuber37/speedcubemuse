"""Helpers for the /delegate slash command: embed building, thread history, cooldowns."""
import logging
import re
import time
from collections import deque

import discord

logger = logging.getLogger(__name__)

QUESTION_FIELD = "Question"
SOURCES_FIELD = "Sources"
FOOTER_TEXT = "Not official advice — for on-site rulings, ask a WCA Delegate."

MAX_EMBED_DESCRIPTION = 4096
MAX_FIELD_VALUE = 1024
MAX_HISTORY_FETCH = 50
MAX_MESSAGE_CHARS = 2000

_CITATION_RE = re.compile(r'\[([^\[\]]+)\]')


def build_answer_embed(result: dict, question: str | None = None) -> discord.Embed:
    """Render a DelegateRAGService.answer() result as a Discord embed.

    The Question field doubles as the machine-readable user turn that
    build_history() reads back when reconstructing a thread conversation.
    """
    answer = (result.get('answer') or '').strip()
    if len(answer) > MAX_EMBED_DESCRIPTION:
        answer = answer[:MAX_EMBED_DESCRIPTION - 1] + '…'

    embed = discord.Embed(description=answer, color=discord.Color.blurple())

    if question:
        embed.add_field(name=QUESTION_FIELD, value=question[:MAX_FIELD_VALUE], inline=False)

    sources_value = _format_sources(answer, result.get('sources') or [])
    if sources_value:
        embed.add_field(name=SOURCES_FIELD, value=sources_value, inline=False)

    embed.set_footer(text=FOOTER_TEXT)
    return embed


def _format_sources(answer: str, sources: list[dict]) -> str:
    """Markdown links for source regulation IDs, answer-cited IDs first, capped at 1024."""
    if not sources:
        return ''

    # IDs the answer actually cites (e.g. "[9b1]" or "[9b1, 9b2]"), in citation order
    cited_order: list[str] = []
    for group in _CITATION_RE.findall(answer):
        for reg_id in (part.strip() for part in group.split(',')):
            if reg_id and reg_id not in cited_order:
                cited_order.append(reg_id)

    by_id = {}
    for s in sources:
        by_id.setdefault(s['regulation_id'], s)

    ordered = [by_id[rid] for rid in cited_order if rid in by_id]
    ordered += [s for rid, s in by_id.items() if rid not in cited_order]

    links = []
    total = 0
    for s in ordered:
        link = f"[{s['regulation_id']}]({s['url']})" if s.get('url') else f"`{s['regulation_id']}`"
        # +2 for the ", " separator
        if total + len(link) + 2 > MAX_FIELD_VALUE:
            break
        links.append(link)
        total += len(link) + 2
    return ', '.join(links)


async def build_history(thread: discord.Thread, before: discord.Message,
                        bot_user_id: int) -> list[dict]:
    """Rebuild the conversation from a bot-owned thread as role/content dicts.

    Stateless by design (mirrors the web client re-sending history each request):
    the starter message's embed carries the original question + answer, and
    thread messages alternate user follow-ups with bot embed replies.
    DelegateRAGService._trim_history caps the result at 16 messages server-side.
    """
    history: list[dict] = []

    starter = thread.starter_message
    if starter is None and thread.parent is not None:
        try:
            # A public thread's ID equals its starter message's ID
            starter = await thread.parent.fetch_message(thread.id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            starter = None
    if starter is not None:
        history.extend(_message_to_turns(starter, bot_user_id))

    # Newest-first fetch so long threads keep their most recent context
    # (the service trims to the LAST 16 messages), then restore chronology.
    thread_msgs = [msg async for msg in thread.history(limit=MAX_HISTORY_FETCH)
                   if msg.id != before.id]
    for msg in reversed(thread_msgs):
        history.extend(_message_to_turns(msg, bot_user_id))

    return history


def _message_to_turns(msg: discord.Message, bot_user_id: int) -> list[dict]:
    """Convert one Discord message into zero or more conversation turns."""
    if msg.author.id == bot_user_id:
        # Only embeds carry conversation turns; plain-content bot messages
        # (the thread intro, error notices) are not part of the dialogue.
        turns = []
        for embed in msg.embeds:
            for field in embed.fields:
                if field.name == QUESTION_FIELD and field.value:
                    turns.append({'role': 'user', 'content': field.value})
            if embed.description:
                turns.append({'role': 'assistant', 'content': embed.description})
        return turns

    if msg.author.bot:
        return []

    content = msg.content.strip()
    return [{'role': 'user', 'content': content[:MAX_MESSAGE_CHARS]}] if content else []


class UserCooldown:
    """Sliding-window per-user rate limit for thread follow-ups.

    app_commands cooldown decorators don't cover raw on_message handlers.
    """

    def __init__(self, rate: int, per_seconds: float):
        self.rate = rate
        self.per_seconds = per_seconds
        self._hits: dict[int, deque[float]] = {}

    def check(self, user_id: int) -> float:
        """Record a hit; return 0 if allowed, else seconds until the next slot."""
        now = time.monotonic()
        hits = self._hits.setdefault(user_id, deque())
        while hits and now - hits[0] > self.per_seconds:
            hits.popleft()
        if len(hits) >= self.rate:
            return self.per_seconds - (now - hits[0])
        hits.append(now)
        return 0.0
