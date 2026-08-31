"""Ask a Delegate — Retrieval-Augmented Generation over WCA Regulations + Guidelines."""
import logging
import re
from typing import Optional

import voyageai
from anthropic import Anthropic

from config import (ANTHROPIC_API_KEY, ANTHROPIC_MODEL, DELEGATE_MAX_HISTORY_TURNS,
                    DELEGATE_RERANK_INITIAL_K, DELEGATE_RETRIEVAL_K,
                    VOYAGE_API_KEY, VOYAGE_EMBED_MODEL, VOYAGE_RERANK_MODEL)
from services.auth import get_supabase

logger = logging.getLogger(__name__)

REWRITER_MODEL = "claude-haiku-4-5-20251001"

# Matches WCA regulation IDs: 1a, 9b1, 9b1a, 9b1+, A4, A4a, B3+, etc.
# Requires either a digit-letter or letter-digit start so we don't match English words.
_REG_ID_RE = re.compile(
    r'\b(?:\d{1,2}[a-z][a-zA-Z0-9]*\+?|[A-E][1-9][a-zA-Z0-9]*\+?)\b'
)

_SELECT_COLS = (
    'regulation_id, article_num, article_title, section_path, content, kind, url'
)

_SYSTEM_PROMPT_TMPL = """You are "Ask a Delegate", an expert assistant on the WCA (World Cube Association) Regulations and Guidelines.

You have been given a curated set of relevant sources from the official WCA Regulations and Guidelines. Use them to answer the user's question directly and confidently.

== CITED SOURCES ==
{sources}
== END SOURCES ==

How to answer:
1. **Answer the question** when the sources address it — even when you have to synthesize across multiple regulations. Quote the regulation text directly when it's the clearest way to convey the rule.
2. **Cite every factual claim** with the regulation ID in square brackets — `[9b1]` for Regulations, `[9b1+]` for Guidelines. Inline. Group multiple as `[9b1, 9b2]`.
3. **Be clear** about Regulation (binding) vs Guideline (explanatory, non-binding) when the distinction affects the answer.
4. If a regulation references another by ID (e.g., "see Regulation 4d"), only cite IDs you actually have text for in the sources above.
5. Only say you "don't have enough information" when the sources truly don't cover the topic — not for routine factual questions.
6. Reserve "consult an on-site WCA Delegate" for genuinely contested judgments at an actual competition. Do NOT add that disclaimer to every answer.

Length: 1–4 sentences for simple questions; concise bullet points for multi-part answers."""


_REWRITER_SYSTEM = """You rewrite a follow-up question from a multi-turn conversation about the WCA Regulations into a single standalone search query that captures all relevant context from the prior turns.

Expand abbreviations you recognize (BLD = blindfolded, FMC = fewest moves, OH = one-handed, MBLD = multi-blind).

Output ONLY the rewritten query — no prefix, no quotes, no explanation. Keep it under 25 words."""


class DelegateRAGService:
    """RAG pipeline: rewrite → hybrid retrieve → rerank → expand → generate."""

    def __init__(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set; Ask a Delegate will not work")
            self.anthropic = None
        else:
            self.anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

        if not VOYAGE_API_KEY:
            logger.warning("VOYAGE_API_KEY not set; Ask a Delegate will not work")
            self.voyage = None
        else:
            self.voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

        self.model = ANTHROPIC_MODEL

    # ---------------- Public API ----------------

    def is_ready(self) -> bool:
        return (self.anthropic is not None
                and self.voyage is not None
                and get_supabase() is not None)

    def answer(self, history: list[dict], question: str) -> dict:
        """Full pipeline: rewrite → hybrid retrieve → rerank → expand → generate."""
        if not self.is_ready():
            raise RuntimeError("Ask a Delegate is not configured (missing API keys or Supabase)")

        history = self._trim_history(history)
        standalone = self._rewrite_query(history, question) if history else question
        logger.info("Standalone query: %s", standalone)

        # Direct ID lookup from original question (catches "what does 9b1 say?")
        id_hits = self._lookup_by_ids(question)
        if id_hits:
            logger.info("ID lookup found: %s",
                        [(h['regulation_id'], h['kind']) for h in id_hits])

        # Vector retrieval (broad set for reranking)
        vector_hits = self._retrieve_vector(standalone, k=DELEGATE_RERANK_INITIAL_K)

        # Rerank down to the final K
        if vector_hits:
            vector_hits = self._rerank(standalone, vector_hits, top_n=DELEGATE_RETRIEVAL_K)

        # Merge ID hits to the front of the list (they're high-signal)
        merged = self._dedupe(id_hits + vector_hits)

        # Expand with parents + paired regulation/guideline
        expanded = self._expand(merged)

        answer_text = self._generate(history, question, expanded)
        return {
            'answer': answer_text,
            'sources': expanded,
            'standalone_query': standalone,
        }

    # ---------------- Pipeline steps ----------------

    def _trim_history(self, history: list[dict]) -> list[dict]:
        max_msgs = DELEGATE_MAX_HISTORY_TURNS * 2
        return history[-max_msgs:] if len(history) > max_msgs else history

    def _rewrite_query(self, history: list[dict], question: str) -> str:
        try:
            convo = []
            for msg in history:
                role = msg.get('role')
                content = (msg.get('content') or '').strip()
                if role in ('user', 'assistant') and content:
                    convo.append({'role': role, 'content': content})
            convo.append({
                'role': 'user',
                'content': f"Follow-up question: {question}\n\nRewritten standalone query:",
            })

            resp = self.anthropic.messages.create(
                model=REWRITER_MODEL,
                max_tokens=120,
                temperature=0,
                system=_REWRITER_SYSTEM,
                messages=convo,
            )
            return self._first_text(resp).strip() or question
        except Exception as e:
            logger.warning("Query rewrite failed, using raw question: %s", e)
            return question

    def _embed(self, text: str) -> list[float]:
        result = self.voyage.embed(
            [text], model=VOYAGE_EMBED_MODEL, input_type='query'
        )
        return result.embeddings[0]

    def _retrieve_vector(self, query: str, k: int) -> list[dict]:
        embedding = self._embed(query)
        client = get_supabase()
        resp = client.rpc('match_regulations', {
            'query_embedding': embedding,
            'match_count': k,
        }).execute()
        return resp.data or []

    def _rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]:
        if not hits:
            return hits
        docs = [self._format_for_rerank(h) for h in hits]
        try:
            result = self.voyage.rerank(
                query=query,
                documents=docs,
                model=VOYAGE_RERANK_MODEL,
                top_k=top_n,
            )
            return [hits[r.index] for r in result.results]
        except Exception as e:
            logger.warning("Rerank failed, falling back to vector order: %s", e)
            return hits[:top_n]

    def _lookup_by_ids(self, text: str) -> list[dict]:
        ids = set(_REG_ID_RE.findall(text))
        if not ids:
            return []
        client = get_supabase()
        resp = (
            client.table('regulation_chunks')
            .select(_SELECT_COLS)
            .in_('regulation_id', list(ids))
            .execute()
        )
        return resp.data or []

    def _expand(self, hits: list[dict]) -> list[dict]:
        """Add parent + paired regulation/guideline for each hit."""
        if not hits:
            return hits

        seen = {(h['regulation_id'], h['kind']) for h in hits}
        wanted_ids: set[str] = set()

        for h in hits:
            parent = self._parent_id(h['regulation_id'])
            if parent:
                wanted_ids.add(parent)

            # Pair regulation <-> guideline (same numeric ID, different kind)
            if h['kind'] == 'regulation':
                wanted_ids.add(h['regulation_id'] + '+')
            else:
                base = h['regulation_id'].rstrip('+')
                if base:
                    wanted_ids.add(base)

        # Don't refetch things we already have
        wanted_ids -= {rid for (rid, _) in seen}
        if not wanted_ids:
            return hits

        client = get_supabase()
        resp = (
            client.table('regulation_chunks')
            .select(_SELECT_COLS)
            .in_('regulation_id', list(wanted_ids))
            .execute()
        )
        extras = resp.data or []

        out = list(hits)
        for e in extras:
            key = (e['regulation_id'], e['kind'])
            if key not in seen:
                out.append(e)
                seen.add(key)
        return out

    # ---------------- Helpers ----------------

    @staticmethod
    def _first_text(resp) -> str:
        """Text of the first text block; responses may lead with thinking blocks."""
        return next((b.text for b in resp.content if b.type == 'text'), '')

    @staticmethod
    def _parent_id(reg_id: str) -> Optional[str]:
        """Strip the last group of letters or digits to get the parent ID."""
        base = reg_id.rstrip('+')
        if not base:
            return None
        parent = re.sub(r'(?:[a-z]+|\d+)$', '', base)
        return parent if parent and parent != base else None

    @staticmethod
    def _dedupe(chunks: list[dict]) -> list[dict]:
        seen = set()
        out = []
        for c in chunks:
            key = (c['regulation_id'], c['kind'])
            if key not in seen:
                out.append(c)
                seen.add(key)
        return out

    def _format_for_rerank(self, chunk: dict) -> str:
        parts = [f"[{chunk['regulation_id']}] Article {chunk.get('article_num', '?')}: {chunk.get('article_title', '')}"]
        if chunk.get('section_path'):
            parts.append(chunk['section_path'])
        parts.append(chunk['content'])
        return '\n'.join(parts)

    def _format_sources(self, sources: list[dict]) -> str:
        if not sources:
            return "(no sources retrieved)"
        lines = []
        for s in sources:
            kind_tag = 'Guideline' if s.get('kind') == 'guideline' else 'Regulation'
            header = f"[{s['regulation_id']}] {kind_tag} — Article {s.get('article_num', '?')}: {s.get('article_title', '')}"
            if s.get('section_path'):
                header += f" ({s['section_path']})"
            lines.append(f"{header}\n{s['content']}")
        return "\n\n".join(lines)

    def _generate(self, history: list[dict], question: str, sources: list[dict]) -> str:
        system = _SYSTEM_PROMPT_TMPL.format(sources=self._format_sources(sources))
        messages = []
        for msg in history:
            role = msg.get('role')
            content = (msg.get('content') or '').strip()
            if role in ('user', 'assistant') and content:
                messages.append({'role': role, 'content': content})
        messages.append({'role': 'user', 'content': question})

        resp = self.anthropic.messages.create(
            model=self.model,
            # Headroom for adaptive thinking (on by default for current models),
            # which counts against max_tokens alongside the visible answer.
            max_tokens=2048,
            system=system,
            messages=messages,
        )
        return self._first_text(resp).strip()
