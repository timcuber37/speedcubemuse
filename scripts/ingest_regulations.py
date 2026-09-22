"""Ingest WCA Regulations + Guidelines into the Supabase vector store.

One-time / refresh script. Run with:
    python scripts/ingest_regulations.py

Requires VOYAGE_API_KEY and SUPABASE_SERVICE_ROLE_KEY in the environment.
Pins to REGULATIONS_REF (defaults to the `official` branch).
"""
import logging
import os
import re
import sys
import time
from typing import Optional

import requests
import voyageai
from dotenv import load_dotenv

# Make the project root importable so we can reuse services/auth and config.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

load_dotenv()

from config import (REGULATIONS_REF, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_URL,
                    VOYAGE_API_KEY, VOYAGE_EMBED_MODEL)
from services.auth import get_supabase_admin
from services.api_usage import APICall, usage_source

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger(__name__)

REGULATIONS_URL = (
    f"https://raw.githubusercontent.com/thewca/wca-regulations/"
    f"{REGULATIONS_REF}/wca-regulations.md"
)
WCA_REGS_PAGE = "https://www.worldcubeassociation.org/regulations"

# `## <article-9><events><events> Article 9: Events`
ARTICLE_RE = re.compile(r'^##\s+<article-([^>]+)>(?:<[^>]*>)*\s*(.*?)\s*$')
# `    - 9f12a) Some regulation text...`  (indent in groups of 4)
REG_RE = re.compile(r'^(\s*)-\s+([0-9A-Z][a-zA-Z0-9]*\+?)\)\s+(.+)$')

EMBED_BATCH = 32         # ~6K tokens/batch — fits the free-tier 10K TPM cap
UPSERT_BATCH = 100
MAX_EMBED_RETRIES = 6    # ~6m worst-case waiting per stuck batch


def fetch_regulations() -> str:
    log.info("Fetching %s", REGULATIONS_URL)
    resp = requests.get(REGULATIONS_URL, timeout=30)
    resp.raise_for_status()
    return resp.text


def _strip_article_prefix(title: str) -> str:
    """Trim a leading 'Article N: ' from the heading text."""
    return re.sub(r'^Article\s+[0-9A-Z]+\s*:\s*', '', title).strip()


def parse_chunks(text: str) -> list[dict]:
    """Parse the WCA regulations markdown into a list of leaf chunks.

    Each chunk carries:
      - content: the leaf regulation text (for display + citation)
      - section_path: short breadcrumb of parent IDs + truncated text (for display)
      - _parent_context: FULL parent regulation text(s), used only as embedding input
        (stripped before DB upsert)
    """
    chunks: list[dict] = []
    article_num: Optional[str] = None
    article_title: Optional[str] = None
    parents: dict[int, tuple[str, str]] = {}  # depth -> (id, text)
    current: Optional[dict] = None
    current_depth: int = 0

    def commit_current():
        nonlocal current
        if current is not None:
            current['content'] = re.sub(r'\s+', ' ', current['content']).strip()
            chunks.append(current)
            parents[current_depth] = (current['regulation_id'], current['content'])
            current = None

    for raw_line in text.split('\n'):
        line = raw_line.rstrip()

        # Article heading resets all state for the article.
        m = ARTICLE_RE.match(line)
        if m:
            commit_current()
            article_num = m.group(1).strip()
            article_title = _strip_article_prefix(m.group(2))
            parents = {}
            continue

        # New regulation/guideline list item.
        m = REG_RE.match(line)
        if m and article_num is not None:
            commit_current()
            indent = len(m.group(1))
            depth = indent // 4
            reg_id = m.group(2)
            body = m.group(3).strip()
            kind = 'guideline' if reg_id.endswith('+') else 'regulation'

            # Drop any parent slots at depths >= this one.
            parents = {d: p for d, p in parents.items() if d < depth}

            # Short breadcrumb for display
            section_parts = []
            for d in sorted(parents):
                pid, ptext = parents[d]
                trimmed = (ptext[:80] + '…') if len(ptext) > 80 else ptext
                section_parts.append(f"{pid}: {trimmed}")
            section_path = ' / '.join(section_parts) if section_parts else None

            # Full parent text for embedding (not stored in DB)
            parent_context_parts = []
            for d in sorted(parents):
                pid, ptext = parents[d]
                parent_context_parts.append(f"[{pid}] {ptext}")
            parent_context = '\n'.join(parent_context_parts) if parent_context_parts else None

            current = {
                'regulation_id': reg_id,
                'article_num': article_num,
                'article_title': article_title,
                'section_path': section_path,
                'content': body,
                'kind': kind,
                'url': f"{WCA_REGS_PAGE}/#{reg_id}",
                '_parent_context': parent_context,
            }
            current_depth = depth
            continue

        # Possible wrapped continuation of the current regulation.
        if current is not None and line.strip() and not line.lstrip().startswith('#'):
            current['content'] += ' ' + line.strip()
            continue

        if not line.strip():
            commit_current()

    commit_current()
    return chunks


def embed_texts(client: voyageai.Client, texts: list[str]) -> list[list[float]]:
    """Embed a batch with retry-on-rate-limit (handles free-tier 3 RPM / 10K TPM)."""
    for attempt in range(MAX_EMBED_RETRIES):
        try:
            with APICall('regulations.embed', 'voyage', VOYAGE_EMBED_MODEL) as call:
                result = call.capture(client.embed(
                    texts, model=VOYAGE_EMBED_MODEL, input_type='document'))
            return result.embeddings
        except voyageai.error.RateLimitError as e:
            wait = min(25 + attempt * 15, 90)  # 25s, 40s, 55s, ...
            log.warning("Rate limited (attempt %d/%d) — sleeping %ds. (%s)",
                        attempt + 1, MAX_EMBED_RETRIES, wait, str(e).split('.')[0])
            time.sleep(wait)
    raise RuntimeError(
        "Voyage rate limit exhausted after retries. "
        "Verify the API key's organization has a payment method in dashboard.voyageai.com."
    )


def build_embedding_input(chunk: dict) -> str:
    """Compose the text fed to the embedder: id + article + full parent chain + body.

    Full parent text gives the embedding model the hierarchical context that a
    leaf regulation often relies on (e.g., "9b1" is meaningless without "9b").
    """
    parts = [f"[{chunk['regulation_id']}] Article {chunk['article_num']}: {chunk['article_title']}"]
    if chunk.get('_parent_context'):
        parts.append("Context:\n" + chunk['_parent_context'])
    kind_tag = 'Guideline' if chunk['kind'] == 'guideline' else 'Regulation'
    parts.append(f"{kind_tag}: {chunk['content']}")
    return '\n'.join(parts)


def _strip_private(chunk: dict) -> dict:
    """Drop underscore-prefixed fields before sending to Supabase."""
    return {k: v for k, v in chunk.items() if not k.startswith('_')}


def upsert_chunks(chunks: list[dict]) -> None:
    if not VOYAGE_API_KEY:
        sys.exit("VOYAGE_API_KEY is not set")
    if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
        sys.exit("Supabase admin credentials not configured")

    admin = get_supabase_admin()
    if admin is None:
        sys.exit("Could not initialize Supabase admin client")

    voyage = voyageai.Client(api_key=VOYAGE_API_KEY)

    log.info("Truncating regulation_chunks (full reload)")
    # Delete everything; bulk re-embedding is cheap at this scale.
    admin.table('regulation_chunks').delete().neq('id', 0).execute()

    enriched: list[dict] = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i:i + EMBED_BATCH]
        inputs = [build_embedding_input(c) for c in batch]
        log.info("Embedding %d–%d / %d", i + 1, i + len(batch), len(chunks))
        embeddings = embed_texts(voyage, inputs)
        for chunk, emb in zip(batch, embeddings):
            enriched.append({**_strip_private(chunk), 'embedding': emb})
        time.sleep(0.1)

    for i in range(0, len(enriched), UPSERT_BATCH):
        batch = enriched[i:i + UPSERT_BATCH]
        log.info("Upserting %d–%d / %d", i + 1, i + len(batch), len(enriched))
        admin.table('regulation_chunks').upsert(
            batch, on_conflict='regulation_id,kind'
        ).execute()


@usage_source("maintenance")
def main():
    text = fetch_regulations()
    chunks = parse_chunks(text)
    n_reg = sum(1 for c in chunks if c['kind'] == 'regulation')
    n_guide = sum(1 for c in chunks if c['kind'] == 'guideline')
    log.info("Parsed %d chunks (%d regulations, %d guidelines)",
             len(chunks), n_reg, n_guide)

    if not chunks:
        sys.exit("No chunks parsed — check the markdown format / regex")

    upsert_chunks(chunks)
    log.info("Done.")


if __name__ == '__main__':
    main()
