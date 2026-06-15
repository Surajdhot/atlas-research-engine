"""arXiv source backed by the public Atom API, parsed with feedparser."""
from __future__ import annotations

import logging

import feedparser
import httpx

import config
from models import Evidence

logger = logging.getLogger(__name__)

ARXIV_URL = "http://export.arxiv.org/api/query"


async def search(query: str, max_results: int) -> list[Evidence]:
    """Search arXiv and return up to ``max_results`` paper abstracts as Evidence.

    Returns an empty list (and logs a warning) on any API or parse failure.
    """
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            response = await client.get(ARXIV_URL, params=params)
            response.raise_for_status()
            feed = feedparser.parse(response.text)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("arXiv search failed for %r: %s", query, exc)
        return []
    return _to_evidence(feed.entries)


def _to_evidence(entries: list) -> list[Evidence]:
    """Map parsed Atom entries into Evidence objects."""
    evidence: list[Evidence] = []
    for entry in entries:
        summary = getattr(entry, "summary", "").strip()
        if not summary:
            continue
        title = getattr(entry, "title", "arXiv paper").strip()
        evidence.append(
            Evidence(
                source_name=f"arXiv — {title}",
                source_url=getattr(entry, "id", ""),
                content=summary,
            )
        )
    return evidence
