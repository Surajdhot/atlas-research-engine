"""Web search source backed by the Tavily API."""
from __future__ import annotations

import logging

import httpx

import config
from models import Evidence

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


async def search(query: str, max_results: int) -> list[Evidence]:
    """Search the web via Tavily and return up to ``max_results`` Evidence items.

    Returns an empty list (and logs a warning) on any API or network failure.
    """
    if not config.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY is not set; skipping web search.")
        return []
    payload = {
        "api_key": config.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
    }
    try:
        async with httpx.AsyncClient(timeout=config.HTTP_TIMEOUT) as client:
            response = await client.post(TAVILY_URL, json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Web search failed for %r: %s", query, exc)
        return []
    return _to_evidence(data.get("results", []))


def _to_evidence(results: list[dict]) -> list[Evidence]:
    """Map raw Tavily result dicts into Evidence objects."""
    evidence: list[Evidence] = []
    for item in results:
        content = item.get("content") or item.get("title") or ""
        if not content:
            continue
        title = item.get("title", "Web result")
        evidence.append(
            Evidence(
                source_name=f"Web — {title}",
                source_url=item.get("url", ""),
                content=content,
            )
        )
    return evidence
