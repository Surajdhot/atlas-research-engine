"""Wikipedia source backed by the public REST API (no API key required)."""
from __future__ import annotations

import logging
from typing import Optional

import httpx

import config
from models import Evidence

logger = logging.getLogger(__name__)

SEARCH_URL = "https://en.wikipedia.org/w/rest.php/v1/search/page"
SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
USER_AGENT = "AtlasResearchEngine/1.0 (research demo)"


async def search(query: str) -> Optional[Evidence]:
    """Return the top matching Wikipedia article summary as Evidence, or None."""
    try:
        async with httpx.AsyncClient(
            timeout=config.HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT}
        ) as client:
            title = await _top_title(client, query)
            if title is None:
                return None
            return await _summary(client, title)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Wikipedia lookup failed for %r: %s", query, exc)
        return None


async def _top_title(client: httpx.AsyncClient, query: str) -> Optional[str]:
    """Find the best-matching article title for a query."""
    response = await client.get(SEARCH_URL, params={"q": query, "limit": 1})
    response.raise_for_status()
    pages = response.json().get("pages", [])
    return pages[0].get("title") if pages else None


async def _summary(client: httpx.AsyncClient, title: str) -> Optional[Evidence]:
    """Fetch the REST summary for an article title and build Evidence."""
    response = await client.get(SUMMARY_URL + title.replace(" ", "_"))
    response.raise_for_status()
    data = response.json()
    extract = data.get("extract")
    if not extract:
        return None
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    return Evidence(
        source_name=f"Wikipedia — {data.get('title', title)}",
        source_url=page_url,
        content=extract,
    )
