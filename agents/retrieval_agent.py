"""Retrieval agent: gathers evidence for one sub-question from model-chosen sources.

The model selects which sources fit the sub-question (and the search query) via
structured JSON output; the chosen source tools are then queried concurrently.
This keeps source selection model-driven while staying reliable on free/open
models that handle native function-calling poorly.
"""
from __future__ import annotations

import asyncio
import logging

import config
from agents.base_agent import BaseAgent
from llm_client import LLMClient
from models import Evidence, SubQuestion
from tools import arxiv, web_search, wikipedia

logger = logging.getLogger(__name__)

_VALID_SOURCES = ("web_search", "wikipedia", "arxiv")

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {"type": "string", "enum": list(_VALID_SOURCES)},
        },
        "query": {"type": "string"},
    },
    "required": ["sources", "query"],
    "additionalProperties": False,
}


class RetrievalAgent(BaseAgent):
    """Asks the model which sources fit a sub-question, then gathers evidence."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create the retrieval agent with its prompt file."""
        super().__init__(llm_client, "retrieval", "retrieval.txt")

    async def run(self, sub_question: SubQuestion) -> list[Evidence]:
        """Gather and return evidence for a single sub-question."""
        sources, query = await self._choose_sources(sub_question)
        evidence = await self._gather(sources, query, sub_question.id)
        sub_question.status = "answered" if evidence else "no_evidence"
        logger.info(
            "Retrieval for %s used %s; gathered %d evidence item(s).",
            sub_question.id,
            sources,
            len(evidence),
        )
        return evidence[: config.MAX_EVIDENCE_PER_QUESTION]

    async def _choose_sources(
        self, sub_question: SubQuestion
    ) -> tuple[list[str], str]:
        """Use the model to pick which sources to query and a refined query."""
        user = (
            f"Sub-question: {sub_question.text}\n\n"
            "Choose which sources to use and the single best search query."
        )
        data = await self._llm.call_structured(self.load_prompt(), user, _PLAN_SCHEMA)
        sources = [s for s in (data.get("sources") or []) if s in _VALID_SOURCES]
        if not sources:
            sources = list(_VALID_SOURCES)  # fall back to all sources
        query = (data.get("query") or sub_question.text).strip()
        return sources, query

    async def _gather(
        self, sources: list[str], query: str, sub_question_id: str
    ) -> list[Evidence]:
        """Query all chosen sources concurrently and merge the tagged evidence."""
        results = await asyncio.gather(
            *(self._query_source(source, query) for source in sources)
        )
        evidence: list[Evidence] = []
        for items in results:
            for item in items:
                item.sub_question_id = sub_question_id
                evidence.append(item)
        return evidence

    async def _query_source(self, source: str, query: str) -> list[Evidence]:
        """Dispatch a single source name to its tool module."""
        if source == "web_search":
            return await web_search.search(query, config.MAX_EVIDENCE_PER_QUESTION)
        if source == "arxiv":
            return await arxiv.search(query, config.MAX_EVIDENCE_PER_QUESTION)
        if source == "wikipedia":
            item = await wikipedia.search(query)
            return [item] if item else []
        return []
