"""Retrieval agent: gathers evidence for one sub-question via dynamic tool use."""
from __future__ import annotations

import logging

import config
from agents.base_agent import BaseAgent
from llm_client import LLMClient, ToolSpec
from models import Evidence, SubQuestion
from tools import arxiv, web_search, wikipedia

logger = logging.getLogger(__name__)

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "The search query"}},
    "required": ["query"],
}


class RetrievalAgent(BaseAgent):
    """Lets Claude choose web/Wikipedia/arXiv tools to gather evidence."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create the retrieval agent with its prompt file."""
        super().__init__(llm_client, "retrieval", "retrieval.txt")

    async def run(self, sub_question: SubQuestion) -> list[Evidence]:
        """Gather and return evidence for a single sub-question."""
        collected: list[Evidence] = []
        tools = self._build_tools(sub_question, collected)
        system = self.load_prompt()
        user = (
            f"Sub-question: {sub_question.text}\n\n"
            "Choose the most appropriate sources, gather evidence, then briefly "
            "summarise what you found."
        )
        await self._llm.call_with_tools(system, user, tools)
        sub_question.status = "answered" if collected else "no_evidence"
        logger.info(
            "Retrieval for %s gathered %d evidence item(s).",
            sub_question.id,
            len(collected),
        )
        return collected[: config.MAX_EVIDENCE_PER_QUESTION]

    def _build_tools(
        self, sub_question: SubQuestion, collected: list[Evidence]
    ) -> list[ToolSpec]:
        """Build the web/Wikipedia/arXiv ToolSpecs bound to this sub-question."""

        async def run_web(tool_input: dict) -> str:
            """Execute a Tavily web search and record the evidence."""
            found = await web_search.search(
                tool_input.get("query", sub_question.text),
                config.MAX_EVIDENCE_PER_QUESTION,
            )
            return _record(found, sub_question.id, collected)

        async def run_wikipedia(tool_input: dict) -> str:
            """Execute a Wikipedia lookup and record the evidence."""
            item = await wikipedia.search(tool_input.get("query", sub_question.text))
            return _record([item] if item else [], sub_question.id, collected)

        async def run_arxiv(tool_input: dict) -> str:
            """Execute an arXiv search and record the evidence."""
            found = await arxiv.search(
                tool_input.get("query", sub_question.text),
                config.MAX_EVIDENCE_PER_QUESTION,
            )
            return _record(found, sub_question.id, collected)

        return [
            ToolSpec(
                "web_search",
                "Search the web for current information, news, and recent developments.",
                _QUERY_SCHEMA,
                run_web,
            ),
            ToolSpec(
                "wikipedia",
                "Look up general knowledge, definitions, and background on Wikipedia.",
                _QUERY_SCHEMA,
                run_wikipedia,
            ),
            ToolSpec(
                "arxiv",
                "Search arXiv for academic and scientific papers and abstracts.",
                _QUERY_SCHEMA,
                run_arxiv,
            ),
        ]


def _record(
    found: list[Evidence], sub_question_id: str, collected: list[Evidence]
) -> str:
    """Tag evidence with its sub-question, store it, and summarise for the model."""
    for item in found:
        item.sub_question_id = sub_question_id
        collected.append(item)
    if not found:
        return "No results found."
    lines = [f"- {item.source_name}: {item.content[:300]}" for item in found]
    return f"Found {len(found)} result(s):\n" + "\n".join(lines)
