"""Tests for the retrieval agent (LLM client and source tools mocked)."""
from __future__ import annotations

from unittest.mock import AsyncMock

from agents import retrieval_agent as retrieval_module
from agents.retrieval_agent import RetrievalAgent
from models import Evidence, SubQuestion


def _agent(sources: list[str], query: str = "q") -> RetrievalAgent:
    """Build a RetrievalAgent whose mocked LLM picks the given sources/query."""
    llm = AsyncMock()
    llm.call_structured = AsyncMock(return_value={"sources": sources, "query": query})
    return RetrievalAgent(llm)


async def test_queries_only_selected_sources(monkeypatch) -> None:
    """Only the model-selected sources are queried, and evidence is tagged."""
    called: list[str] = []

    async def fake_wikipedia(query: str):
        """Stub Wikipedia returning one Evidence item."""
        called.append("wikipedia")
        return Evidence("Wikipedia — X", "https://w/x", "content")

    async def fake_web(query: str, max_results: int):
        """Stub web search; should not be called in this test."""
        called.append("web_search")
        return [Evidence("Web — Y", "https://y", "content")]

    monkeypatch.setattr(retrieval_module.wikipedia, "search", fake_wikipedia)
    monkeypatch.setattr(retrieval_module.web_search, "search", fake_web)

    sub_question = SubQuestion(id="sq1", text="What is X?")
    evidence = await _agent(["wikipedia"]).run(sub_question)

    assert called == ["wikipedia"]
    assert all(e.sub_question_id == "sq1" for e in evidence)
    assert sub_question.status == "answered"


async def test_empty_source_selection_falls_back_to_all(monkeypatch) -> None:
    """If the model returns no valid sources, all sources are queried."""
    called: list[str] = []

    async def fake_list(query: str, max_results: int):
        """Stub a list-returning source tool."""
        called.append("list")
        return []

    async def fake_wiki(query: str):
        """Stub Wikipedia returning nothing."""
        called.append("wiki")
        return None

    monkeypatch.setattr(retrieval_module.web_search, "search", fake_list)
    monkeypatch.setattr(retrieval_module.arxiv, "search", fake_list)
    monkeypatch.setattr(retrieval_module.wikipedia, "search", fake_wiki)

    evidence = await _agent([]).run(SubQuestion(id="sq1", text="Q?"))

    assert len(called) == 3  # all three sources were queried
    assert evidence == []
