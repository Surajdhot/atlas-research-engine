"""Tests for the planner agent (LLM client fully mocked)."""
from __future__ import annotations

from unittest.mock import AsyncMock

from agents.planner_agent import PlannerAgent


def _make_planner(sub_questions: list[str]) -> PlannerAgent:
    """Build a PlannerAgent whose mocked LLM returns the given sub-questions."""
    llm = AsyncMock()
    llm.call_structured = AsyncMock(return_value={"sub_questions": sub_questions})
    return PlannerAgent(llm)


async def test_returns_between_two_and_five_sub_questions() -> None:
    """The planner caps output at five even when the model returns more."""
    planner = _make_planner([f"Sub-question {i}?" for i in range(8)])
    result = await planner.run("A broad research question?")
    assert 2 <= len(result) <= 5


async def test_sub_questions_are_non_empty() -> None:
    """Blank model outputs are filtered out, leaving only real sub-questions."""
    planner = _make_planner(["  ", "What is X?", "", "How does Y work?"])
    result = await planner.run("Question?")
    assert len(result) == 2
    assert all(sub_question.text for sub_question in result)


async def test_duplicate_sub_questions_are_removed() -> None:
    """Overlapping (duplicate) sub-questions are de-duplicated."""
    planner = _make_planner(["What is X?", "what is x?", "How does Y work?"])
    result = await planner.run("Question?")
    assert len(result) == 2
