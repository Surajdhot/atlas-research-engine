"""Tests for the orchestrator's end-to-end flow and parallel retrieval."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from models import Evidence, Report, SubQuestion
from orchestrator import Orchestrator


def _planner(sub_questions: list[SubQuestion]) -> AsyncMock:
    """Build a mock planner returning the given sub-questions."""
    planner = AsyncMock()
    planner.run = AsyncMock(return_value=sub_questions)
    return planner


def _synthesizer(report: Report) -> AsyncMock:
    """Build a mock synthesizer returning the given report."""
    synthesizer = AsyncMock()
    synthesizer.run = AsyncMock(return_value=report)
    return synthesizer


async def test_full_flow_runs_end_to_end() -> None:
    """Planner -> retrieval -> synthesis returns the synthesizer's report."""
    sub_questions = [SubQuestion(id="sq1", text="A?"), SubQuestion(id="sq2", text="B?")]
    expected = Report(question="Q?", overall_confidence=0.5)

    def factory() -> AsyncMock:
        """Return a retrieval agent that yields one evidence item."""
        agent = AsyncMock()
        agent.run = AsyncMock(return_value=[Evidence("S", "u", "c", "sq1")])
        return agent

    orchestrator = Orchestrator(
        llm_client=MagicMock(),
        planner=_planner(sub_questions),
        synthesizer=_synthesizer(expected),
        retrieval_factory=factory,
    )
    report = await orchestrator.research("Q?")
    assert report is expected


async def test_retrieval_runs_in_parallel() -> None:
    """All retrieval agents run concurrently, not one after another."""
    sub_questions = [SubQuestion(id=f"sq{i}", text=f"{i}?") for i in range(3)]
    running = 0
    peak_concurrency = 0

    async def slow_run(_sub_question: SubQuestion) -> list[Evidence]:
        """Track concurrent executions while simulating slow retrieval."""
        nonlocal running, peak_concurrency
        running += 1
        peak_concurrency = max(peak_concurrency, running)
        await asyncio.sleep(0.05)
        running -= 1
        return []

    def factory() -> AsyncMock:
        """Return a retrieval agent whose run tracks concurrency."""
        agent = AsyncMock()
        agent.run = AsyncMock(side_effect=slow_run)
        return agent

    orchestrator = Orchestrator(
        llm_client=MagicMock(),
        planner=_planner(sub_questions),
        synthesizer=_synthesizer(Report(question="Q?")),
        retrieval_factory=factory,
    )
    await orchestrator.research("Q?")
    # If retrieval were sequential, peak concurrency would be 1.
    assert peak_concurrency == len(sub_questions)
