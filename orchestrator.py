"""Orchestrator: coordinates planner, parallel retrieval, and synthesis.

The retrieval stage is the technical centrepiece: one retrieval agent per
sub-question is launched and they run *concurrently* via ``asyncio.gather`` —
not one after another.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from agents.planner_agent import PlannerAgent
from agents.retrieval_agent import RetrievalAgent
from agents.synthesis_agent import SynthesisAgent
from llm_client import LLMClient
from models import Evidence, Report, SubQuestion

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]
RetrievalFactory = Callable[[], RetrievalAgent]


class Orchestrator:
    """Runs the end-to-end research flow, fanning retrieval out concurrently."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        planner: Optional[PlannerAgent] = None,
        synthesizer: Optional[SynthesisAgent] = None,
        retrieval_factory: Optional[RetrievalFactory] = None,
    ) -> None:
        """Wire up the agents, allowing injection of mocks for testing."""
        self._llm = llm_client or LLMClient()
        self._planner = planner or PlannerAgent(self._llm)
        self._synthesizer = synthesizer or SynthesisAgent(self._llm)
        self._retrieval_factory = retrieval_factory or self._default_retrieval_agent

    def _default_retrieval_agent(self) -> RetrievalAgent:
        """Create a fresh retrieval agent sharing the orchestrator's LLM client."""
        return RetrievalAgent(self._llm)

    async def research(
        self, question: str, on_progress: Optional[ProgressCallback] = None
    ) -> Report:
        """Plan, retrieve in parallel, and synthesise a report for a question."""
        notify = on_progress or (lambda _message: None)

        notify("Planning sub-questions…")
        sub_questions = await self._planner.run(question)
        notify(f"Planned {len(sub_questions)} sub-question(s).")

        notify("Retrieving evidence from all sources in parallel…")
        evidence = await self._gather_evidence(sub_questions)
        notify(f"Gathered {len(evidence)} piece(s) of evidence.")

        notify("Synthesising the report and resolving conflicts…")
        report = await self._synthesizer.run(question, evidence)
        notify("Research complete.")
        return report

    async def _gather_evidence(
        self, sub_questions: list[SubQuestion]
    ) -> list[Evidence]:
        """Run one retrieval agent per sub-question concurrently with gather."""
        if not sub_questions:
            return []
        tasks = [
            self._retrieval_factory().run(sub_question)
            for sub_question in sub_questions
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]
