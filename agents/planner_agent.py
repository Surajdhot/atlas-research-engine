"""Planner agent: decomposes a research question into focused sub-questions."""
from __future__ import annotations

import logging

import config
from agents.base_agent import BaseAgent
from llm_client import LLMClient
from models import SubQuestion

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_questions": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["sub_questions"],
    "additionalProperties": False,
}


class PlannerAgent(BaseAgent):
    """Breaks the research question into 2-5 non-overlapping sub-questions."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create the planner with its prompt file."""
        super().__init__(llm_client, "planner", "planner.txt")

    async def run(self, question: str) -> list[SubQuestion]:
        """Decompose the question into a list of SubQuestion objects."""
        system = self.load_prompt()
        user = (
            f"Research question: {question}\n\n"
            f"Return between {config.MIN_SUB_QUESTIONS} and {config.MAX_SUB_QUESTIONS} "
            "focused, non-overlapping sub-questions."
        )
        data = await self._llm.call_structured(system, user, _SCHEMA)
        return self._build_sub_questions(data.get("sub_questions", []))

    def _build_sub_questions(self, raw: list[str]) -> list[SubQuestion]:
        """Clean, de-duplicate, and cap the model's sub-question list."""
        seen: set[str] = set()
        sub_questions: list[SubQuestion] = []
        for text in raw:
            cleaned = (text or "").strip()
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            sub_questions.append(
                SubQuestion(id=f"sq{len(sub_questions) + 1}", text=cleaned)
            )
            if len(sub_questions) >= config.MAX_SUB_QUESTIONS:
                break
        logger.info("Planner produced %d sub-questions.", len(sub_questions))
        return sub_questions
