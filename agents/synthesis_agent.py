"""Synthesis agent: turns gathered evidence into a cited, scored report."""
from __future__ import annotations

import logging

import config
from agents.base_agent import BaseAgent
from llm_client import LLMClient
from models import Claim, Evidence, Report

logger = logging.getLogger(__name__)

_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                    "supporting_ids": {"type": "array", "items": {"type": "integer"}},
                    "conflicting_ids": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["text", "confidence", "supporting_ids", "conflicting_ids"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}


class SynthesisAgent(BaseAgent):
    """Combines evidence into claims, flags conflicts, and scores confidence."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Create the synthesis agent with its prompt file."""
        super().__init__(llm_client, "synthesis", "synthesis.txt")

    async def run(self, question: str, evidence: list[Evidence]) -> Report:
        """Produce the final Report from the question and all gathered evidence."""
        if not evidence:
            logger.warning("No evidence gathered; returning an empty report.")
            return Report(question=question, overall_confidence=0.0)
        by_id = {i + 1: ev for i, ev in enumerate(evidence)}
        data = await self._llm.call_structured(
            self.load_prompt(), self._format_evidence(question, by_id), _SCHEMA
        )
        claims = self._build_claims(data.get("claims", []), by_id)
        overall = self._average_confidence(claims)
        logger.info("Synthesis produced %d claim(s); overall %.2f.", len(claims), overall)
        return Report(
            question=question,
            claims=claims,
            sources=evidence,
            overall_confidence=overall,
        )

    def _format_evidence(self, question: str, by_id: dict[int, Evidence]) -> str:
        """Render the question and numbered evidence for the model prompt."""
        lines = [f"Research question: {question}", "", "Evidence:"]
        for idx, ev in by_id.items():
            lines.append(f"[{idx}] ({ev.source_name}) {ev.content[:500]}")
        lines.append(
            "\nWrite claims that together answer the question. Cite supporting "
            "evidence by its number, and mark any conflicting evidence explicitly."
        )
        return "\n".join(lines)

    def _build_claims(
        self, raw: list[dict], by_id: dict[int, Evidence]
    ) -> list[Claim]:
        """Map raw claim dicts to Claim objects, dropping unsupported claims."""
        claims: list[Claim] = []
        for item in raw:
            support = [by_id[i] for i in item.get("supporting_ids", []) if i in by_id]
            if not support:
                continue  # never produce a claim with no supporting evidence
            conflict = [by_id[i] for i in item.get("conflicting_ids", []) if i in by_id]
            claims.append(
                Claim(
                    text=(item.get("text") or "").strip(),
                    confidence=self._score(item.get("confidence", 0.0), conflict),
                    supporting_evidence=support,
                    conflicting_evidence=conflict,
                )
            )
        return claims

    @staticmethod
    def _score(base: float, conflict: list[Evidence]) -> float:
        """Clamp base confidence to [0, 1] and penalise conflicting sources."""
        try:
            value = float(base)
        except (TypeError, ValueError):
            value = 0.0
        value = max(0.0, min(1.0, value))
        if conflict:
            value *= 1.0 - config.CONFLICT_PENALTY
        return round(value, 3)

    @staticmethod
    def _average_confidence(claims: list[Claim]) -> float:
        """Return the mean confidence across claims (0.0 when there are none)."""
        if not claims:
            return 0.0
        return round(sum(c.confidence for c in claims) / len(claims), 3)
