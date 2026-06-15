"""Tests for the synthesis agent (LLM client fully mocked)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agents.synthesis_agent import SynthesisAgent
from models import Evidence


def _evidence(count: int) -> list[Evidence]:
    """Create ``count`` simple Evidence objects."""
    return [
        Evidence(
            source_name=f"Source {i}",
            source_url=f"https://example.com/{i}",
            content=f"Fact {i}",
            sub_question_id="sq1",
        )
        for i in range(1, count + 1)
    ]


def _make_agent(claims: list[dict]) -> SynthesisAgent:
    """Build a SynthesisAgent whose mocked LLM returns the given claims."""
    llm = AsyncMock()
    llm.call_structured = AsyncMock(return_value={"claims": claims})
    return SynthesisAgent(llm)


async def test_conflicting_evidence_lowers_confidence() -> None:
    """A claim with conflicting evidence scores lower than one without."""
    agent = _make_agent(
        [
            {"text": "Unconflicted", "confidence": 0.8, "supporting_ids": [1], "conflicting_ids": []},
            {"text": "Conflicted", "confidence": 0.8, "supporting_ids": [2], "conflicting_ids": [3]},
        ]
    )
    report = await agent.run("Q?", _evidence(3))
    unconflicted = next(c for c in report.claims if c.text == "Unconflicted")
    conflicted = next(c for c in report.claims if c.text == "Conflicted")
    assert conflicted.confidence < unconflicted.confidence


async def test_claim_without_supporting_evidence_is_never_produced() -> None:
    """Claims with no supporting evidence are dropped from the report."""
    agent = _make_agent(
        [
            {"text": "Grounded", "confidence": 0.7, "supporting_ids": [1], "conflicting_ids": []},
            {"text": "Unsupported", "confidence": 0.9, "supporting_ids": [], "conflicting_ids": []},
        ]
    )
    report = await agent.run("Q?", _evidence(2))
    texts = [c.text for c in report.claims]
    assert "Grounded" in texts
    assert "Unsupported" not in texts


async def test_overall_confidence_is_average_of_claims() -> None:
    """Overall confidence equals the mean of the individual claim confidences."""
    agent = _make_agent(
        [
            {"text": "A", "confidence": 0.6, "supporting_ids": [1], "conflicting_ids": []},
            {"text": "B", "confidence": 0.4, "supporting_ids": [2], "conflicting_ids": []},
        ]
    )
    report = await agent.run("Q?", _evidence(2))
    expected = sum(c.confidence for c in report.claims) / len(report.claims)
    assert report.overall_confidence == pytest.approx(expected, abs=1e-3)


async def test_no_evidence_returns_empty_report() -> None:
    """With no evidence, an empty zero-confidence report is returned."""
    agent = _make_agent([])
    report = await agent.run("Q?", [])
    assert report.claims == []
    assert report.overall_confidence == 0.0
