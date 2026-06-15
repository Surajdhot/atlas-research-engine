"""Domain dataclasses shared across the Atlas research engine."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SubQuestion:
    """A single focused sub-question derived from the research question."""

    id: str
    text: str
    status: str = "pending"


@dataclass
class Evidence:
    """A single piece of evidence gathered from one source for a sub-question."""

    source_name: str
    source_url: str
    content: str
    sub_question_id: str = ""


@dataclass
class Claim:
    """A synthesised claim with confidence and its supporting/conflicting evidence."""

    text: str
    confidence: float
    supporting_evidence: list[Evidence] = field(default_factory=list)
    conflicting_evidence: list[Evidence] = field(default_factory=list)


@dataclass
class Report:
    """The final cited research report with per-claim and overall confidence."""

    question: str
    claims: list[Claim] = field(default_factory=list)
    sources: list[Evidence] = field(default_factory=list)
    overall_confidence: float = 0.0
