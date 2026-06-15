"""Streamlit UI for the Atlas multi-agent research engine."""
from __future__ import annotations

import asyncio

import streamlit as st

import config
from models import Claim, Report
from orchestrator import Orchestrator


def _confidence_color(confidence: float) -> str:
    """Return a Streamlit colour name for a confidence score."""
    if confidence > config.CONFIDENCE_GREEN:
        return "green"
    if confidence >= config.CONFIDENCE_AMBER:
        return "orange"
    return "red"


def _render_claim(index: int, claim: Claim) -> None:
    """Render a single claim with a colour-coded confidence bar and citations."""
    st.markdown(f"**Claim {index}.** {claim.text}")
    color = _confidence_color(claim.confidence)
    st.markdown(f":{color}[Confidence: {claim.confidence:.0%}]")
    st.progress(min(max(claim.confidence, 0.0), 1.0))
    label = (
        f"Evidence — {len(claim.supporting_evidence)} supporting, "
        f"{len(claim.conflicting_evidence)} conflicting"
    )
    with st.expander(label):
        for ev in claim.supporting_evidence:
            st.markdown(f"- ✅ [{ev.source_name}]({ev.source_url})")
        for ev in claim.conflicting_evidence:
            st.markdown(f"- ⚠️ Conflicts: [{ev.source_name}]({ev.source_url})")


def _render_report(report: Report) -> None:
    """Render the full report: overall confidence, claims, and sources."""
    color = _confidence_color(report.overall_confidence)
    st.subheader("Overall confidence")
    st.markdown(f":{color}[{report.overall_confidence:.0%}]")
    st.progress(min(max(report.overall_confidence, 0.0), 1.0))

    st.subheader("Claims")
    if not report.claims:
        st.info("No well-supported claims could be produced for this question.")
    for index, claim in enumerate(report.claims, start=1):
        _render_claim(index, claim)

    _render_sources(report)


def _render_sources(report: Report) -> None:
    """Render the de-duplicated list of all sources used in the report."""
    st.subheader("Sources")
    seen: set[str] = set()
    for ev in report.sources:
        key = ev.source_url or ev.source_name
        if key in seen:
            continue
        seen.add(key)
        st.markdown(f"- [{ev.source_name}]({ev.source_url})")


def _run_research(question: str) -> Report:
    """Run the orchestrator for a question, streaming progress to the UI."""
    status = st.status("Starting research…", expanded=True)

    def on_progress(message: str) -> None:
        """Forward an orchestrator progress message to the status panel."""
        status.update(label=message)
        status.write(message)

    report = asyncio.run(Orchestrator().research(question, on_progress))
    status.update(label="Research complete.", state="complete")
    return report


def main() -> None:
    """Render the Atlas Streamlit application."""
    st.set_page_config(page_title="Atlas Research Engine", page_icon="🧭")
    st.title("🧭 Atlas — Multi-Agent Research Engine")
    st.caption(
        "Ask a research question. Atlas plans sub-questions, gathers evidence "
        "from the web, Wikipedia, and arXiv in parallel, then synthesises a "
        "cited report with confidence scores."
    )

    question = st.text_input(
        "Research question",
        placeholder="e.g. What are the leading approaches to grid-scale energy storage?",
    )
    if not st.button("Research", type="primary"):
        return
    if not question.strip():
        st.warning("Please enter a research question.")
        return
    try:
        config.validate_config()
    except config.ConfigError as exc:
        st.error(str(exc))
        return
    _render_report(_run_research(question.strip()))


if __name__ == "__main__":
    main()
