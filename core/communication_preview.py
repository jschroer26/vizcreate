"""Phase 5A: build grounded communication previews from completed analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommunicationItem:
    question: str
    scope_text: str
    chart_type: str
    insights: str
    evidence_overview: str
    evidence_findings: list[str] = field(default_factory=list)
    translation: str = ""
    translation_sections: dict[str, str] = field(default_factory=dict)
    image_png: bytes | None = None


@dataclass
class CommunicationPreview:
    scope: str
    title: str
    introduction: str
    items: list[CommunicationItem] = field(default_factory=list)
    overall_translation: str = ""
    limitations: list[str] = field(default_factory=list)


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def common_language_translation(
    *,
    question: str,
    insights: str,
    evidence_overview: str,
    evidence_findings: list[str] | None = None,
    scope_text: str = "",
) -> str:
    """Translate analytical output into accessible, non-technical language.

    The translation reorganizes evidence already produced by VizCreate. It does
    not add causes, prescriptions, or claims beyond the completed analysis.
    """
    findings = [_clean(item) for item in (evidence_findings or []) if _clean(item)]
    overview = _clean(evidence_overview)
    insight_text = _clean(insights)
    question_text = _clean(question)

    if findings:
        lead = findings[0]
        supporting = findings[1] if len(findings) > 1 else ""
    elif overview:
        lead = overview
        supporting = ""
    elif insight_text:
        lead = insight_text
        supporting = ""
    else:
        return (
            "This analysis did not identify a sufficiently clear pattern to summarize "
            "in everyday language. The visualization should be reviewed alongside the "
            "original data and local context."
        )

    parts = []
    if question_text:
        parts.append(f"For the question “{question_text},” the clearest result is this: {lead}")
    else:
        parts.append(f"The clearest result is this: {lead}")

    if supporting and supporting.lower() not in lead.lower():
        parts.append(f"Another useful detail is that {supporting[0].lower() + supporting[1:]}")

    if scope_text:
        parts.append(f"This statement applies only to the selected data scope: {scope_text}.")

    parts.append(
        "This describes a pattern in the available data. It does not by itself explain "
        "why the pattern occurred or determine what action should be taken."
    )
    return " ".join(parts)


def current_preview(
    *,
    question: str,
    scope_text: str,
    chart_type: str,
    insights: str,
    evidence_overview: str,
    evidence_findings: list[str],
    image_png: bytes | None,
) -> CommunicationPreview:
    item = CommunicationItem(
        question=_clean(question),
        scope_text=_clean(scope_text),
        chart_type=_clean(chart_type) or "visualization",
        insights=_clean(insights),
        evidence_overview=_clean(evidence_overview),
        evidence_findings=[_clean(item) for item in evidence_findings if _clean(item)],
        translation=common_language_translation(
            question=question,
            insights=insights,
            evidence_overview=evidence_overview,
            evidence_findings=evidence_findings,
            scope_text=scope_text,
        ),
        image_png=image_png,
    )
    return CommunicationPreview(
        scope="current",
        title="Current Visualization Communication Preview",
        introduction=(
            "This preview organizes the current question, visualization, evidence, "
            "and a common-language explanation for reuse in a presentation, memo, "
            "discussion, or locally developed report."
        ),
        items=[item],
        limitations=[
            "The preview summarizes the selected dataset and analytical scope.",
            "The findings are descriptive and do not establish causation.",
        ],
    )


def entire_preview(
    *,
    state: Any,
    overall_summary: Any | None = None,
) -> CommunicationPreview:
    completed = [
        step for step in getattr(state, "history", [])
        if getattr(step, "step_type", "completed_analysis") != "transition"
    ]

    items = []
    for step in completed:
        findings = list(getattr(step, "evidence_findings", []) or [])
        stored_translation = dict(getattr(step, "common_translation", {}) or {})
        translation_text = _clean(stored_translation.get("combined_text"))
        if not translation_text:
            translation_text = common_language_translation(
                question=getattr(step, "prompt", ""),
                insights=getattr(step, "insights", ""),
                evidence_overview=getattr(step, "evidence_overview", ""),
                evidence_findings=findings,
                scope_text=getattr(step, "scope_text", ""),
            )

        items.append(CommunicationItem(
            question=_clean(getattr(step, "prompt", "")),
            scope_text=_clean(getattr(step, "scope_text", "")),
            chart_type=_clean(getattr(step, "chart_type", "")) or "visualization",
            insights=_clean(getattr(step, "insights", "")),
            evidence_overview=_clean(getattr(step, "evidence_overview", "")),
            evidence_findings=[_clean(item) for item in findings if _clean(item)],
            translation=translation_text,
            translation_sections=stored_translation,
            image_png=getattr(step, "image_png", None),
        ))

    overall_translation = ""
    limitations = [
        "Each section reflects the scope used for that particular question.",
        "Patterns repeated across analyses may be more useful for follow-up, but repetition does not establish causation.",
    ]
    if overall_summary is not None:
        summary_text = _clean(getattr(overall_summary, "summary_text", ""))
        recurring = [
            _clean(item)
            for item in getattr(overall_summary, "recurring_patterns", [])
            if _clean(item)
        ]
        differences = [
            _clean(item)
            for item in getattr(overall_summary, "important_differences", [])
            if _clean(item)
        ]
        pieces = []
        if summary_text:
            pieces.append(summary_text)
        if recurring:
            pieces.append("Across the questions, the most consistent pattern was: " + recurring[0])
        if differences:
            pieces.append("Some findings were specific to a particular view: " + differences[0])
        if pieces:
            overall_translation = (
                " ".join(pieces)
                + " Together, these analyses provide a structured starting point for "
                "professional interpretation, discussion, and further inquiry."
            )
        limitations = list(getattr(overall_summary, "limitations", []) or limitations)

    return CommunicationPreview(
        scope="entire",
        title="Entire Investigation Communication Preview",
        introduction=(
            f"This preview preserves the sequence of {len(items)} completed "
            f"analysis{'es' if len(items) != 1 else ''}, including each question, "
            "scope, visualization, evidence, and common-language translation."
        ),
        items=items,
        overall_translation=overall_translation,
        limitations=limitations,
    )
