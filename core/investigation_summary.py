"""Phase 4A: deterministic summaries for current views and full investigations."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re


@dataclass
class InvestigationSummary:
    scope: str
    detail_level: str
    summary_text: str
    recurring_patterns: list[str] = field(default_factory=list)
    important_differences: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommended_next_step: str = ""
    visualizations_reviewed: int = 0
    confidence: str = "preliminary"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _tokens(text: str) -> set[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
        "this", "to", "was", "were", "while", "with",
    }
    return {
        token for token in _normalize(text).split()
        if len(token) > 2 and token not in stop
    }


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedupe(items: list[str], threshold: float = 0.72) -> list[str]:
    result: list[str] = []
    for item in items:
        if not item:
            continue
        if any(_similarity(item, existing) >= threshold for existing in result):
            continue
        result.append(item)
    return result


def _confidence_label(recurring_count: int, independent_views: int) -> str:
    if recurring_count >= 2 and independent_views >= 3:
        return "high"
    if recurring_count >= 1 and independent_views >= 2:
        return "moderate"
    return "preliminary"


def summarize_current_view(
    prompt: str,
    spec: dict[str, Any],
    read_description: str,
    evidence_summary: Any,
    next_step: str = "",
    detail_level: str = "concise",
) -> InvestigationSummary:
    chart_type = str(spec.get("chart_type") or "visualization").replace("_", " ")
    findings = [finding.observation for finding in getattr(evidence_summary, "findings", [])]
    interpretations = [finding.interpretation for finding in getattr(evidence_summary, "findings", [])]

    if findings:
        lead = findings[0]
        if detail_level == "detailed":
            summary_text = (
                f"The current question was: “{prompt}” VizCreate used a {chart_type} to examine the available evidence. "
                f"{read_description} The strongest observed pattern is that {lead[0].lower() + lead[1:]}"
            )
            if len(findings) > 1:
                summary_text += " Additional observations include " + "; ".join(findings[1:]) + "."
            if interpretations:
                summary_text += " These patterns suggest " + "; ".join(
                    item[0].lower() + item[1:] if item else item for item in interpretations[:2]
                )
        else:
            summary_text = (
                f"This {chart_type} addresses “{prompt}” The clearest observed pattern is that "
                f"{lead[0].lower() + lead[1:]}"
            )
    else:
        summary_text = (
            f"This {chart_type} addresses “{prompt}” but VizCreate did not detect a sufficiently distinct "
            "pattern in the current filtered data."
        )

    limitations = [
        "The summary describes the current filtered dataset and does not establish causation.",
        "Observed differences should be interpreted alongside instructional context, data quality, and sample size.",
    ]

    confidence = (
        getattr(evidence_summary.findings[0], "confidence", "preliminary")
        if getattr(evidence_summary, "findings", [])
        else "preliminary"
    )

    return InvestigationSummary(
        scope="current_view",
        detail_level=detail_level,
        summary_text=summary_text,
        recurring_patterns=findings[:2],
        important_differences=findings[2:3],
        limitations=limitations,
        recommended_next_step=next_step or (
            "Compare the pattern across another relevant group, measure, or time period."
        ),
        visualizations_reviewed=1,
        confidence=confidence,
    )


def summarize_entire_investigation(
    state: Any,
    current_prompt: str,
    current_spec: dict[str, Any],
    read_description: str,
    evidence_summary: Any,
    next_step: str = "",
    detail_level: str = "concise",
) -> InvestigationSummary:
    records: list[dict[str, Any]] = []

    completed_steps = [
        step for step in getattr(state, "history", [])
        if getattr(step, "step_type", "completed_analysis") != "transition"
    ]

    for step in completed_steps:
        records.append({
            "prompt": getattr(step, "prompt", ""),
            "chart_type": getattr(step, "chart_type", "") or "visualization",
            "findings": list(getattr(step, "evidence_findings", []) or []),
            "scope_text": getattr(step, "scope_text", ""),
        })

    # Backward compatibility: if the current version has not yet recorded the
    # active chart, include it once. In the normal Phase 2F memory flow, the
    # active chart has already been upserted into completed_steps.
    current_exists = any(
        _normalize(record["prompt"]) == _normalize(current_prompt)
        and record["chart_type"] == str(current_spec.get("chart_type") or "visualization")
        for record in records
    )
    if not current_exists:
        records.append({
            "prompt": current_prompt,
            "chart_type": str(current_spec.get("chart_type") or "visualization"),
            "findings": [
                finding.observation for finding in getattr(evidence_summary, "findings", [])
            ],
            "scope_text": "",
        })

    all_findings = []
    for record_index, record in enumerate(records):
        for finding in record["findings"]:
            all_findings.append((finding, record_index, record["chart_type"]))

    clusters: list[dict[str, Any]] = []
    for finding, record_index, chart_type in all_findings:
        matched = None
        for cluster in clusters:
            if _similarity(finding, cluster["representative"]) >= 0.48:
                matched = cluster
                break
        if matched:
            matched["items"].append(finding)
            matched["records"].add(record_index)
            matched["charts"].add(chart_type)
        else:
            clusters.append({
                "representative": finding,
                "items": [finding],
                "records": {record_index},
                "charts": {chart_type},
            })

    recurring = _dedupe([
        cluster["representative"]
        for cluster in clusters
        if len(cluster["records"]) >= 2
    ])
    unique_findings = _dedupe([item[0] for item in all_findings])
    differences = [
        finding for finding in unique_findings
        if not any(_similarity(finding, recurring_item) >= 0.48 for recurring_item in recurring)
    ][:3]

    reviewed = len(records)
    independent_views = len({
        (record["chart_type"], _normalize(record["prompt"]))
        for record in records
    })

    if recurring:
        if detail_level == "detailed":
            summary_text = (
                f"Across {reviewed} visualization{'s' if reviewed != 1 else ''}, "
                f"the investigation returned to several related patterns. The most consistent finding was that "
                f"{recurring[0][0].lower() + recurring[0][1:]}"
            )
            if len(recurring) > 1:
                summary_text += " Other recurring patterns included " + "; ".join(recurring[1:]) + "."
            if differences:
                summary_text += " Some findings were specific to individual views, including " + "; ".join(differences) + "."
        else:
            summary_text = (
                f"Across {reviewed} visualization{'s' if reviewed != 1 else ''}, the clearest recurring pattern was that "
                f"{recurring[0][0].lower() + recurring[0][1:]}"
            )
    elif unique_findings:
        summary_text = (
            f"Across {reviewed} visualization{'s' if reviewed != 1 else ''}, VizCreate identified several findings, "
            "but no single pattern appeared consistently across multiple distinct views. "
            f"One notable observation was that {unique_findings[0][0].lower() + unique_findings[0][1:]}"
        )
    else:
        summary_text = (
            f"The investigation included {reviewed} visualization{'s' if reviewed != 1 else ''}, "
            "but the stored evidence did not contain a sufficiently distinct recurring pattern."
        )

    limitations = [
        "Repeated findings are treated as stronger only when they appear across distinct prompts or visualization types.",
        "The investigation summarizes observed patterns and does not establish causes.",
        "Results remain dependent on the available fields, filters, sample sizes, and data quality.",
    ]

    return InvestigationSummary(
        scope="entire_investigation",
        detail_level=detail_level,
        summary_text=summary_text,
        recurring_patterns=recurring[:4],
        important_differences=differences,
        limitations=limitations,
        recommended_next_step=next_step or (
            "Investigate the strongest recurring pattern using a different measure, subgroup, or time frame."
            if recurring else
            "Use a new analytical perspective to determine whether any observed findings persist."
        ),
        visualizations_reviewed=reviewed,
        confidence=_confidence_label(len(recurring), independent_views),
    )
