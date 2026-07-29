"""Phase 3B: deterministic evidence extraction for VizCreate."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.filters import apply_spec


@dataclass
class EvidenceFinding:
    finding_id: str
    observation: str
    interpretation: str
    confidence: str
    confidence_reason: str = ""
    evidence_type: str = ""
    focus_label: Optional[str] = None
    focus_column: Optional[str] = None
    magnitude: Optional[float] = None


@dataclass
class EvidenceSummary:
    overview: str
    findings: list[EvidenceFinding] = field(default_factory=list)
    evidence_notes: list[str] = field(default_factory=list)
    recommendation_hints: dict[str, dict[str, Any]] = field(default_factory=dict)
    rows_analyzed: int = 0
    scope_description: str = "current visualization"


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt(value: Any, column: Optional[str] = None) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = str(column or "").lower()
    if any(token in text for token in ["percent", "proficient", "rate"]):
        return f"{number:.1f}%"
    if abs(number - round(number)) < 1e-9:
        return f"{int(round(number)):,}"
    return f"{number:,.1f}"


def _difference_confidence(
    difference: float, spread: float, groups: int, rows: int
) -> tuple[str, str]:
    effect = abs(difference) / spread if spread and np.isfinite(spread) else 0.0
    if rows >= 20 and groups >= 3 and effect >= 0.75:
        return "high", "The pattern is comparatively large and supported by several observations."
    if rows >= 8 and effect >= 0.35:
        return "moderate", "The pattern is noticeable, but more evidence would strengthen the conclusion."
    return "preliminary", "The difference is small or based on limited observations."


def _relationship_confidence(correlation: float, rows: int) -> tuple[str, str]:
    size = abs(correlation)
    if rows >= 30 and size >= 0.60:
        return "high", "The relationship is strong and supported by a substantial number of observations."
    if rows >= 12 and size >= 0.35:
        return "moderate", "The relationship is noticeable, though it should not be interpreted as causal."
    return "preliminary", "The relationship is weak or based on limited observations."


def _measure_columns(df: pd.DataFrame) -> list[str]:
    measures = []
    for column in df.columns:
        name = str(column)
        lower = name.lower()
        converted = _numeric(df[column])
        if converted.notna().mean() < 0.70:
            continue
        if any(
            token in lower
            for token in [
                "score", "percent", "proficient", "math", "reading", "ela",
                "science", "fraction", "geometry", "algebra", "operation",
                "measurement", "statistics", "probability", "construct",
                "rating", "response",
            ]
        ):
            measures.append(name)
    if not measures:
        measures = [
            str(column)
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
            and not any(token in str(column).lower() for token in ["id", "year", "grade"])
        ]
    return list(dict.fromkeys(measures))


def _category_evidence(data, category, measure, aggregation):
    findings, hints = [], {}
    if category not in data.columns or measure not in data.columns:
        return findings, hints
    frame = data[[category, measure]].copy()
    frame[measure] = _numeric(frame[measure])
    frame = frame.dropna()
    if frame.empty or frame[category].nunique() < 2:
        return findings, hints
    agg = aggregation if aggregation in {"mean", "median", "sum", "min", "max"} else "mean"
    grouped = getattr(frame.groupby(category, dropna=False)[measure], agg)().dropna()
    if len(grouped) < 2:
        return findings, hints
    ordered = grouped.sort_values()
    low_label, low_value = str(ordered.index[0]), float(ordered.iloc[0])
    high_label, high_value = str(ordered.index[-1]), float(ordered.iloc[-1])
    difference = high_value - low_value
    confidence, reason = _difference_confidence(
        difference, float(frame[measure].std(ddof=0)), len(grouped), len(frame)
    )
    findings.append(EvidenceFinding(
        "largest_category_difference",
        f"{high_label} has the highest {measure} ({_fmt(high_value, measure)}), while "
        f"{low_label} has the lowest ({_fmt(low_value, measure)}).",
        f"The difference suggests that {low_label} may warrant closer investigation before decisions are made.",
        confidence, reason, "comparison", low_label, category, difference,
    ))
    key = "compare_grades" if "grade" in category.lower() else (
        "compare_subjects" if "subject" in category.lower() else "inspect_variation"
    )
    hints[key] = {
        "focus_label": low_label,
        "focus_column": category,
        "measure": measure,
        "reason": f"{low_label} had the lowest observed {measure}.",
        "boost": 15,
    }
    hints["inspect_variation"] = {
        "focus_label": low_label,
        "focus_column": category,
        "measure": measure,
        "reason": f"The difference between the highest and lowest groups was {_fmt(difference, measure)}.",
        "boost": 6,
    }
    return findings, hints


def _trend_evidence(data, time_col, measure, group_col):
    findings, hints = [], {}
    needed = [time_col, measure] + ([group_col] if group_col else [])
    if any(column not in data.columns for column in needed):
        return findings, hints
    frame = data[needed].copy()
    frame[measure] = _numeric(frame[measure])
    frame = frame.dropna(subset=[time_col, measure])
    groups = [(None, frame)] if not group_col else list(frame.groupby(group_col, dropna=False))
    changes = []
    for label, group in groups:
        trend = group.sort_values(time_col).groupby(time_col)[measure].mean().dropna()
        if len(trend) >= 2:
            changes.append((str(label) if label is not None else "Overall", float(trend.iloc[-1] - trend.iloc[0])))
    if not changes:
        return findings, hints
    label, change = min(changes, key=lambda item: item[1])
    if change >= 0:
        label, change = max(changes, key=lambda item: item[1])
        direction_word = "increase"
        interpretation = f"This improvement may represent a strength worth comparing with other groups or periods."
    else:
        direction_word = "decline"
        interpretation = (
            f"This decline suggests that {label} should be examined across related groups or measures "
            "to determine whether the pattern is broad or localized."
        )
    confidence, reason = _difference_confidence(
        change, float(frame[measure].std(ddof=0)), len(changes), len(frame)
    )
    findings.append(EvidenceFinding(
        "largest_trend_change",
        f"{label} shows the largest {direction_word} in {measure}, changing by "
        f"{_fmt(change, measure)} from the earliest to latest available period.",
        interpretation, confidence, reason, "trend", label, group_col, change,
    ))
    hints["persistent_patterns"] = {
        "focus_label": label, "focus_column": group_col, "measure": measure,
        "reason": f"{label} showed the largest change over time.", "boost": 17,
    }
    for key in ["compare_schools", "compare_subjects", "compare_grades", "compare_subgroups"]:
        hints[key] = {
            "focus_label": label, "focus_column": group_col, "measure": measure,
            "reason": f"Disaggregating {label} can help locate the observed change.", "boost": 10,
        }
    return findings, hints


def _measure_evidence(data):
    findings, hints = [], {}
    measures = _measure_columns(data)
    means = {}
    for column in measures:
        values = _numeric(data[column]).dropna()
        if len(values):
            means[column] = float(values.mean())
    if len(means) < 2:
        return findings, hints
    ordered = sorted(means.items(), key=lambda item: item[1])
    low_col, low_value = ordered[0]
    high_col, high_value = ordered[-1]
    all_values = pd.concat([_numeric(data[col]) for col in means], ignore_index=True).dropna()
    difference = high_value - low_value
    confidence, reason = _difference_confidence(
        difference, float(all_values.std(ddof=0)), len(means), len(data)
    )
    findings.append(EvidenceFinding(
        "weakest_measure",
        f"{low_col} has the lowest average ({_fmt(low_value, low_col)}) among the available measures, "
        f"while {high_col} has the highest ({_fmt(high_value, high_col)}).",
        f"This pattern suggests that {low_col} may be a productive focus for the next investigation.",
        confidence, reason, "measure_comparison", low_col, low_col, difference,
    ))
    hints["investigate_skills"] = {
        "focus_label": low_col, "focus_column": low_col, "measure": low_col,
        "reason": f"{low_col} was the lowest available measure.", "boost": 22,
    }
    hints["compare_student_profiles"] = {
        "focus_label": low_col, "measure": low_col,
        "reason": f"Student profiles may show whether the {low_col} pattern is widespread or concentrated.",
        "boost": 8,
    }
    return findings, hints


def _relationship_evidence(data, x_col, y_col):
    findings, hints = [], {}
    if x_col not in data.columns or y_col not in data.columns:
        return findings, hints
    frame = pd.DataFrame({x_col: _numeric(data[x_col]), y_col: _numeric(data[y_col])}).dropna()
    if len(frame) < 4 or frame[x_col].nunique() < 2 or frame[y_col].nunique() < 2:
        return findings, hints
    corr = float(frame[x_col].corr(frame[y_col]))
    if not np.isfinite(corr):
        return findings, hints
    confidence, reason = _relationship_confidence(corr, len(frame))
    direction = "positive" if corr > 0 else "negative"
    findings.append(EvidenceFinding(
        "measure_relationship",
        f"{x_col} and {y_col} show a {direction} relationship (correlation {corr:.2f}) in the current data.",
        "The measures tend to move together, but this pattern does not establish that one causes the other.",
        confidence, reason, "relationship", f"{x_col} and {y_col}", None, corr,
    ))
    hints["examine_relationships"] = {
        "focus_label": f"{x_col} and {y_col}", "measure": y_col,
        "reason": f"The current relationship has a correlation of {corr:.2f}.", "boost": 12,
    }
    return findings, hints


def _outlier_evidence(data, measure):
    findings, hints = [], {}
    if not measure or measure not in data.columns:
        return findings, hints
    values = _numeric(data[measure]).dropna()
    if len(values) < 8:
        return findings, hints
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return findings, hints
    outliers = values[(values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)]
    if outliers.empty:
        return findings, hints
    confidence = "moderate" if len(values) >= 20 else "preliminary"
    findings.append(EvidenceFinding(
        "outliers",
        f"{len(outliers)} observation{'s' if len(outliers) != 1 else ''} in {measure} "
        "fall outside the typical range of the current data.",
        "These observations may represent meaningful individual differences, data-quality issues, or cases deserving review.",
        confidence,
        "The observations meet a standard interquartile-range rule; their educational meaning requires context.",
        "distribution", measure, measure, float(len(outliers)),
    ))
    hints["find_unusual_profiles"] = {
        "focus_label": measure, "measure": measure,
        "reason": f"{len(outliers)} unusual observations were detected in {measure}.", "boost": 18,
    }
    hints["inspect_variation"] = {
        "focus_label": measure, "measure": measure,
        "reason": f"The data contain {len(outliers)} observations outside the typical range.", "boost": 12,
    }
    return findings, hints


def generate_evidence_summary(
    df: pd.DataFrame,
    spec: dict[str, Any],
    ui_filters: Optional[dict[str, Any]] = None,
    max_findings: int = 3,
) -> EvidenceSummary:
    if not spec:
        return EvidenceSummary("VizCreate does not yet have enough chart information to identify a specific pattern.")
    data = apply_spec(df, spec, ui_filters=ui_filters)
    if data.empty:
        return EvidenceSummary(
            "No observations remain after the current filters, so VizCreate cannot identify a reliable pattern.",
            rows_analyzed=0,
        )

    chart_type = str(spec.get("chart_type") or "").lower()
    special_mode = str(spec.get("special_mode") or "").lower()
    x_col, y_col, group_col = spec.get("x"), spec.get("y"), spec.get("group")
    aggregation = str(spec.get("aggregation") or "mean").lower()
    findings, hints = [], {}

    def collect(result):
        new_findings, new_hints = result
        findings.extend(new_findings)
        for key, value in new_hints.items():
            if key not in hints or value.get("boost", 0) > hints[key].get("boost", 0):
                hints[key] = value

    if special_mode in {"student_support_map", "likert_construct_summary"} or chart_type == "heatmap":
        collect(_measure_evidence(data))
    if chart_type == "line" and x_col and y_col:
        collect(_trend_evidence(data, x_col, y_col, group_col))
    elif chart_type == "scatter" and x_col and y_col:
        collect(_relationship_evidence(data, x_col, y_col))
    elif x_col and y_col:
        collect(_category_evidence(data, x_col, y_col, aggregation))
    collect(_outlier_evidence(data, y_col if y_col in data.columns else None))

    confidence_order = {"high": 3, "moderate": 2, "preliminary": 1}
    unique = {finding.finding_id: finding for finding in findings}
    findings = sorted(
        unique.values(),
        key=lambda item: (confidence_order.get(item.confidence, 0), abs(item.magnitude or 0)),
        reverse=True,
    )[:max_findings]

    if findings:
        first = findings[0].observation
        overview = (
            f"VizCreate identified {len(findings)} pattern{'s' if len(findings) != 1 else ''} "
            f"in the current visualization; the clearest is that {first[0].lower() + first[1:]}"
        )
    else:
        overview = (
            "VizCreate did not detect a sufficiently distinct pattern in the current visualization; "
            "another comparison, distribution, or trend may reveal more."
        )

    return EvidenceSummary(
        overview=overview,
        findings=findings,
        evidence_notes=[
            "Observations describe patterns in the current filtered dataset, not causal conclusions.",
            "Confidence reflects pattern size and available observations.",
        ],
        recommendation_hints=hints,
        rows_analyzed=len(data),
    )
