"""Build compact metadata for the VizCreate Analysis Session card."""

from __future__ import annotations

from html import escape
from typing import Any, Optional

import pandas as pd


def _ordered_unique(series: pd.Series) -> list[str]:
    """Return non-null values as unique strings in their first-seen order."""
    return list(dict.fromkeys(series.dropna().astype(str).tolist()))


def _compact_values(values: list[str], *, max_items: int = 4) -> str:
    """Format a short list without allowing the status card to become too tall."""
    if not values:
        return "—"
    if len(values) <= max_items:
        return ", ".join(values)
    shown = ", ".join(values[:max_items])
    return f"{shown} +{len(values) - max_items} more"


def _chart_label(spec: Optional[dict]) -> str:
    """Convert the chart spec name into a readable label."""
    if not spec:
        return "Not generated"

    chart_type = str(spec.get("chart_type", "")).strip().lower()
    labels = {
        "bar": "Bar chart",
        "stacked_bar": "Stacked bar chart",
        "line": "Line chart",
        "heatmap": "Heatmap",
        "box": "Box-and-whisker plot",
    }
    return labels.get(chart_type, chart_type.replace("_", " ").title() or "Not generated")


def build_analysis_session_summary(
    df: pd.DataFrame,
    *,
    filename: str,
    family_name: str,
    spec: Optional[dict] = None,
    health: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Summarize the active dataset and current chart in a UI-friendly dictionary."""
    year_col = next((c for c in ["School Year", "Year", "Date"] if c in df.columns), None)
    grade_col = next((c for c in ["Grade", "Grade Level", "Testing Grade"] if c in df.columns), None)
    subject_col = "Subject" if "Subject" in df.columns else None

    years = _ordered_unique(df[year_col]) if year_col else []
    grades = _ordered_unique(df[grade_col]) if grade_col else []
    subjects = _ordered_unique(df[subject_col]) if subject_col else []

    filters = spec.get("filters", {}) if isinstance(spec, dict) else {}
    if not isinstance(filters, dict):
        filters = {}

    active_filter_parts = []
    for column in ["Subject", "Subgroup", grade_col, year_col]:
        if not column or column not in filters:
            continue
        value = filters[column]
        if isinstance(value, list):
            value_text = _compact_values([str(item) for item in value])
        else:
            value_text = str(value)
        active_filter_parts.append(f"{column}: {value_text}")

    return {
        "filename": filename,
        "family": family_name,
        "rows": len(df),
        "columns": len(df.columns),
        "year_label": year_col or "Years",
        "years": years,
        "grade_label": grade_col or "Grades",
        "grades": grades,
        "subjects": subjects,
        "chart": _chart_label(spec),
        "active_filters": active_filter_parts,
        "health_status": (health or {}).get("status"),
        "health_label": (health or {}).get("label"),
    }


def analysis_session_card_html(summary: Optional[dict[str, Any]] = None) -> str:
    """Return the styled HTML used in the top-right Analysis Session card."""
    if not summary:
        body = """
            <div class="vc-session-empty">No dataset loaded</div>
            <div class="vc-session-muted">Upload a CSV or Excel file to begin.</div>
        """
    else:
        years = summary.get("years", [])
        grades = summary.get("grades", [])
        subjects = summary.get("subjects", [])
        active_filters = summary.get("active_filters", [])

        year_text = _compact_values(years)
        if len(years) > 1:
            year_text = f"{years[0]} → {years[-1]} ({len(years)})"

        body = f"""
            <div class="vc-session-family">✓ {escape(str(summary.get('family', 'Unknown dataset')))}</div>
            <div class="vc-session-file" title="{escape(str(summary.get('filename', '')))}">
                {escape(str(summary.get('filename', '')))}
            </div>
            <div class="vc-session-grid">
                <div><span>Rows</span><strong>{int(summary.get('rows', 0)):,}</strong></div>
                <div><span>Columns</span><strong>{int(summary.get('columns', 0)):,}</strong></div>
            </div>
        """

        detail_rows = []
        if years:
            detail_rows.append((summary.get("year_label", "Years"), year_text))
        if subjects:
            detail_rows.append(("Subjects", _compact_values(subjects)))
        if grades:
            detail_rows.append((summary.get("grade_label", "Grades"), _compact_values(grades)))

        for label, value in detail_rows:
            body += (
                '<div class="vc-session-detail">'
                f'<span>{escape(str(label))}</span><strong>{escape(str(value))}</strong>'
                '</div>'
            )

        health_label = summary.get("health_label")
        if health_label:
            health_status = str(summary.get("health_status") or "ready")
            body += (
                f'<div class="vc-session-health vc-health-{escape(health_status)}">'
                '<span>Dataset health</span>'
                f'<strong>{escape(str(health_label))}</strong>'
                '</div>'
            )

        body += (
            '<div class="vc-session-analysis">'
            '<span>Current analysis</span>'
            f'<strong>{escape(str(summary.get("chart", "Not generated")))}</strong>'
            '</div>'
        )

        if active_filters:
            body += (
                '<div class="vc-session-filters">'
                f'{escape(" · ".join(active_filters))}'
                '</div>'
            )

    # Streamlit Markdown treats any line beginning with four spaces as a
    # code block, even when unsafe_allow_html=True. Minify the card HTML so
    # no generated line can retain indentation from a triple-quoted string.
    card_html = f"""
    <div class="vc-session-card">
        <div class="vc-session-heading">📊 Analysis Session</div>
        {body}
    </div>
    """
    return "".join(line.strip() for line in card_html.splitlines())
