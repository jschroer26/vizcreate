"""Profile-aware chart-spec guidance and deterministic repairs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.measure_hierarchy import (
    detect_measure_families,
    detect_measure_scope,
    resolve_measure_columns,
)

from profiles.base_profile import DatasetProfileResult


CONCEPTUAL_FIELD_NAMES = {
    "survey_construct",
    "survey construct",
    "construct",
    "constructs",
    "rating",
    "ratings",
    "response",
    "responses",
    "score",
}


def exact_survey_item_columns(
    df: pd.DataFrame,
    profile: DatasetProfileResult,
) -> list[str]:
    """Extract exact wide-format survey item columns from profile metadata."""
    role_value = profile.detected_roles.get("survey_items", "")
    candidates = [
        item.strip()
        for item in str(role_value).split(",")
        if item.strip()
    ]

    exact = [
        column
        for column in candidates
        if column in df.columns
        and pd.api.types.is_numeric_dtype(df[column])
    ]
    return exact


def profile_prompt_block(profile: DatasetProfileResult) -> str:
    """Build explicit, executable profile guidance for the LLM."""
    role_lines = [
        f"- {role.replace('_', ' ').title()}: {value}"
        for role, value in profile.detected_roles.items()
    ]
    role_text = "\n".join(role_lines) if role_lines else "- No specialized roles detected."

    recommended = ", ".join(profile.recommended_charts) or "Use supported chart types."
    discouraged = ", ".join(profile.discouraged_charts) or "None listed."
    cautions = "\n".join(f"- {item}" for item in profile.cautions) or "- None listed."

    special_guidance = ""
    if profile.profile_id == "likert_survey":
        special_guidance = """
LIKERT EXECUTION RULES
- Detected construct labels and role names are metadata, not dataframe columns.
- Never use invented fields such as survey_construct, construct, rating, or response
  unless those exact names actually occur in the dataset schema.
- For a wide-format numeric Likert request comparing several constructs, use:
  special_mode = "likert_construct_summary"
  x = null
  y = null
  group = null
  item_columns = an array containing exact survey item column names.
- item_columns must contain only exact column names from the dataset.
- Use aggregation = "mean" only as a concise descriptive ranking, and note that
  the responses are ordinal. The plotting engine will reshape the wide items.
"""
    elif profile.profile_id == "student_assessment":
        special_guidance = """
STUDENT-ASSESSMENT EXECUTION RULES
- A single assessment score per student supports distributions and group comparisons.
- Do not describe a box plot or grade comparison as progress, growth, or movement
  toward a goal unless repeated observations and a time field exist.
"""
    elif profile.profile_id == "cbm_progress_monitoring":
        special_guidance = """
CBM EXECUTION RULES
- Use a line chart for repeated student scores over an exact date or screening-window column.
- Use a goal field only when the exact goal column exists.
- Do not substitute a grade-level box plot for progress-monitoring or goal-attainment analysis.
"""

    return f"""
PROFILE-GOVERNED PLANNING
-------------------------
Profile: {profile.display_name}
Structure: {profile.structure}
Confidence: {profile.confidence_percent}%
Description: {profile.description}

Detected roles:
{role_text}

Recommended views: {recommended}
Use cautiously or avoid: {discouraged}

Profile guidance:
{profile.prompt_guidance}

Analytical cautions:
{cautions}
{special_guidance}
""".strip()


def repair_spec_for_profile(
    df: pd.DataFrame,
    spec: dict[str, Any],
    profile: DatasetProfileResult,
    user_prompt: str,
) -> dict[str, Any]:
    """
    Repair common profile-specific planning mistakes.

    This does not replace validation. It converts conceptual survey fields into an
    executable special mode when the uploaded survey is wide-format.
    """
    repaired = dict(spec or {})
    repaired.setdefault("filters", {})

    # Phase 2G.3 Revised: resolve subject families and measure scope
    # deterministically before applying profile-specific repairs.
    hierarchy = detect_measure_families(df, profile=profile)
    measure_scope = detect_measure_scope(user_prompt)
    selected_measures = resolve_measure_columns(hierarchy, measure_scope)

    if (
        measure_scope.subject_id is not None
        and measure_scope.scope in {"categories", "all_measures"}
        and len(selected_measures) >= 2
    ):
        group_column = next(
            (
                column
                for column in ["Grade Level", "Testing Grade", "Grade"]
                if column in df.columns
            ),
            repaired.get("x") if repaired.get("x") in df.columns else None,
        )

        if group_column is not None:
            repaired.update(
                {
                    "chart_type": "bar",
                    "special_mode": "multi_measure_bar",
                    "x": group_column,
                    "y": None,
                    "group": None,
                    "row": None,
                    "col": None,
                    "item_columns": selected_measures,
                    "aggregation": "mean",
                    "sort_x": "ascending",
                    "facets": None,
                }
            )

            family = hierarchy.families.get(measure_scope.subject_id)
            family_name = (
                family.display_name
                if family is not None
                else measure_scope.subject_id.title()
            )
            scope_label = (
                "all measures"
                if measure_scope.scope == "all_measures"
                else "category/subscore measures"
            )
            existing_notes = str(repaired.get("notes", "")).strip()
            repair_note = (
                f"VizCreate interpreted the request as {scope_label} within "
                f"{family_name}. It selected {len(selected_measures)} numeric "
                f"measure columns and will compare their mean scores across "
                f"{group_column}."
            )
            repaired["notes"] = (
                f"{existing_notes} {repair_note}".strip()
                if existing_notes
                else repair_note
            )
            repaired["measure_scope_metadata"] = {
                "subject_id": measure_scope.subject_id,
                "scope": measure_scope.scope,
                "selected_columns": selected_measures,
                "overall_columns": (
                    list(family.overall) if family is not None else []
                ),
                "category_columns": (
                    list(family.categories) if family is not None else []
                ),
                "confidence": measure_scope.confidence,
            }
            return repaired

    # For an explicit overall-score request, prefer the identified overall measure.
    if (
        measure_scope.subject_id is not None
        and measure_scope.scope == "overall"
        and selected_measures
    ):
        group_column = next(
            (
                column
                for column in ["Grade Level", "Testing Grade", "Grade"]
                if column in df.columns
            ),
            repaired.get("x") if repaired.get("x") in df.columns else None,
        )
        repaired["y"] = selected_measures[0]
        if group_column is not None:
            repaired["x"] = group_column
        repaired["item_columns"] = []

    if profile.profile_id != "likert_survey":
        return repaired

    structure = str(profile.structure).lower()
    if "wide-format" not in structure:
        return repaired

    prompt_lower = user_prompt.lower()
    asks_across_constructs = any(
        phrase in prompt_lower
        for phrase in [
            "which survey construct",
            "which construct",
            "strongest ratings",
            "highest ratings",
            "compare constructs",
            "construct received",
            "survey items",
            "which item received",
        ]
    )

    referenced = {
        str(repaired.get(field, "")).strip().lower()
        for field in ["x", "y", "group", "row", "col"]
        if repaired.get(field) is not None
    }
    invented_conceptual_fields = bool(
        referenced.intersection(CONCEPTUAL_FIELD_NAMES)
    )
    missing_references = any(
        repaired.get(field) is not None
        and repaired.get(field) not in df.columns
        for field in ["x", "y", "group", "row", "col"]
    )

    item_columns = exact_survey_item_columns(df, profile)

    if item_columns and (
        asks_across_constructs
        or invented_conceptual_fields
        or missing_references
        or repaired.get("special_mode") == "likert_construct_summary"
    ):
        repaired.update(
            {
                "chart_type": "bar",
                "special_mode": "likert_construct_summary",
                "x": None,
                "y": None,
                "group": None,
                "row": None,
                "col": None,
                "item_columns": item_columns,
                "aggregation": "mean",
                "sort_x": "descending",
                "facets": None,
            }
        )

        existing_notes = str(repaired.get("notes", "")).strip()
        repair_note = (
            "VizCreate compared the exact wide-format survey item columns. "
            "Mean ratings are used as a concise descriptive ranking of ordinal responses."
        )
        repaired["notes"] = (
            f"{existing_notes} {repair_note}".strip()
            if existing_notes
            else repair_note
        )

    return repaired
