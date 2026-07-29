import io
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import re
from openai import OpenAI

from config.palettes import PALETTES, palette_names
from core.data_loader import load_dataframe
from core.chart_engine import make_chart_from_spec
from core.filters import apply_spec
from core.intent_filters import enrich_spec_with_intent_filters
from core.analysis_session import (
    analysis_session_card_html,
    build_analysis_session_summary,
)
from core.data_health import assess_dataset_health, health_icon
from core.dataset_intelligence_ui import render_dataset_intelligence
from core.profile_detector import (
    detect_dataset_profile,
    legacy_family_name,
    profile_guidance,
    rank_dataset_profiles,
)
from core.measure_hierarchy import (
    detect_measure_families,
    detect_measure_scope,
    measure_hierarchy_prompt_block,
)
from core.profile_planner import (
    profile_prompt_block,
    repair_spec_for_profile,
)
from core.intent_recognition import (
    AnalysisIntentResult,
    detect_analysis_intent,
    intent_prompt_block,
    repair_spec_for_intent,
)
from core.role_coach import (
    ROLE_OPTIONS,
    AnalystCoachPlan,
    UserRoleResult,
    analyst_plan_prompt_block,
    build_analyst_coach_plan,
    resolve_user_role,
    role_prompt_block,
)
from core.decision_context import (
    DecisionBasisResult,
    TargetUnitResult,
    decision_context_prompt_block,
    detect_decision_basis,
    detect_target_unit,
)
from core.planner_v2 import (
    PlannerResult,
    build_planner_prompt,
    extract_json_object,
    fallback_planner_result,
    parse_and_rank_planner_response,
)
from core.dive_deeper import (
    InvestigationState,
    add_coach_investigation_step,
    add_investigation_step,
    record_completed_analysis,
    completed_analysis_steps,
    set_current_analysis_translation,
    generate_dive_deeper_suggestions,
)
from core.coach_bridge import generate_coach_bridge_suggestions
from core.evidence_engine import generate_evidence_summary
from core.common_language_translation import generate_translation
from core.communication_preview import (
    current_preview,
    entire_preview,
)
from core.report_generator import (
    ReportModel,
    build_analyst_summary,
    build_pdf_report,
    build_word_report,
    build_pdf_investigation,
    build_word_investigation,
)

from core.investigation_summary import (
    summarize_current_view,
    summarize_entire_investigation,
)
from profiles.base_profile import DatasetProfileResult
from core.session_manager import (
    dataset_changed,
    initialize_session_state,
    reset_analysis_state,
    start_new_analysis,
)

# -------------------------
# OpenAI client
# -------------------------
client = OpenAI()  # uses OPENAI_API_KEY from environment


# -------------------------
# LLM helper functions
# -------------------------
def summarize_dataframe_for_prompt(df: pd.DataFrame, max_examples: int = 3) -> str:
    """
    Build a human-readable schema description for the LLM:
    column names, inferred types, and a few example values.
    """
    lines = []
    lines.append("Columns:")

    for col in df.columns:
        series = df[col]
        # crude type mapping
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        else:
            col_type = "categorical"

        # get up to max_examples unique non-null values
        non_null = series.dropna().unique()
        examples = list(non_null[:max_examples])
        examples_str = ", ".join(map(lambda x: str(x), examples))

        lines.append(f"- {col} (type: {col_type}, example values: [{examples_str}])")

    return "\n".join(lines)


def detect_dataset_family(df: pd.DataFrame) -> tuple[str, str]:
    """Classify the uploaded dataset using its column structure."""
    columns = set(df.columns.astype(str))

    wytopp_required = {
        "School Year",
        "Grade",
        "Subject",
        "Percent Basic and Below",
        "Percent Proficient and Advanced",
    }
    if wytopp_required.issubset(columns):
        return (
            "WYTOPP longitudinal proficiency summary",
            "Aggregated Grade × School Year × Subject data. Prefer line charts, heatmaps, "
            "or the special WYTOPP stacked-bar mode. Do not recommend box plots because "
            "the rows are aggregated rather than individual student observations.",
        )

    student_markers = {"Student #", "Student ID", "Student"}
    grade_markers = {"Grade", "Grade Level", "Testing Grade"}
    has_student = bool(columns.intersection(student_markers))
    has_grade = bool(columns.intersection(grade_markers))
    numeric_count = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

    if has_student and has_grade and numeric_count >= 1:
        return (
            "student-level assessment data",
            "One row appears to represent one student. Bar charts, heatmaps, line charts, "
            "and box plots may be appropriate depending on the request.",
        )

    likert_terms = {"Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"}
    object_values = set()
    for col in df.select_dtypes(exclude=["number"]).columns[:10]:
        object_values.update(df[col].dropna().astype(str).unique()[:20])
    if len(likert_terms.intersection(object_values)) >= 3:
        return (
            "survey or Likert-response data",
            "Prefer bar charts or stacked bars for response categories and avoid treating "
            "ordinal responses as continuous unless the user explicitly requests it.",
        )

    if any(name in columns for name in ["School Year", "Year", "Date"]):
        return (
            "longitudinal summary data",
            "The file includes a time field. Prefer line charts for trends and bar charts "
            "for discrete comparisons.",
        )

    return (
        "general tabular data",
        "Choose a chart supported by the available categorical and numeric columns. "
        "Do not invent columns or request unsupported distributions.",
    )


def build_vizcreate_prompt(
    df: pd.DataFrame,
    user_prompt: str,
    profile: Optional[DatasetProfileResult] = None,
    intent: Optional[AnalysisIntentResult] = None,
    role: Optional[UserRoleResult] = None,
    coach_plan: Optional[AnalystCoachPlan] = None,
    target_unit: Optional[TargetUnitResult] = None,
    decision_basis: Optional[DecisionBasisResult] = None,
) -> str:
    """Construct the dataset-aware prompt sent to the visualization planner."""
    schema_text = summarize_dataframe_for_prompt(df)

    if profile is None:
        profile = detect_dataset_profile(df)

    family_name = legacy_family_name(profile)
    family_guidance = profile_guidance(profile)
    intelligence_text = profile_prompt_block(profile)
    measure_hierarchy_text = measure_hierarchy_prompt_block(
        detect_measure_families(df, profile=profile)
    )

    if intent is None:
        intent = detect_analysis_intent(
            df,
            user_prompt,
            profile,
        )
    intent_text = intent_prompt_block(intent)

    if role is None:
        role = resolve_user_role(user_prompt)
    role_text = role_prompt_block(role)

    if target_unit is None:
        target_unit = detect_target_unit(df, user_prompt, profile)
    if decision_basis is None:
        decision_basis = detect_decision_basis(
            df,
            user_prompt,
            profile,
            target_unit,
        )
    decision_context_text = decision_context_prompt_block(
        target_unit,
        decision_basis,
    )

    if coach_plan is None:
        coach_plan = build_analyst_coach_plan(
            df,
            profile,
            intent,
            role,
            target_unit=target_unit,
            decision_basis=decision_basis,
            user_prompt=user_prompt,
        )
    coach_plan_text = analyst_plan_prompt_block(coach_plan)

    system_instructions = """
You are VizCreate, a data visualization planning assistant.

Your job is to inspect a tabular dataset and convert the user's request into one valid chart specification.

Supported chart types:
- "bar": a single/grouped bar chart; for wide subject subcategory comparisons,
  use special_mode "multi_measure_bar", y null, and exact numeric item_columns
- "stacked_bar": a stacked bar chart
- "line": an ordered trend chart, usually over time
- "heatmap": a grid where row and col are categorical columns and y is numeric
- "box": a box-and-whisker plot; use only for student-level or observation-level data
- "scatter": a row-level relationship between two different numeric columns

Rules:
- Use only exact column names from the dataset.
- Never invent a column.
- Respect the detected dataset family and avoid charts that the data cannot support.
- For a heatmap, always populate row, col, and y explicitly.
- For time trends, normally use School Year or Year as x.
- When the dataset contains separate columns named Percent Basic and Below and
  Percent Proficient and Advanced, and the user asks to compare or stack those measures,
  use chart_type "stacked_bar" and special_mode "wytopp_stacked".
- In WYTOPP stacked mode, set x to Grade or School Year, based on the user's wording.
  Set y to null and group to null. The plotting engine will stack the two known percentage columns.
- Distinguish overall subject scores from category/subscore measures.
- When the user asks for categories, subscores, domains, strands, standards, components,
  or reasoning skills within a subject, select every exact numeric category/subscore
  column in that subject family, exclude the broad overall score, use chart_type "bar",
  special_mode "multi_measure_bar", x as the requested comparison group, and y null.
- This rule applies generically to Math, ELA/Reading, Science, and any future subject
  family detected from column names and profile metadata.
- For broad or open-ended WYTOPP requests, do not average subjects together.
  When multiple years are available, Python may select the subject with the greatest
  earliest-to-latest increase. When only one year is available, prefer an ELA
  proficiency comparison across grade levels rather than claiming growth.
- Return JSON only, with no markdown or explanation outside the JSON.
"""

    json_schema = """
Return exactly one JSON object with these fields:
{
  "chart_type": "bar | stacked_bar | line | heatmap | box | scatter",
  "special_mode": "wytopp_stacked | likert_construct_summary | student_support_map | student_ranked_scores | multi_measure_bar | multi_measure_box | null",
  "x": "exact x-axis column name, or null",
  "y": "exact primary numeric column name, or null",
  "group": "exact grouping/series column name, or null",
  "row": "heatmap row column, or null",
  "col": "heatmap column column, or null",
  "filters": {"ColumnName": "exact value or array of values"},
  "item_columns": ["exact numeric wide-format category/subscore columns for approved multi-measure modes"],
  "aggregation": "mean | sum | count | none",
  "sort_x": "none | ascending | descending",
  "facets": null,
  "notes": "brief interpretation"
}

Heatmap example:
{
  "chart_type": "heatmap",
  "special_mode": null,
  "x": null,
  "y": "Percent Proficient and Advanced",
  "group": null,
  "row": "Grade",
  "col": "School Year",
  "filters": {"Subject": "Math"},
  "aggregation": "mean",
  "sort_x": "ascending",
  "facets": null,
  "notes": "Math proficiency by grade and school year."
}


Scatterplot example:
{
  "chart_type": "scatter",
  "special_mode": null,
  "x": "Math Scale Score",
  "y": "Reading/ELA Scale Score",
  "group": null,
  "row": null,
  "col": null,
  "filters": {},
  "item_columns": [],
  "aggregation": "none",
  "sort_x": "none",
  "facets": null,
  "notes": "Examine the row-level relationship between two student scale scores."
}

WYTOPP stacked-bar example:
{
  "chart_type": "stacked_bar",
  "special_mode": "wytopp_stacked",
  "x": "Grade",
  "y": null,
  "group": null,
  "row": null,
  "col": null,
  "filters": {"Subject": "Math"},
  "aggregation": "mean",
  "sort_x": "ascending",
  "facets": null,
  "notes": "Stack Basic and Below with Proficient and Advanced for each grade."
}

Wide numeric Likert construct-summary example:
{
  "chart_type": "bar",
  "special_mode": "likert_construct_summary",
  "x": null,
  "y": null,
  "group": null,
  "row": null,
  "col": null,
  "filters": {},
  "item_columns": [
    "Q1_EaseOfUse",
    "Q2_CognitiveLoad",
    "Q3_PerceivedUtility"
  ],
  "aggregation": "mean",
  "sort_x": "descending",
  "facets": null,
  "notes": "Compare exact wide-format survey item columns using descriptive mean ratings."
}
"""

    return f"""
{system_instructions}

DETECTED DATASET FAMILY
-----------------------
Family: {family_name}
Guidance: {family_guidance}

{intelligence_text}

{measure_hierarchy_text}

{intent_text}

{role_text}

{decision_context_text}

{coach_plan_text}

DATASET SCHEMA
--------------
{schema_text}

USER REQUEST
------------
{user_prompt}

OUTPUT FORMAT
-------------
{json_schema}
""".strip()



def choose_subject_with_greatest_growth(
    df: pd.DataFrame,
    subject_col: str = "Subject",
    year_col: str = "School Year",
    proficiency_col: str = "Percent Proficient and Advanced",
    n_col: str = "Number of Students Tested",
) -> Optional[str]:
    """
    Return the subject with the largest earliest-to-latest increase in
    Percent Proficient and Advanced.

    When Number of Students Tested is available, yearly subject values are
    weighted by N. Otherwise, a simple mean is used. Subjects need at least
    two distinct years to be considered.
    """
    required = {subject_col, year_col, proficiency_col}
    if not required.issubset(df.columns):
        return None

    data = df.copy()
    data[proficiency_col] = pd.to_numeric(
        data[proficiency_col],
        errors="coerce",
    )

    if n_col in data.columns:
        data[n_col] = pd.to_numeric(
            data[n_col],
            errors="coerce",
        )

    data = data.dropna(
        subset=[subject_col, year_col, proficiency_col]
    )

    if data.empty:
        return None

    # Restrict the comparison to All Students when available.
    if "Subgroup" in data.columns:
        subgroup_values = (
            data["Subgroup"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if "All Students" in subgroup_values:
            data = data[
                data["Subgroup"].astype(str) == "All Students"
            ]

    summary_rows = []

    for (subject, year), group in data.groupby(
        [subject_col, year_col],
        dropna=False,
    ):
        if (
            n_col in group.columns
            and group[n_col].notna().any()
            and group[n_col].fillna(0).sum() > 0
        ):
            valid = group[
                group[proficiency_col].notna()
                & group[n_col].notna()
                & (group[n_col] > 0)
            ]

            if valid.empty:
                value = group[proficiency_col].mean()
            else:
                value = np.average(
                    valid[proficiency_col],
                    weights=valid[n_col],
                )
        else:
            value = group[proficiency_col].mean()

        summary_rows.append(
            {
                subject_col: subject,
                year_col: str(year),
                "Proficiency": value,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return None

    growth_results = []

    for subject, group in summary.groupby(subject_col):
        group = group.dropna(subset=["Proficiency"]).copy()

        if group[year_col].nunique() < 2:
            continue

        group = group.sort_values(year_col)
        earliest = group.iloc[0]
        latest = group.iloc[-1]

        growth_results.append(
            {
                "Subject": str(subject),
                "Growth": float(
                    latest["Proficiency"] - earliest["Proficiency"]
                ),
            }
        )

    if not growth_results:
        return None

    growth_df = pd.DataFrame(growth_results)
    best_row = growth_df.loc[growth_df["Growth"].idxmax()]
    return str(best_row["Subject"])


def choose_ela_fallback(df: pd.DataFrame) -> Optional[str]:
    """Return the exact ELA-related Subject value used in the uploaded file."""
    if "Subject" not in df.columns:
        return None

    subject_values = (
        df["Subject"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    preferred_values = [
        "ELA",
        "English",
        "English Language Arts",
        "Reading",
        "Reading/ELA",
    ]

    for preferred in preferred_values:
        if preferred in subject_values:
            return preferred

    for value in subject_values:
        value_lower = value.lower()
        if any(
            term in value_lower
            for term in [
                "ela",
                "english",
                "reading",
                "language arts",
            ]
        ):
            return value

    return None


def user_named_subject_in_prompt(
    user_prompt: str,
    subject_values: list[str],
) -> bool:
    """
    Return True when the prompt explicitly names an available subject or a
    common subject synonym. This prevents broad-prompt safeguards from
    overriding a user's explicit request.
    """
    prompt_lower = user_prompt.lower()

    common_subject_terms = [
        "ela",
        "english",
        "reading",
        "language arts",
        "math",
        "mathematics",
        "science",
    ]
    if any(term in prompt_lower for term in common_subject_terms):
        return True

    return any(
        str(subject).lower() in prompt_lower
        for subject in subject_values
        if str(subject).strip()
    )




def _response_text(response) -> str:
    """Extract text from an OpenAI Responses API result."""
    pieces = []
    for output_item in getattr(response, "output", []) or []:
        for block in getattr(output_item, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                pieces.append(text)
    if pieces:
        return "".join(pieces).strip()
    return str(response)


def get_planner_result_from_llm(
    df: pd.DataFrame,
    user_prompt: str,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    coach_plan: AnalystCoachPlan,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
) -> PlannerResult:
    """Run Planner 2.0 and return the ranked plan plus executable chart spec."""
    schema_text = summarize_dataframe_for_prompt(df)
    planner_prompt = build_planner_prompt(
        schema_text=schema_text,
        user_prompt=user_prompt,
        profile_text=profile_prompt_block(profile),
        intent_text=intent_prompt_block(intent),
        role_text=role_prompt_block(role),
        decision_context_text=decision_context_prompt_block(
            target_unit,
            decision_basis,
        ),
        coach_plan_text=analyst_plan_prompt_block(coach_plan),
    )

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=planner_prompt,
        )
        raw_text = _response_text(response)
        raw_plan = extract_json_object(raw_text)
        planner_result = parse_and_rank_planner_response(
            raw_plan,
            df=df,
            profile=profile,
            intent=intent,
            role=role,
            target_unit=target_unit,
            decision_basis=decision_basis,
            user_prompt=user_prompt,
        )

        spec = repair_spec_for_profile(
            df,
            planner_result.final_spec,
            profile,
            user_prompt,
        )
        spec = repair_spec_for_intent(
            df,
            spec,
            intent,
            profile=profile,
            target_unit=target_unit,
            decision_basis=decision_basis,
            analytical_objective=planner_result.analytical_objective,
        )
        planner_result.final_spec = spec
        return planner_result

    except Exception as planner_error:
        fallback_spec = get_chart_spec_from_llm_legacy(
            df,
            user_prompt,
            profile=profile,
            intent=intent,
            role=role,
            coach_plan=coach_plan,
            target_unit=target_unit,
            decision_basis=decision_basis,
        )
        return fallback_planner_result(
            spec=fallback_spec,
            coach_plan=coach_plan,
            intent=intent,
            target_unit=target_unit,
            decision_basis=decision_basis,
            error_message=f"Planner 2.0 fallback: {planner_error}",
        )


def get_chart_spec_from_llm_legacy(
    df: pd.DataFrame,
    user_prompt: str,
    profile: Optional[DatasetProfileResult] = None,
    intent: Optional[AnalysisIntentResult] = None,
    role: Optional[UserRoleResult] = None,
    coach_plan: Optional[AnalystCoachPlan] = None,
    target_unit: Optional[TargetUnitResult] = None,
    decision_basis: Optional[DecisionBasisResult] = None,
) -> dict:
    """
    Call the OpenAI API with our prompt and return a parsed chart spec dict.

    If anything goes wrong, show the error in Streamlit and return a simple fallback spec.
    """
    if profile is None:
        profile = detect_dataset_profile(df)
    if intent is None:
        intent = detect_analysis_intent(
            df,
            user_prompt,
            profile,
        )
    if role is None:
        role = resolve_user_role(user_prompt)
    if target_unit is None:
        target_unit = detect_target_unit(df, user_prompt, profile)
    if decision_basis is None:
        decision_basis = detect_decision_basis(
            df,
            user_prompt,
            profile,
            target_unit,
        )
    if coach_plan is None:
        coach_plan = build_analyst_coach_plan(
            df,
            profile,
            intent,
            role,
            target_unit=target_unit,
            decision_basis=decision_basis,
        )

    prompt_text = build_vizcreate_prompt(
        df,
        user_prompt,
        profile=profile,
        intent=intent,
        role=role,
        coach_plan=coach_plan,
        target_unit=target_unit,
        decision_basis=decision_basis,
    )

    try:
        # Call the Responses API
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt_text,
        )

        # Extract text from the response
        try:
            blocks = response.output[0].content
            pieces = []
            for block in blocks:
                if hasattr(block, "text") and block.text is not None:
                    pieces.append(block.text)
            raw_text = "".join(pieces).strip()
        except Exception as e:
            raw_text = str(response)
            st.warning(f"Could not extract text cleanly from response: {e}")

        # Try to parse JSON
        try:
            spec = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                spec = json.loads(match.group(0))
            else:
                raise

        # Convert profile-level concepts into executable, exact-column specs.
        spec = repair_spec_for_profile(
            df,
            spec,
            profile,
            user_prompt,
        )

        spec = repair_spec_for_intent(
            df,
            spec,
            intent,
            profile=profile,
            target_unit=target_unit,
            decision_basis=decision_basis,
        )

        # -------------------------------------------------
        # Dataset-aware safeguards for broad WYTOPP prompts
        # -------------------------------------------------
        columns = set(df.columns.astype(str))

        is_wytopp = {
            "School Year",
            "Grade",
            "Subject",
            "Percent Basic and Below",
            "Percent Proficient and Advanced",
        }.issubset(columns)

        broad_prompt_phrases = [
            "if you were the principal",
            "what chart would you look at first",
            "what chart should i look at first",
            "what should i look at first",
            "what should i notice first",
            "show me the most important trend",
            "show me the most important trends",
            "where should we start",
            "give me an overview",
            "what should we focus on",
            "what should i focus on",
        ]

        prompt_lower = user_prompt.lower().strip()
        is_broad_prompt = any(
            phrase in prompt_lower
            for phrase in broad_prompt_phrases
        )

        if is_wytopp and is_broad_prompt:
            filters = spec.get("filters", {})
            if not isinstance(filters, dict):
                filters = {}

            subject_values = (
                df["Subject"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            user_named_subject = user_named_subject_in_prompt(
                user_prompt,
                subject_values,
            )

            year_count = (
                df["School Year"]
                .dropna()
                .astype(str)
                .nunique()
            )

            # If the user did not name a subject and the model did not select
            # one, choose the subject showing the greatest growth. When fewer
            # than two years are available, growth cannot be calculated, so
            # VizCreate falls back to the exact ELA-related label in the file.
            selected_subject = filters.get("Subject")

            if not user_named_subject and not selected_subject:
                if year_count >= 2:
                    selected_subject = (
                        choose_subject_with_greatest_growth(df)
                    )

                if selected_subject is None:
                    selected_subject = choose_ela_fallback(df)

                if selected_subject is not None:
                    filters["Subject"] = selected_subject

            # Use All Students when that exact subgroup value exists.
            if "Subgroup" in df.columns and "Subgroup" not in filters:
                subgroup_values = (
                    df["Subgroup"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                if "All Students" in subgroup_values:
                    filters["Subgroup"] = "All Students"

            spec["filters"] = filters

            existing_notes = str(spec.get("notes", "")).strip()

            if year_count < 2:
                # A one-year file cannot support growth claims. Shift the
                # broad default from a time trend to a current-status grade
                # comparison and use ELA when possible.
                fallback_subject = (
                    filters.get("Subject")
                    or choose_ela_fallback(df)
                )
                if fallback_subject is not None:
                    filters["Subject"] = fallback_subject
                    spec["filters"] = filters

                spec["chart_type"] = "bar"
                spec["special_mode"] = None
                spec["x"] = "Grade"
                spec["y"] = "Percent Proficient and Advanced"
                spec["group"] = None
                spec["row"] = None
                spec["col"] = None
                spec["aggregation"] = "mean"
                spec["sort_x"] = "ascending"

                note = (
                    "Only one school year is available, so growth cannot be "
                    "calculated. VizCreate defaulted to a current-status "
                    "proficiency comparison across grade levels."
                )
                spec["notes"] = (
                    f"{existing_notes} {note}".strip()
                )

            elif (
                not user_named_subject
                and filters.get("Subject") is not None
            ):
                note = (
                    f"For this broad overview, VizCreate selected "
                    f"{filters['Subject']} because it showed the greatest "
                    f"earliest-to-latest increase in proficiency among the "
                    f"available subjects."
                )
                spec["notes"] = (
                    f"{existing_notes} {note}".strip()
                )

        return spec

    except Exception as e:
        if "insufficient_quota" in str(e) or "quota" in str(e):
           st.warning("VizCreate AI is temporarily unavailable due to API usage limits. "
        "You can still create charts using the manual configuration below."
           )
        else:
            st.error(f"VizCreate agent error: {e}")


        # Fallback so the rest of the app keeps working
        return {
            "chart_type": "bar",
            "special_mode": None,
            "x": None,
            "y": None,
            "group": None,
            "row": None,
            "col": None,
            "filters": {},
            "aggregation": "mean",
            "sort_x": "none",
            "facets": None,
            "notes": "Fallback spec returned due to an error in the agent.",
        }

def _format_chart_value(value, column_name: Optional[str] = None) -> str:
    """Format a chart value for the Read This Chart section."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    column_text = str(column_name or "").lower()
    if "percent" in column_text:
        return f"{numeric_value:.1f}%"
    if abs(numeric_value - round(numeric_value)) < 1e-9:
        return f"{int(round(numeric_value)):,}"
    return f"{numeric_value:,.1f}"


def generate_read_this_chart(
    df: pd.DataFrame,
    spec: dict,
    ui_filters: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Build a concise, deterministic chart description and a few data-based
    observations using the same filters that generated the visualization.
    """
    if not spec:
        return "This visualization summarizes the selected data.", []

    data = apply_spec(df, spec, ui_filters=ui_filters)
    if data.empty:
        return "This visualization contains no data after the selected filters were applied.", []

    chart_type_raw = str(spec.get("chart_type", "")).strip().lower()
    chart_names = {
        "bar": "bar chart",
        "stacked_bar": "stacked bar chart",
        "line": "line chart",
        "heatmap": "heatmap",
        "box": "box-and-whisker plot",
        "scatter": "scatterplot",
    }
    chart_name = chart_names.get(chart_type_raw, "chart")

    special_mode = spec.get("special_mode")
    x_col = spec.get("x")
    y_col = spec.get("y")
    group_col = spec.get("group")
    row_col = spec.get("row")
    col_col = spec.get("col")
    aggregation = spec.get("aggregation", "mean")
    use_agg = "mean" if aggregation in ["none", None] else aggregation

    # Normalize the same common heatmap structure handled by the plotter.
    if chart_type_raw == "heatmap":
        if row_col is None and group_col is not None:
            row_col = group_col
        if col_col is None and x_col is not None:
            col_col = x_col

    # Match the plotter's multi-grade behavior for line and bar charts.
    grade_dim = next(
        (column for column in ["Grade", "Grade Level", "Testing Grade"] if column in data.columns),
        None,
    )
    if (
        ui_filters
        and grade_dim
        and x_col in ["School Year", "Year"]
        and chart_type_raw in ["bar", "line", "stacked_bar"]
    ):
        selected_grades = ui_filters.get(grade_dim)
        if isinstance(selected_grades, list) and len(selected_grades) > 1:
            if group_col is None or group_col == grade_dim:
                group_col = grade_dim

    filters = dict(spec.get("filters", {}) or {})
    if ui_filters:
        filters.update(ui_filters)

    filter_parts = []
    for column, value in filters.items():
        if isinstance(value, list):
            if len(value) == 0:
                continue
            displayed = ", ".join(str(item) for item in value)
        else:
            displayed = str(value)
        filter_parts.append(f"{column}: {displayed}")

    filter_phrase = ""
    if filter_parts:
        filter_phrase = " The displayed data are filtered to " + "; ".join(filter_parts) + "."

    notices: list[str] = []

    # -----------------------------------------------------
    # Wide-format Likert construct summary
    # -----------------------------------------------------
    if special_mode == "likert_construct_summary":
        item_columns = [
            column
            for column in spec.get("item_columns", [])
            if column in data.columns
        ]
        if item_columns:
            numeric_items = data[item_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            means = numeric_items.mean(axis=0, skipna=True).dropna()
            description = (
                "This bar chart compares descriptive mean ratings across "
                f"{len(means)} survey constructs. Because Likert responses are "
                "ordinal, the ranking should be interpreted alongside response distributions."
                + filter_phrase
            )
            if not means.empty:
                highest_column = means.idxmax()
                lowest_column = means.idxmin()
                highest_label = re.sub(
                    r"^(Q|Item|Question)\d+[_\-\s]*",
                    "",
                    str(highest_column),
                    flags=re.IGNORECASE,
                ).replace("_", " ").replace("-", " ")
                lowest_label = re.sub(
                    r"^(Q|Item|Question)\d+[_\-\s]*",
                    "",
                    str(lowest_column),
                    flags=re.IGNORECASE,
                ).replace("_", " ").replace("-", " ")
                notices.append(
                    f"{highest_label.title()} has the highest mean rating at "
                    f"{means.loc[highest_column]:.2f}."
                )
                if len(means) > 1:
                    notices.append(
                        f"{lowest_label.title()} has the lowest mean rating at "
                        f"{means.loc[lowest_column]:.2f}."
                    )
            return description, notices[:3]

    # -----------------------------------------------------
    # Special WYTOPP stacked chart
    # -----------------------------------------------------
    if chart_type_raw == "stacked_bar" and special_mode == "wytopp_stacked":
        basic_col = "Percent Basic and Below"
        prof_col = "Percent Proficient and Advanced"
        description = (
            f"This stacked bar chart compares {basic_col} with {prof_col}"
            + (f" across {x_col}." if x_col else ".")
            + filter_phrase
        )

        if x_col in data.columns and prof_col in data.columns:
            grouped = data.groupby(x_col)[[basic_col, prof_col]].mean().reset_index()
            grouped = grouped.dropna(subset=[prof_col])
            if not grouped.empty:
                highest = grouped.loc[grouped[prof_col].idxmax()]
                lowest = grouped.loc[grouped[prof_col].idxmin()]
                notices.append(
                    f"The highest proficient-and-advanced value is "
                    f"{_format_chart_value(highest[prof_col], prof_col)} for {highest[x_col]}."
                )
                if len(grouped) > 1:
                    notices.append(
                        f"The lowest proficient-and-advanced value is "
                        f"{_format_chart_value(lowest[prof_col], prof_col)} for {lowest[x_col]}."
                    )
        return description, notices[:3]

    if (
        chart_type_raw == "bar"
        and spec.get("special_mode") == "multi_measure_bar"
        and x_col
        and spec.get("item_columns")
    ):
        item_columns = [
            column
            for column in spec.get("item_columns", [])
            if column in data.columns
        ]
        metadata = spec.get("measure_scope_metadata", {}) or {}
        subject_name = str(metadata.get("subject_id") or "subject").upper()
        description = (
            f"This grouped bar chart compares mean scores for "
            f"{len(item_columns)} {subject_name} category or subscore measures "
            f"across {x_col}."
            + filter_phrase
        )

        if item_columns:
            working = data[[x_col] + item_columns].copy()
            for column in item_columns:
                working[column] = pd.to_numeric(
                    working[column],
                    errors="coerce",
                )
            long_df = working.melt(
                id_vars=[x_col],
                value_vars=item_columns,
                var_name="Category",
                value_name="Score",
            ).dropna(subset=["Score"])
            if not long_df.empty:
                means = (
                    long_df.groupby(["Category", x_col])["Score"]
                    .mean()
                    .reset_index()
                )
                highest = means.loc[means["Score"].idxmax()]
                notices.append(
                    f"The highest displayed category mean is "
                    f"{_format_chart_value(highest['Score'], str(highest['Category']))} "
                    f"for {highest[x_col]} in {highest['Category']}."
                )
        return description, notices[:3]

    # -----------------------------------------------------
    # Line, bar, and ordinary stacked bar charts
    # -----------------------------------------------------
    if chart_type_raw in ["line", "bar", "stacked_bar"] and x_col and y_col:
        if chart_type_raw == "line":
            description = (
                f"This line chart displays {use_agg} {y_col} across {x_col}."
            )
            if group_col:
                description += f" Each line represents {group_col}."
        elif chart_type_raw == "bar":
            description = f"This bar chart compares {use_agg} {y_col} across {x_col}."
            if group_col:
                description += f" Bars are grouped by {group_col}."
        else:
            description = f"This stacked bar chart displays {use_agg} {y_col} across {x_col}."
            if group_col:
                description += f" Each stack represents {group_col}."
        description += filter_phrase

        group_columns = [x_col] + ([group_col] if group_col else [])
        if use_agg == "sum":
            grouped = data.groupby(group_columns)[y_col].sum().reset_index()
        elif use_agg == "count":
            grouped = data.groupby(group_columns)[y_col].count().reset_index()
        else:
            grouped = data.groupby(group_columns)[y_col].mean().reset_index()
        grouped = grouped.dropna(subset=[y_col])

        if not grouped.empty:
            if chart_type_raw == "line":
                if group_col:
                    changes = []
                    latest_rows = []
                    for group_value, subset in grouped.groupby(group_col):
                        subset = subset.sort_values(x_col)
                        if subset.empty:
                            continue
                        latest_rows.append(subset.iloc[-1])
                        if len(subset) >= 2:
                            change = float(subset.iloc[-1][y_col]) - float(subset.iloc[0][y_col])
                            changes.append((group_value, change, subset.iloc[0], subset.iloc[-1]))

                    if changes:
                        greatest = max(changes, key=lambda item: item[1])
                        direction = "increased" if greatest[1] >= 0 else "decreased"
                        notices.append(
                            f"{greatest[0]} shows the largest change, having {direction} by "
                            f"{_format_chart_value(abs(greatest[1]), y_col)} from "
                            f"{greatest[2][x_col]} to {greatest[3][x_col]}."
                        )

                        declines = [item for item in changes if item[1] < 0]
                        if declines:
                            largest_decline = min(declines, key=lambda item: item[1])
                            if largest_decline[0] != greatest[0]:
                                notices.append(
                                    f"{largest_decline[0]} shows the largest decline: "
                                    f"{_format_chart_value(abs(largest_decline[1]), y_col)}."
                                )

                    if latest_rows:
                        latest_df = pd.DataFrame(latest_rows)
                        highest_latest = latest_df.loc[latest_df[y_col].idxmax()]
                        notices.append(
                            f"At the latest displayed {x_col}, {highest_latest[group_col]} has the "
                            f"highest value at {_format_chart_value(highest_latest[y_col], y_col)}."
                        )
                else:
                    grouped = grouped.sort_values(x_col)
                    if len(grouped) >= 2:
                        first = grouped.iloc[0]
                        last = grouped.iloc[-1]
                        change = float(last[y_col]) - float(first[y_col])
                        direction = "increased" if change >= 0 else "decreased"
                        notices.append(
                            f"The value {direction} by {_format_chart_value(abs(change), y_col)} "
                            f"from {first[x_col]} to {last[x_col]}."
                        )
            else:
                highest = grouped.loc[grouped[y_col].idxmax()]
                lowest = grouped.loc[grouped[y_col].idxmin()]
                high_label = str(highest[x_col])
                low_label = str(lowest[x_col])
                if group_col:
                    high_label += f" / {highest[group_col]}"
                    low_label += f" / {lowest[group_col]}"
                notices.append(
                    f"The highest displayed value is {_format_chart_value(highest[y_col], y_col)} "
                    f"for {high_label}."
                )
                if len(grouped) > 1:
                    notices.append(
                        f"The lowest displayed value is {_format_chart_value(lowest[y_col], y_col)} "
                        f"for {low_label}."
                    )

        return description, notices[:3]

    # -----------------------------------------------------
    # Heatmaps
    # -----------------------------------------------------
    if chart_type_raw == "heatmap" and row_col and col_col and y_col:
        description = (
            f"This heatmap shows {use_agg} {y_col} for combinations of "
            f"{row_col} and {col_col}. Darker and lighter cells indicate relative differences."
            + filter_phrase
        )

        if use_agg == "sum":
            grouped = data.groupby([row_col, col_col])[y_col].sum().reset_index()
        else:
            grouped = data.groupby([row_col, col_col])[y_col].mean().reset_index()
        grouped = grouped.dropna(subset=[y_col])

        if not grouped.empty:
            highest = grouped.loc[grouped[y_col].idxmax()]
            lowest = grouped.loc[grouped[y_col].idxmin()]
            notices.append(
                f"The highest cell is {highest[row_col]} × {highest[col_col]} at "
                f"{_format_chart_value(highest[y_col], y_col)}."
            )
            if len(grouped) > 1:
                notices.append(
                    f"The lowest cell is {lowest[row_col]} × {lowest[col_col]} at "
                    f"{_format_chart_value(lowest[y_col], y_col)}."
                )
        return description, notices[:3]

    # -----------------------------------------------------
    # Scatterplots
    # -----------------------------------------------------
    if chart_type_raw == "scatter" and x_col and y_col:
        description = (
            f"This scatterplot examines the row-level relationship between "
            f"{x_col} and {y_col}. Each point represents one complete paired observation."
            + filter_phrase
        )
        paired = data[[x_col, y_col]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(paired) >= 2:
            correlation = paired[x_col].corr(paired[y_col])
            if pd.notna(correlation):
                magnitude = abs(float(correlation))
                if magnitude >= 0.70:
                    strength = "strong"
                elif magnitude >= 0.40:
                    strength = "moderate"
                elif magnitude >= 0.20:
                    strength = "weak"
                else:
                    strength = "very weak"
                direction = "positive" if correlation >= 0 else "negative"
                notices.append(
                    f"The Pearson correlation is {correlation:.2f}, indicating a "
                    f"{strength} {direction} linear relationship in the displayed data."
                )
                notices.append(
                    "Correlation describes association and does not establish causation."
                )
        return description, notices[:3]

    # -----------------------------------------------------
    # Box plots
    # -----------------------------------------------------
    if (
        chart_type_raw == "box"
        and x_col
        and not y_col
        and spec.get("item_columns")
    ):
        item_columns = [
            column for column in spec.get("item_columns", [])
            if column in data.columns
        ]
        description = (
            f"This grouped box-and-whisker plot compares score distributions for "
            f"{len(item_columns)} math categories across {x_col}. "
            "Each box shows the median, middle half of scores, and broader spread."
            + filter_phrase
        )
        if item_columns:
            numeric = data[[x_col] + item_columns].copy()
            for column in item_columns:
                numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
            long_df = numeric.melt(
                id_vars=[x_col],
                value_vars=item_columns,
                var_name="Math Category",
                value_name="Score",
            ).dropna(subset=["Score"])
            if not long_df.empty:
                medians = (
                    long_df.groupby(["Math Category", x_col])["Score"]
                    .median()
                    .reset_index()
                )
                highest = medians.loc[medians["Score"].idxmax()]
                notices.append(
                    f"The highest displayed median is "
                    f"{_format_chart_value(highest['Score'], str(highest['Math Category']))} "
                    f"for {highest[x_col]} in {highest['Math Category']}."
                )
        return description, notices[:3]

    if chart_type_raw == "box" and x_col and y_col:
        description = (
            f"This box-and-whisker plot displays the distribution of {y_col} across {x_col}. "
            "The center line marks the median, the box shows the middle half of values, "
            "and the whiskers show the broader spread."
            + filter_phrase
        )
        grouped = data[[x_col, y_col]].dropna()
        if not grouped.empty:
            medians = grouped.groupby(x_col)[y_col].median()
            highest_group = medians.idxmax()
            lowest_group = medians.idxmin()
            notices.append(
                f"{highest_group} has the highest median at "
                f"{_format_chart_value(medians.loc[highest_group], y_col)}."
            )
            if len(medians) > 1:
                notices.append(
                    f"{lowest_group} has the lowest median at "
                    f"{_format_chart_value(medians.loc[lowest_group], y_col)}."
                )
        return description, notices[:3]

    return (
        f"This {chart_name} summarizes the selected data using the requested fields."
        + filter_phrase,
        notices,
    )





def render_confidence_badge(confidence: str) -> None:
    """Render a small confidence badge for an Evidence Engine finding."""
    styles = {
        "high": ("High confidence", "#e6f4ea", "#246b3c", "#b7dfc2"),
        "moderate": ("Moderate confidence", "#fff5d6", "#7a5a00", "#ead58c"),
        "preliminary": ("Preliminary observation", "#f1f3f5", "#56616a", "#d5dadd"),
    }
    label, background, text_color, border = styles.get(confidence, styles["preliminary"])
    st.markdown(
        f"""
        <span style="
            display:inline-block; padding:0.18rem 0.55rem; border-radius:999px;
            background:{background}; color:{text_color}; border:1px solid {border};
            font-size:0.78rem; font-weight:650; line-height:1.2; margin-bottom:0.35rem;">
            {label}
        </span>
        """,
        unsafe_allow_html=True,
    )


# -------------------------
# Streamlit page settings & CSS
# -------------------------
st.set_page_config(page_title="VizCreate", layout="wide")
initialize_session_state()

# Light blue-green background and big buttons
st.markdown(
    """
    <style>
    body, .main {
        background-color: #e7f6f7 !important;  /* soft blue-green */
    }

    /* Make all st.button elements larger */
    /* Keep the main visualization action visually dominant */
    [class*="st-key-use_ai_button"] button {
        background: #C69214 !important;
        color: #17232A !important;
        border: 2px solid #8A650B !important;
        box-shadow: 0 5px 14px rgba(138, 101, 11, 0.28) !important;
        font-weight: 800 !important;
    }
    [class*="st-key-use_ai_button"] button:hover {
        background: #D9AA2B !important;
        border-color: #6F5008 !important;
        transform: translateY(-1px);
    }

    .stButton > button {
        padding-top: 24px;
        padding-bottom: 24px;
        font-size: 20px !important;
        border-radius: 14px;
        font-weight: 600;
    }

    .vc-session-card {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid rgba(31, 119, 180, 0.22);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(30, 70, 80, 0.08);
        min-height: 126px;
    }
    .vc-session-heading {
        font-size: 0.95rem;
        font-weight: 700;
        margin-bottom: 8px;
        color: #234b55;
    }
    .vc-session-family {
        font-weight: 700;
        line-height: 1.25;
        color: #1f5f4a;
    }
    .vc-session-file {
        color: #66777b;
        font-size: 0.78rem;
        margin: 3px 0 9px 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .vc-session-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-bottom: 8px;
    }
    .vc-session-grid div {
        background: #eef8f8;
        border-radius: 8px;
        padding: 6px 8px;
        display: flex;
        justify-content: space-between;
        gap: 8px;
    }
    .vc-session-grid span,
    .vc-session-detail span,
    .vc-session-health span,
    .vc-session-analysis span {
        color: #66777b;
        font-size: 0.76rem;
    }
    .vc-session-grid strong,
    .vc-session-detail strong,
    .vc-session-health strong,
    .vc-session-analysis strong {
        color: #24383d;
        font-size: 0.78rem;
        text-align: right;
    }
    .vc-session-detail,
    .vc-session-health,
    .vc-session-analysis {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        padding: 3px 0;
        border-top: 1px solid rgba(35, 75, 85, 0.08);
    }
    .vc-session-health {
        margin-top: 5px;
        padding-top: 7px;
    }
    .vc-session-health strong {
        font-size: 0.76rem;
    }
    .vc-health-ready strong {
        color: #26734d;
    }
    .vc-health-warning strong {
        color: #9a6700;
    }
    .vc-health-error strong {
        color: #b42318;
    }
    .vc-session-analysis {
        margin-top: 3px;
        padding-top: 7px;
    }
    .vc-session-filters {
        color: #66777b;
        font-size: 0.72rem;
        line-height: 1.35;
        margin-top: 5px;
    }
    .vc-session-empty {
        font-weight: 650;
        color: #4d646a;
        padding-top: 9px;
    }
    .vc-session-muted {
        color: #788a8e;
        font-size: 0.78rem;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="text-align:center; margin-bottom:1.15rem;">
        <h1 style="margin-bottom:0.1rem;">VizCreate</h1>
        <div style="font-size:1.16rem; color:#4b6475; font-weight:500;">
            Helping educators think with data.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#555;'>Describe the visualization you want, "
    "then refine it with the controls below.</p>",
    unsafe_allow_html=True,
)

workspace_col, session_col = st.columns([2.8, 1.2], gap="large")
with workspace_col:
    if st.button("＋ New Analysis", key="new_analysis_button", use_container_width=False):
        start_new_analysis()
        st.rerun()
    st.caption(
        "Start fresh at any time. VizCreate clears the uploaded file, prompt, chart, "
        "filters, and display overrides while keeping your preferred color scheme."
    )
with session_col:
    analysis_session_placeholder = st.empty()
    analysis_session_placeholder.markdown(
        analysis_session_card_html(),
        unsafe_allow_html=True,
    )

# -------------------------
# 1. Chart-type gallery (always visible)
# -------------------------
st.subheader("1. Choose a chart style (optional)")

def choose_chart(chart_type: str, example_prompt: str):
    st.session_state.chart_type = chart_type
    st.session_state.prompt = example_prompt
    st.session_state.prompt_input = example_prompt


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊\nBar Chart", use_container_width=True, key="tile_bar"):
        choose_chart(
            "Bar",
            "Create a bar chart showing the average of a numeric score by a category "
            "(e.g., average Math Scale Score by Grade).",
        )

with col2:
    if st.button("🧱\nStacked Bar", use_container_width=True, key="tile_stacked_bar"):
        choose_chart(
            "Stacked Bar",
            "Create a stacked bar chart showing Basic & Below vs Proficient & Advanced by Grade.",
        )

with col3:
    if st.button("📈\nLine Chart", use_container_width=True, key="tile_line"):
        choose_chart(
            "Line",
            "Create a line chart showing a score over time (e.g., average Math Scale Score by School Year).",
        )

with col4:
    if st.button("🟩\nHeatmap", use_container_width=True, key="tile_heatmap"):
        choose_chart(
            "Heatmap",
            "Create a heatmap where rows are grades, columns are tests or subscores, and colors show the average score.",
        )

with col5:
    if st.button("📦\nBox Plot", use_container_width=True, key="tile_box"):
        choose_chart(
            "Box",
            "Create a box-and-whisker plot showing the distribution of a score by grade or subgroup.",
        )

# -------------------------
# 2. Prompt (always visible)
# -------------------------
st.subheader("2. Describe your visualization")

role_selection = st.selectbox(
    "Your perspective for this analysis",
    options=ROLE_OPTIONS,
    index=0,
    help=(
        "Choose a role to tailor the analysis recommendation and interpretation. "
        "Auto-detect uses role language in the prompt when present."
    ),
    key="analysis_role_selection",
)

if st.session_state.get("pending_dive_prompt"):
    st.session_state["prompt_input"] = st.session_state.pending_dive_prompt
    st.session_state.pending_dive_prompt = None

prompt = st.text_area(
    "Prompt",
    key="prompt_input",
    height=100,
    placeholder=(
        "Example: Create a stacked bar chart showing Percent Basic and Below vs "
        "Percent Proficient and Advanced by Grade."
    ),
)
st.session_state.prompt = prompt

st.markdown(
    "<p style='color:#777; font-size:0.9em;'>"
    "Describe the educational question in your own words. VizCreate will plan the "
    "analysis, and you can still refine the result through advanced controls."
    "</p>",
    unsafe_allow_html=True,
)

# -------------------------
# 3. File upload
# -------------------------
st.subheader("3. Upload your data")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Use the same style of file you use for Vizit / VizitOverTime.",
    key=f"data_uploader_{st.session_state.uploader_version}",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to configure and generate a chart.")
    st.stop()

# A different uploaded file begins a clean analysis automatically.
if dataset_changed(uploaded_file):
    reset_analysis_state(preserve_palette=True)
    st.toast("New dataset detected. Previous analysis controls were cleared.")
    st.rerun()

# -------------------------
# 4. Once file is uploaded: read + preview
# -------------------------
try:
    df = load_dataframe(uploaded_file)
except ValueError as error:
    st.error(str(error))
    st.stop()

# Try to auto-detect the N/enrollment column
DEFAULT_N_COL = "Number of Students Tested" if "Number of Students Tested" in df.columns else None


st.subheader("4. Data preview")
st.dataframe(df.head())

profile_rankings = rank_dataset_profiles(df)
dataset_profile = profile_rankings[0]
family_name = legacy_family_name(dataset_profile)
family_guidance = profile_guidance(dataset_profile)

st.caption(
    f"Detected dataset profile: {dataset_profile.display_name} "
    f"({dataset_profile.confidence_percent}% confidence)"
)

data_health = assess_dataset_health(
    df,
    family_name=family_name,
)

session_summary = build_analysis_session_summary(
    df,
    filename=uploaded_file.name,
    family_name=family_name,
    spec=st.session_state.get("agent_spec"),
    health=data_health,
)
analysis_session_placeholder.markdown(
    analysis_session_card_html(session_summary),
    unsafe_allow_html=True,
)

st.markdown("#### Dataset Health")
if data_health["status"] == "ready":
    st.success(data_health["message"])
elif data_health["status"] == "warning":
    st.warning(data_health["message"])
else:
    st.error(data_health["message"])

health_counts = data_health["counts"]
health_col1, health_col2, health_col3 = st.columns(3)
health_col1.metric("Checks passed", health_counts["pass"])
health_col2.metric("Items to review", health_counts["warning"])
health_col3.metric("Critical issues", health_counts["error"])

with st.expander("View dataset health details"):
    for check in data_health["checks"]:
        icon = health_icon(check["level"])
        st.markdown(f"**{icon} {check['title']}**")
        st.caption(check["detail"])


render_dataset_intelligence(
    dataset_profile,
    alternatives=profile_rankings[1:],
)

analysis_intent = detect_analysis_intent(
    df,
    prompt,
    dataset_profile,
)
analysis_role = resolve_user_role(
    prompt,
    role_selection,
)
target_unit = detect_target_unit(
    df,
    prompt,
    dataset_profile,
)
decision_basis = detect_decision_basis(
    df,
    prompt,
    dataset_profile,
    target_unit,
)
analyst_plan = build_analyst_coach_plan(
    df,
    dataset_profile,
    analysis_intent,
    analysis_role,
    target_unit=target_unit,
    decision_basis=decision_basis,
)

st.markdown("#### Data Analyst Coach")
st.info(f"**{analyst_plan.headline}**\n\n{analyst_plan.recommendation}")
st.caption(analyst_plan.why_this_first)
st.markdown(
    f"**Requested unit:** {target_unit.display_name}  \n"
    f"**Decision basis:** {decision_basis.display_name}"
)
st.caption(decision_basis.rationale)

coach_col1, coach_col2 = st.columns(2)
with coach_col1:
    st.markdown("**What this analysis can answer**")
    for item in analyst_plan.can_answer:
        st.markdown(f"- {item}")
with coach_col2:
    st.markdown("**What it cannot establish**")
    for item in analyst_plan.cannot_answer:
        st.markdown(f"- {item}")

if analyst_plan.caution:
    st.warning(analyst_plan.caution)

with st.expander("View alternative analyses and suggested next questions"):
    st.markdown("**Candidate analyses**")
    for item in analyst_plan.alternatives:
        stars = "★" * item.priority + "☆" * (5 - item.priority)
        st.markdown(
            f"**{stars} {item.title}**  \n"
            f"{item.rationale}  \n"
            f"*Chart family: {item.chart_family}*"
        )
    st.markdown("**Suggested next questions**")
    for question in analyst_plan.next_questions:
        st.markdown(f"- {question}")

# -------------------------
# 5. Let VizCreate interpret the prompt (agent beta)
# -------------------------
st.subheader("5. Ask the VizCreate Planner")

use_ai = st.button(
    "✨ Plan and Create My Visualization",
    key="use_ai_button",
    type="primary",
    use_container_width=True,
)

if "agent_spec" not in st.session_state:
    st.session_state.agent_spec = None
if "planner_result" not in st.session_state:
    st.session_state.planner_result = None
if "investigation_state" not in st.session_state:
    st.session_state.investigation_state = InvestigationState()
if "pending_dive_prompt" not in st.session_state:
    st.session_state.pending_dive_prompt = None
if "summary_scope" not in st.session_state:
    st.session_state.summary_scope = "Current visualization"
if "summary_detail" not in st.session_state:
    st.session_state.summary_detail = "Concise"
if "communication_scope" not in st.session_state:
    st.session_state.communication_scope = "Current visualization"
if "intent_filter_result" not in st.session_state:
    st.session_state.intent_filter_result = None

if use_ai:
    st.session_state.planner_result = get_planner_result_from_llm(
        df,
        prompt,
        profile=dataset_profile,
        intent=analysis_intent,
        role=analysis_role,
        coach_plan=analyst_plan,
        target_unit=target_unit,
        decision_basis=decision_basis,
    )
    enriched_spec, intent_filter_result = enrich_spec_with_intent_filters(
        df,
        prompt,
        st.session_state.planner_result.final_spec,
    )
    st.session_state.planner_result.final_spec = enriched_spec

    # Keep the recommended candidate, final planner spec, and rendered chart
    # synchronized. This prevents a stale exploratory candidate from rendering
    # after Phase 2G selects a different winner.
    recommended_id = st.session_state.planner_result.recommended_analysis_id
    for candidate in st.session_state.planner_result.candidates:
        if candidate.analysis_id == recommended_id:
            candidate.spec = dict(enriched_spec)
            candidate.chart_type = str(enriched_spec.get("chart_type") or candidate.chart_type)
            candidate.special_mode = enriched_spec.get("special_mode")
            break

    st.session_state.agent_spec = dict(enriched_spec)
    st.session_state.intent_filter_result = intent_filter_result

    # Refresh the Analysis Session card immediately after a chart spec is generated.
    session_summary = build_analysis_session_summary(
        df,
        filename=uploaded_file.name,
        family_name=family_name,
        spec=st.session_state.agent_spec,
        health=data_health,
    )
    analysis_session_placeholder.markdown(
        analysis_session_card_html(session_summary),
        unsafe_allow_html=True,
    )

    planner_result = st.session_state.planner_result

    intent_filter_result = st.session_state.get("intent_filter_result")
    if intent_filter_result and intent_filter_result.applied:
        st.success(
            "**Applied analytical scope**  \n"
            f"{intent_filter_result.scope_text}  \n\n"
            f"Rows included: {intent_filter_result.filtered_rows:,} of "
            f"{intent_filter_result.original_rows:,}"
        )
        with st.expander("How VizCreate interpreted the requested scope"):
            for item in intent_filter_result.inferred:
                st.markdown(
                    f"- **{item.column} = {item.value}** — {item.reason} "
                    f"({int(round(item.confidence * 100))}% confidence)"
                )
            for note in intent_filter_result.rejected:
                st.warning(note)

    winner = next(
        (
            candidate
            for candidate in planner_result.candidates
            if candidate.analysis_id == planner_result.recommended_analysis_id
        ),
        planner_result.candidates[0],
    )

    st.success(
        f"Recommended: {winner.title} "
        f"({planner_result.confidence_percent}% confidence)"
    )
    st.caption(
        f"VizCreate understood the decision as: "
        f"{planner_result.educational_decision}. "
        f"Intent: {analysis_intent.display_name} · "
        f"Role: {analysis_role.display_name} · "
        f"Unit: {target_unit.display_name} · "
        f"Basis: {decision_basis.display_name}"
    )

    if planner_result.ambiguities:
        st.warning(
            "The planner noticed: "
            + " ".join(planner_result.ambiguities)
        )

    with st.expander("How VizCreate understood and planned this analysis"):
        st.markdown("**Interpreted request**")
        st.write(planner_result.interpreted_request)

        if planner_result.assumptions:
            st.markdown("**Assumptions used to proceed**")
            for assumption in planner_result.assumptions:
                st.markdown(f"- {assumption}")

        st.markdown("**Candidate visual analyses**")
        ranked_candidates = sorted(
            planner_result.candidates,
            key=lambda item: item.total_score,
            reverse=True,
        )
        for rank, candidate in enumerate(ranked_candidates, start=1):
            selected_text = " — selected" if candidate.analysis_id == planner_result.recommended_analysis_id else ""
            st.markdown(
                f"**{rank}. {candidate.title}{selected_text}**  \n"
                f"{candidate.decision_value}  \n"
                f"*Visualization: {candidate.chart_type}"
                f"{' / ' + candidate.special_mode if candidate.special_mode else ''} · "
                f"Direct answer: {candidate.direct_answer_score:.0f}/100 · "
                f"Statistical suitability: {candidate.statistical_suitability:.0f}/100 · "
                f"Overall recommendation: {candidate.total_score:.1f}/100*"
            )
            if candidate.limitations:
                st.caption("Limitation: " + " ".join(candidate.limitations))

        st.markdown("**Why this recommendation**")
        st.write(planner_result.confidence_reason)

        if planner_result.suggested_follow_up_questions:
            st.markdown("**Suggested next questions**")
            for question in planner_result.suggested_follow_up_questions:
                st.markdown(f"- {question}")

    with st.expander("Planner technical details"):
        st.write("Final chart spec:", st.session_state.agent_spec)
        st.write(
            "Candidate validation:",
            [
                {
                    "analysis_id": candidate.analysis_id,
                    "valid": candidate.is_valid,
                    "notes": candidate.validation_notes,
                    "direct_answer_score": candidate.direct_answer_score,
                    "directly_answers_question": candidate.directly_answers_question,
                    "statistical_suitability": candidate.statistical_suitability,
                    "deterministic_support_score": candidate.deterministic_score,
                    "overall_recommendation": candidate.total_score,
                }
                for candidate in planner_result.candidates
            ],
        )


# -------------------------
# Type inference
# -------------------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if c not in numeric_cols]

if not numeric_cols:
    st.error("No numeric columns detected. You’ll need at least one numeric column to plot.")
    st.stop()


spec = st.session_state.agent_spec

if spec:
    st.subheader("6. Visualization (from prompt)")

    # --- Advanced display options ---
    with st.expander("Advanced display options"):
        title_override = st.text_input(
            "Chart title (optional override)",
            value="",
            key="title_override",
        )
        x_label_override = st.text_input(
            "X-axis label (optional override)",
            value="",
            key="x_label_override",
        )
        y_label_override = st.text_input(
            "Y-axis label (optional override)",
            value="",
            key="y_label_override",
        )

        show_value_labels = st.checkbox(
            "Show value labels on chart (bars/lines)",
            value=True,
            key="show_value_labels",
        )

        n_col_name_input = st.text_input(
            "Name of N/enrollment column (optional, e.g., 'Number of Students Tested')",
            value=DEFAULT_N_COL or "",
            key="n_col_name_input",
        )
        color_scheme = st.selectbox(
            "Color scheme",
            options=palette_names(),
            index=0,
            key="color_scheme",
        )

    # Convert empty strings to None so the helper keeps defaults if not overridden
    title_arg = title_override.strip() or None
    x_label_arg = x_label_override.strip() or None
    y_label_arg = y_label_override.strip() or None
    n_col_arg = n_col_name_input.strip() or None
    color_scheme_arg = color_scheme


    # --- Quick filters (optional) ---
    st.markdown("**Quick filters (optional)**")
    st.caption(
        "These controls begin with the scope inferred from your prompt. "
        "Changing them manually will intentionally override that scope."
    )

    # We'll build a dictionary of UI filters to pass into make_chart_from_spec
    ui_filters = {}

    # Grade filter (if available)
    grade_col = None
    for candidate in ["Grade", "Grade Level", "Testing Grade"]:
        if candidate in df.columns:
            grade_col = candidate
            break

    if grade_col is not None:
        grade_vals = sorted(df[grade_col].dropna().unique())
        spec_grade_filter = (spec.get("filters") or {}).get(grade_col)
        if spec_grade_filter is None:
            default_grades = grade_vals
        elif isinstance(spec_grade_filter, list):
            default_grades = [
                value for value in spec_grade_filter if value in grade_vals
            ]
        else:
            default_grades = (
                [spec_grade_filter] if spec_grade_filter in grade_vals else grade_vals
            )

        grade_filter_signature = tuple(default_grades)
        if st.session_state.get("_grade_filter_signature") != grade_filter_signature:
            st.session_state.grade_filter = default_grades
            st.session_state._grade_filter_signature = grade_filter_signature

        selected_grades = st.multiselect(
            f"Filter by {grade_col}",
            options=grade_vals,
            default=default_grades,
            key="grade_filter",
        )
        if selected_grades:
            ui_filters[grade_col] = selected_grades

    # School Year filter (if available)
    year_col = None
    for candidate in ["School Year", "Year"]:
        if candidate in df.columns:
            year_col = candidate
            break

    if year_col is not None:
        year_vals = sorted(
            list(df[year_col].dropna().unique()),
            key=lambda value: str(value),
        )
        spec_year_filter = (spec.get("filters") or {}).get(year_col)
        if spec_year_filter is None:
            default_years = year_vals
        elif isinstance(spec_year_filter, list):
            default_years = [
                value for value in spec_year_filter if value in year_vals
            ]
        else:
            default_years = (
                [spec_year_filter] if spec_year_filter in year_vals else year_vals
            )

        year_filter_signature = tuple(default_years)
        if st.session_state.get("_year_filter_signature") != year_filter_signature:
            st.session_state.year_filter = default_years
            st.session_state._year_filter_signature = year_filter_signature

        selected_years = st.multiselect(
            f"Filter by {year_col}",
            options=year_vals,
            default=default_years,
            key="year_filter",
        )
        if selected_years:
            ui_filters[year_col] = selected_years

    # --- Generate the chart from the spec on EVERY rerun ---
    fig, msg = make_chart_from_spec(
        df,
        spec,
        title_override=title_arg,
        x_label_override=x_label_arg,
        y_label_override=y_label_arg,
        show_value_labels=show_value_labels,
        n_col_name=n_col_arg,
        ui_filters=ui_filters,
        color_palette=color_scheme_arg,
    )


    if fig is None:
        st.error(f"Could not generate chart from spec: {msg}")
    else:
        st.pyplot(fig)

        current_chart_buffer = io.BytesIO()
        fig.savefig(current_chart_buffer, format="png", bbox_inches="tight", dpi=180)
        current_chart_png = current_chart_buffer.getvalue()

        read_description, notice_items = generate_read_this_chart(
            df,
            spec,
            ui_filters=ui_filters,
        )

        st.markdown("### 💡 Insights")
        st.caption("Understand the visualization")
        st.write(read_description)

        if notice_items:
            with st.expander("Details to help read this visualization"):
                for notice in notice_items:
                    st.markdown(f"- {notice}")

        evidence_summary = generate_evidence_summary(
            df,
            spec,
            ui_filters=ui_filters,
            max_findings=3,
        )

        st.markdown("### 🔍 VizCreate Insights")
        st.caption("Patterns detected in your data")
        st.write(evidence_summary.overview)

        if evidence_summary.findings:
            for finding in evidence_summary.findings:
                with st.container(border=True):
                    render_confidence_badge(finding.confidence)
                    st.markdown(f"**Observation**  \n{finding.observation}")
                    st.markdown(f"**Why it may matter**  \n{finding.interpretation}")
                    if finding.confidence_reason:
                        st.caption(finding.confidence_reason)
        else:
            st.info(
                "No sufficiently distinct pattern was detected in this view. "
                "A different comparison, distribution, or trend may reveal more."
            )

        st.caption(
            "These observations describe patterns in the current filtered dataset. "
            "They should guide further investigation rather than serve as final conclusions."
        )

        current_scope_text = ""
        active_intent_filter = st.session_state.get("intent_filter_result")
        if active_intent_filter and active_intent_filter.applied:
            current_scope_text = active_intent_filter.scope_text

        st.session_state.investigation_state = record_completed_analysis(
            st.session_state.investigation_state,
            prompt=prompt,
            current_spec=spec,
            insights=read_description,
            evidence_summary=evidence_summary,
            scope_text=current_scope_text,
            image_png=current_chart_png,
        )

        current_completed_steps = completed_analysis_steps(
            st.session_state.investigation_state
        )
        current_step = next(
            (
                step for step in reversed(current_completed_steps)
                if step.prompt.strip().lower() == prompt.strip().lower()
            ),
            None,
        )
        if current_step is not None and not current_step.common_translation:
            translation_result = generate_translation(
                client,
                question=prompt,
                scope_text=current_scope_text,
                insights=read_description,
                evidence_overview=evidence_summary.overview,
                evidence_findings=[
                    finding.observation for finding in evidence_summary.findings
                ],
            )
            st.session_state.investigation_state = set_current_analysis_translation(
                st.session_state.investigation_state,
                prompt=prompt,
                translation={
                    "what_the_data_shows": translation_result.what_the_data_shows,
                    "what_this_may_mean": translation_result.what_this_may_mean,
                    "what_this_does_not_show": translation_result.what_this_does_not_show,
                    "combined_text": translation_result.combined_text,
                    "source": translation_result.source,
                },
            )

        dive_suggestions = generate_dive_deeper_suggestions(
            df=df,
            profile=dataset_profile,
            intent=analysis_intent,
            role=analysis_role,
            target_unit=target_unit,
            decision_basis=decision_basis,
            current_spec=spec,
            state=st.session_state.investigation_state,
            evidence_summary=evidence_summary,
            analytical_objective=getattr(
                st.session_state.planner_result,
                "analytical_objective",
                None,
            ),
            limit=4,
        )

        st.markdown("### 🌊 Dive Deeper")
        st.caption("Recommended next investigations")

        if dive_suggestions:
            for index, suggestion in enumerate(dive_suggestions):
                with st.container(border=True):
                    st.markdown(f"**{suggestion.title}**")
                    st.write(suggestion.rationale)
                    st.caption(
                        f"{suggestion.estimated_value_label} · "
                        f"Likely evidence: {suggestion.next_visualization_hint}"
                    )
                    if suggestion.priority_reasons:
                        with st.expander("Why this is a useful next step"):
                            for reason in suggestion.priority_reasons:
                                st.markdown(f"- {reason}")
                    if st.button(
                        f"Dive deeper: {suggestion.title}",
                        key=(
                            f"dive_deeper_{suggestion.investigation_id}_"
                            f"{len(st.session_state.investigation_state.history)}_{index}"
                        ),
                        use_container_width=True,
                    ):
                        st.session_state.investigation_state = add_investigation_step(
                            st.session_state.investigation_state,
                            suggestion,
                            spec,
                            insights=read_description,
                            evidence_summary=evidence_summary,
                        )
                        st.session_state.pending_dive_prompt = suggestion.prompt
                        st.session_state.agent_spec = None
                        st.session_state.planner_result = None
                        st.rerun()
        else:
            st.info(
                "**You have explored the strongest guided paths available for this view.**\n\n"
                "VizCreate’s Data Analyst Coach identified several additional ways to continue. "
                "These options are broader and may help you approach the evidence from a different perspective."
            )

            coach_bridge_suggestions = generate_coach_bridge_suggestions(
                coach_plan=analyst_plan,
                state=st.session_state.investigation_state,
                current_prompt=prompt,
                current_spec=spec,
                limit=5,
            )

            st.markdown("#### 🎓 Data Analyst Coach")
            st.caption("Additional ways to continue your investigation")

            if coach_bridge_suggestions:
                alternative_items = [
                    item for item in coach_bridge_suggestions
                    if item.suggestion_type == "alternative_analysis"
                ]
                question_items = [
                    item for item in coach_bridge_suggestions
                    if item.suggestion_type == "next_question"
                ]

                if alternative_items:
                    st.markdown("**Alternative Analyses**")
                    for index, item in enumerate(alternative_items):
                        with st.container(border=True):
                            st.markdown(f"**{item.title}**")
                            st.write(item.rationale)
                            if item.chart_family:
                                st.caption(f"Possible visualization: {item.chart_family}")
                            if st.button(
                                "Use this analysis",
                                key=(
                                    f"coach_bridge_analysis_{item.suggestion_id}_"
                                    f"{len(st.session_state.investigation_state.history)}_{index}"
                                ),
                                use_container_width=True,
                            ):
                                st.session_state.investigation_state = add_coach_investigation_step(
                                    st.session_state.investigation_state,
                                    item,
                                    spec,
                                    insights=read_description,
                                    evidence_summary=evidence_summary,
                                )
                                st.session_state.pending_dive_prompt = item.prompt
                                st.session_state.agent_spec = None
                                st.session_state.planner_result = None
                                st.rerun()

                if question_items:
                    st.markdown("**Suggested Next Questions**")
                    for index, item in enumerate(question_items):
                        with st.container(border=True):
                            st.markdown(f"**{item.title}?**")
                            st.write(item.rationale)
                            if st.button(
                                "Investigate this question",
                                key=(
                                    f"coach_bridge_question_{item.suggestion_id}_"
                                    f"{len(st.session_state.investigation_state.history)}_{index}"
                                ),
                                use_container_width=True,
                            ):
                                st.session_state.investigation_state = add_coach_investigation_step(
                                    st.session_state.investigation_state,
                                    item,
                                    spec,
                                    insights=read_description,
                                    evidence_summary=evidence_summary,
                                )
                                st.session_state.pending_dive_prompt = item.prompt
                                st.session_state.agent_spec = None
                                st.session_state.planner_result = None
                                st.rerun()
            else:
                st.caption(
                    "The Coach did not identify another distinct question supported by the available fields. "
                    "You can revise the main prompt to explore a different educational question."
                )

        completed_steps = completed_analysis_steps(
            st.session_state.investigation_state
        )
        if completed_steps:
            with st.expander(
                f"Investigation Trail ({len(completed_steps)} completed analysis"
                f"{'es' if len(completed_steps) != 1 else ''})"
            ):
                for step_number, step in enumerate(completed_steps, start=1):
                    st.markdown(f"**{step_number}. {step.prompt}**")
                    if step.scope_text:
                        st.caption(f"Scope: {step.scope_text}")
                    if step.evidence_overview:
                        st.write(step.evidence_overview)

        st.markdown("### 📋 Investigation Summary")
        st.caption("Synthesize what you have learned")

        summary_col1, summary_col2 = st.columns(2)
        with summary_col1:
            summary_scope = st.segmented_control(
                "Summary scope",
                options=["Current visualization", "Entire investigation"],
                key="summary_scope",
                default="Current visualization",
            )
            summary_scope = summary_scope or "Current visualization"
        with summary_col2:
            summary_detail = st.segmented_control(
                "Summary detail",
                options=["Concise", "Detailed"],
                key="summary_detail",
                default="Concise",
            )
            summary_detail = summary_detail or "Concise"

        summary_next_step = ""
        if dive_suggestions:
            summary_next_step = dive_suggestions[0].prompt
        else:
            available_coach_suggestions = generate_coach_bridge_suggestions(
                coach_plan=analyst_plan,
                state=st.session_state.investigation_state,
                current_prompt=prompt,
                current_spec=spec,
                limit=1,
            )
            if available_coach_suggestions:
                summary_next_step = available_coach_suggestions[0].prompt

        summary_scope = summary_scope or st.session_state.get(
            "summary_scope", "Current visualization"
        )
        summary_detail = summary_detail or st.session_state.get(
            "summary_detail", "Concise"
        )
        normalized_detail = str(summary_detail).lower()

        if summary_scope == "Entire investigation":
            investigation_summary = summarize_entire_investigation(
                state=st.session_state.investigation_state,
                current_prompt=prompt,
                current_spec=spec,
                read_description=read_description,
                evidence_summary=evidence_summary,
                next_step=summary_next_step,
                detail_level=normalized_detail,
            )
        else:
            investigation_summary = summarize_current_view(
                prompt=prompt,
                spec=spec,
                read_description=read_description,
                evidence_summary=evidence_summary,
                next_step=summary_next_step,
                detail_level=normalized_detail,
            )

        with st.container(border=True):
            render_confidence_badge(investigation_summary.confidence)
            st.write(investigation_summary.summary_text)

            if investigation_summary.recurring_patterns:
                st.markdown("**Patterns Across the Investigation**")
                for item in investigation_summary.recurring_patterns:
                    st.markdown(f"- {item}")

            if investigation_summary.important_differences:
                st.markdown("**Important Differences or View-Specific Findings**")
                for item in investigation_summary.important_differences:
                    st.markdown(f"- {item}")

            st.markdown("**What the Evidence Does Not Show**")
            for item in investigation_summary.limitations:
                st.markdown(f"- {item}")

            st.markdown("**Recommended Next Step**")
            st.write(investigation_summary.recommended_next_step)

            st.caption(
                f"Visualizations reviewed: {investigation_summary.visualizations_reviewed}"
            )

        st.markdown("### 🧭 Communication Preview")
        st.caption(
            "Organize the questions, visualizations, insights, and Common Language "
            "Translations you can reuse in your own presentation or report."
        )

        communication_scope = st.segmented_control(
            "Communication scope",
            options=["Current visualization", "Entire investigation"],
            key="communication_scope",
            default="Current visualization",
        )
        communication_scope = communication_scope or "Current visualization"

        if communication_scope == "Entire investigation":
            entire_summary_for_preview = summarize_entire_investigation(
                state=st.session_state.investigation_state,
                current_prompt=prompt,
                current_spec=spec,
                read_description=read_description,
                evidence_summary=evidence_summary,
                next_step=summary_next_step,
                detail_level="detailed",
            )
            communication_preview = entire_preview(
                state=st.session_state.investigation_state,
                overall_summary=entire_summary_for_preview,
            )
        else:
            communication_preview = current_preview(
                question=prompt,
                scope_text=current_scope_text,
                chart_type=str(spec.get("chart_type") or "visualization"),
                insights=read_description,
                evidence_overview=evidence_summary.overview,
                evidence_findings=[
                    finding.observation for finding in evidence_summary.findings
                ],
                image_png=current_chart_png,
            )

        if communication_scope == "Current visualization":
            active_steps = completed_analysis_steps(
                st.session_state.investigation_state
            )
            active_step = next(
                (
                    step for step in reversed(active_steps)
                    if step.prompt.strip().lower() == prompt.strip().lower()
                ),
                None,
            )
            if active_step and active_step.common_translation:
                communication_preview.items[0].translation_sections = dict(
                    active_step.common_translation
                )
                communication_preview.items[0].translation = (
                    active_step.common_translation.get("combined_text")
                    or communication_preview.items[0].translation
                )

        with st.container(border=True):
            st.markdown(f"#### {communication_preview.title}")
            st.write(communication_preview.introduction)

            for item_number, item in enumerate(
                communication_preview.items,
                start=1,
            ):
                if communication_preview.scope == "entire":
                    st.markdown(f"### Analysis {item_number}")
                st.markdown("**Question Asked**")
                st.write(item.question or "Question not recorded.")

                if item.scope_text:
                    st.markdown("**Applied Scope**")
                    st.write(item.scope_text)

                st.markdown("**Visualization**")
                if item.image_png:
                    st.image(item.image_png, use_container_width=True)
                else:
                    st.info(
                        "A visualization image was not stored for this earlier analysis. "
                        "New analyses created in Phase 5A will retain their PNG preview."
                    )

                st.markdown("**Insights Given**")
                if item.evidence_overview:
                    st.write(item.evidence_overview)
                elif item.insights:
                    st.write(item.insights)
                else:
                    st.write("No insight text was recorded.")

                if item.evidence_findings:
                    for finding in item.evidence_findings:
                        st.markdown(f"- {finding}")

                st.markdown("**Common Language Translation**")
                if item.translation_sections:
                    shows = item.translation_sections.get("what_the_data_shows")
                    may_mean = item.translation_sections.get("what_this_may_mean")
                    does_not = item.translation_sections.get("what_this_does_not_show")
                    if shows:
                        st.markdown("**What the data shows**")
                        st.write(shows)
                    if may_mean:
                        st.markdown("**What this may mean**")
                        st.write(may_mean)
                    if does_not:
                        st.markdown("**What this does not show**")
                        st.write(does_not)
                else:
                    st.write(item.translation)

                if item_number < len(communication_preview.items):
                    st.divider()

            if communication_preview.overall_translation:
                st.markdown("### Across-the-Investigation Translation")
                st.write(communication_preview.overall_translation)

            st.markdown("**Important Cautions**")
            for caution in communication_preview.limitations:
                st.markdown(f"- {caution}")

        # -------------------------
        # Phase 5C: concise export
        # -------------------------
        current_preview_item = communication_preview.items[0] if communication_preview.items else None
        communication_text_parts = []
        if current_preview_item is not None:
            sections = current_preview_item.translation_sections or {}
            for label, key in [
                ("What the data shows", "what_the_data_shows"),
                ("What this may mean", "what_this_may_mean"),
                ("What this does not show", "what_this_does_not_show"),
            ]:
                value = sections.get(key)
                if value:
                    communication_text_parts.append(f"{label}: {value}")
            if not communication_text_parts and current_preview_item.translation:
                communication_text_parts.append(current_preview_item.translation)

        analyst_summary = build_analyst_summary(
            question=prompt,
            chart_description=read_description,
            evidence_overview=evidence_summary.overview,
            evidence_findings=[
                finding.observation for finding in evidence_summary.findings
            ],
        )
        report_model = ReportModel(
            question=prompt,
            chart_png=current_chart_png,
            analyst_summary=analyst_summary,
            key_findings=[read_description, *notice_items],
            supporting_evidence=[
                evidence_summary.overview,
                *[finding.observation for finding in evidence_summary.findings],
            ],
            communication_preview="\n\n".join(communication_text_parts),
        )

        # Build report models for every completed analysis already stored in
        # investigation memory. The current analysis is replaced with the live
        # viewer model so its latest chart and text are always exported.
        investigation_models = []
        completed_steps = [
            step for step in st.session_state.investigation_state.history
            if getattr(step, "step_type", "completed_analysis") != "transition"
        ]
        for step in completed_steps:
            translation = getattr(step, "common_translation", {}) or {}
            translation_parts = []
            for label, key in [
                ("What the data shows", "what_the_data_shows"),
                ("What this may mean", "what_this_may_mean"),
                ("What this does not show", "what_this_does_not_show"),
            ]:
                value = translation.get(key)
                if value:
                    translation_parts.append(f"{label}: {value}")

            findings = list(getattr(step, "evidence_findings", []) or [])
            overview = str(getattr(step, "evidence_overview", "") or "")
            stored_insight = str(getattr(step, "insights", "") or "")
            stored_summary = build_analyst_summary(
                question=getattr(step, "prompt", ""),
                chart_description=stored_insight,
                evidence_overview=overview,
                evidence_findings=findings,
            )
            investigation_models.append(ReportModel(
                question=getattr(step, "prompt", ""),
                chart_png=getattr(step, "image_png", None) or b"",
                analyst_summary=stored_summary,
                key_findings=[item for item in [stored_insight, *findings[:2]] if item],
                supporting_evidence=[item for item in [overview, *findings] if item],
                communication_preview="\n\n".join(translation_parts),
            ))

        # Ensure the live view appears once and contains the freshest text.
        current_prompt_key = " ".join(str(prompt or "").lower().split())
        replaced_current = False
        for index in range(len(investigation_models) - 1, -1, -1):
            model_prompt_key = " ".join(str(investigation_models[index].question or "").lower().split())
            if model_prompt_key == current_prompt_key:
                investigation_models[index] = report_model
                replaced_current = True
                break
        if not replaced_current:
            investigation_models.append(report_model)

        st.markdown("### ⬇ Export")
        export_scope = st.radio(
            "Export scope",
            options=["Current View", "Entire Investigation"],
            horizontal=True,
            key="phase5c_export_scope",
        )
        if export_scope == "Entire Investigation":
            st.caption(
                f"Includes {len(investigation_models)} completed "
                f"analysis{'es' if len(investigation_models) != 1 else ''} in chronological order."
            )
            pdf_data = build_pdf_investigation(investigation_models)
            word_data = build_word_investigation(investigation_models)
            pdf_name = "vizcreate_entire_investigation.pdf"
            word_name = "vizcreate_entire_investigation.docx"
        else:
            pdf_data = build_pdf_report(report_model)
            word_data = build_word_report(report_model)
            pdf_name = "vizcreate_analysis_report.pdf"
            word_name = "vizcreate_analysis_report.docx"

        export_png_col, export_pdf_col, export_word_col = st.columns(3)
        with export_png_col:
            st.download_button(
                label="PNG",
                data=current_chart_png,
                file_name="vizcreate_chart.png",
                mime="image/png",
                key="download_png_phase5c",
                use_container_width=True,
                help="PNG always downloads the chart currently shown.",
            )
        with export_pdf_col:
            st.download_button(
                label="PDF Report",
                data=pdf_data,
                file_name=pdf_name,
                mime="application/pdf",
                key="download_pdf_phase5c",
                use_container_width=True,
            )
        with export_word_col:
            st.download_button(
                label="Word Report",
                data=word_data,
                file_name=word_name,
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                key="download_word_phase5c",
                use_container_width=True,
            )

        reset_col, _ = st.columns([1, 2])
        with reset_col:
            if st.button(
                "Start a New Investigation",
                key="reset_investigation_summary",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.investigation_state = InvestigationState()
                st.session_state.pending_dive_prompt = None
                st.session_state.agent_spec = None
                st.session_state.planner_result = None
                st.session_state.intent_filter_result = None
                st.session_state.summary_scope = "Current visualization"
                st.session_state.summary_detail = "Concise"
                st.session_state.communication_scope = "Current visualization"
                st.session_state.pop("_grade_filter_signature", None)
                st.session_state.pop("_year_filter_signature", None)
                st.session_state.pop("grade_filter", None)
                st.session_state.pop("year_filter", None)
                if "prompt" in st.session_state:
                    st.session_state.prompt = ""
                st.rerun()


# -------------------------
# 6. Configure your chart (manual controls) Expander added
# -------------------------
with st.expander("Advanced: manually configure the chart"):
    st.subheader("Configure your chart")

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Stacked Bar", "Line", "Heatmap", "Box"],
        index=["Bar", "Stacked Bar", "Line", "Heatmap", "Box"].index(
            st.session_state.chart_type
        ),
    )

# Defaults
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    group_col: Optional[str] = None
    row_col: Optional[str] = None
    col_col: Optional[str] = None
    agg_func: str = "mean"

    if chart_type in ["Bar", "Stacked Bar", "Line"]:
        st.markdown("**Axis & grouping**")

        x_col = st.selectbox(
            "X-axis (category or time)",
            options=categorical_cols + numeric_cols,
            index=0 if categorical_cols else 0,
        )

        y_col = st.selectbox(
            "Y-axis (numeric to summarize)",
            options=numeric_cols,
            index=0,
        )

        if chart_type == "Stacked Bar":
            group_col = st.selectbox(
                "Stack by (category)",
                options=[c for c in df.columns if c != x_col],
                index=0,
            )
        else:  # Bar / Line
            group_col = st.selectbox(
                "Optional grouping (color/series)",
                options=["(none)"] + [c for c in df.columns if c != x_col],
                index=0,
            )
            if group_col == "(none)":
                group_col = None

        agg_func = st.selectbox(
            "How to summarize Y",
            options=["mean", "sum", "count"],
            index=0,
        )

    elif chart_type == "Heatmap":
        st.markdown("**Heatmap layout**")
        row_col = st.selectbox("Rows (category)", options=categorical_cols, index=0)
        col_col = st.selectbox(
            "Columns (category)", options=[c for c in categorical_cols if c != row_col],
            index=0,
        )
        y_col = st.selectbox(
            "Color value (numeric)", options=numeric_cols, index=0
        )
        agg_func = st.selectbox(
            "How to summarize value",
            options=["mean", "sum"],
            index=0,
        )

    elif chart_type == "Box":
        st.markdown("**Box plot layout**")
        x_col = st.selectbox("Group by (category)", options=categorical_cols, index=0)
        y_col = st.selectbox("Value (numeric)", options=numeric_cols, index=0)

# -------------------------
# 7. Generate chart
# -------------------------
    generate = st.button("Generate Visualization", type="primary")

    fig = None

    if generate:
        if chart_type in ["Bar", "Stacked Bar", "Line"]:
            data = df.copy()

        # Aggregate
            group_cols = [x_col] + ([group_col] if group_col else [])
            if agg_func == "mean":
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()
            elif agg_func == "sum":
                grouped = data.groupby(group_cols)[y_col].sum().reset_index()
            elif agg_func == "count":
                grouped = data.groupby(group_cols)[y_col].count().reset_index()
            else:
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()

            fig, ax = plt.subplots(figsize=(8, 5))

            if chart_type == "Bar":
                if group_col:
                    categories = grouped[x_col].unique()
                    groups = grouped[group_col].unique()
                    x = np.arange(len(categories))
                    width = 0.8 / len(groups)

                    for i, g in enumerate(groups):
                        sub = grouped[grouped[group_col] == g]
                        heights = [
                            sub[sub[x_col] == cat][y_col].values[0]
                            if not sub[sub[x_col] == cat].empty
                            else 0
                            for cat in categories
                        ]
                        ax.bar(x + i * width, heights, width, label=str(g))
                    ax.set_xticks(x + width * (len(groups) - 1) / 2)
                    ax.set_xticklabels(categories, rotation=45, ha="right")
                    ax.legend()
                else:
                    ax.bar(grouped[x_col].astype(str), grouped[y_col])
                    ax.set_xticklabels(grouped[x_col].astype(str), rotation=45, ha="right")

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"{chart_type} of {y_col} by {x_col}")

            elif chart_type == "Stacked Bar":
                categories = grouped[x_col].unique()
                stacks = grouped[group_col].unique()
                x = np.arange(len(categories))
                bottom = np.zeros(len(categories))

                fig, ax = plt.subplots(figsize=(8, 5))
                for s in stacks:
                    sub = grouped[grouped[group_col] == s]
                    heights = [
                        sub[sub[x_col] == cat][y_col].values[0]
                        if not sub[sub[x_col] == cat].empty
                        else 0
                        for cat in categories
                    ]
                    ax.bar(x, heights, bottom=bottom, label=str(s))
                    bottom += np.array(heights)

                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha="right")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"Stacked Bar of {y_col} by {x_col} and {group_col}")
                ax.legend()

            elif chart_type == "Line":
                fig, ax = plt.subplots(figsize=(8, 5))
                if group_col:
                    for g, sub in grouped.groupby(group_col):
                        ax.plot(sub[x_col], sub[y_col], marker="o", label=str(g))
                    ax.legend()
                else:
                    ax.plot(grouped[x_col], grouped[y_col], marker="o")

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"Line chart of {y_col} by {x_col}")
                plt.xticks(rotation=45, ha="right")

        elif chart_type == "Heatmap":
            data = df.copy()
            if agg_func == "mean":
                grouped = data.groupby([row_col, col_col])[y_col].mean().reset_index()
            else:
                grouped = data.groupby([row_col, col_col])[y_col].sum().reset_index()

            pivot = grouped.pivot(index=row_col, columns=col_col, values=y_col)
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto")

            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index.astype(str))

            ax.set_xlabel(col_col)
            ax.set_ylabel(row_col)
            ax.set_title(f"Heatmap of {y_col} ({agg_func}) by {row_col} × {col_col}")
            cbar = fig.colorbar(im)
            cbar.set_label(y_col)

        elif chart_type == "Box":
            data = df[[x_col, y_col]].dropna()
            groups = [data[data[x_col] == g][y_col].values for g in data[x_col].unique()]
            labels = data[x_col].unique().astype(str)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Distribution of {y_col} by {x_col}")
            plt.xticks(rotation=45, ha="right")

    # Show figure + download
        if fig is not None:
            st.subheader("7. Visualization")
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="Download PNG",
                data=buf,
                file_name="vizcreate_chart.png",
                mime="image/png",
            )
