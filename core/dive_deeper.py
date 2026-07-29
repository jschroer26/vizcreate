"""Phase 3A: deterministic Dive Deeper investigation recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import hashlib
import json

import pandas as pd

from core.decision_context import DecisionBasisResult, TargetUnitResult
from core.intent_recognition import AnalysisIntentResult
from core.visualization_decision import AnalyticalObjective
from core.role_coach import UserRoleResult
from profiles.base_profile import DatasetProfileResult


@dataclass
class InvestigationStep:
    title: str
    prompt: str
    investigation_id: str
    chart_type: str | None = None
    special_mode: str | None = None
    unit_id: str | None = None
    insights: str = ""
    evidence_overview: str = ""
    evidence_findings: list[str] = field(default_factory=list)
    step_type: str = "completed_analysis"
    scope_text: str = ""
    image_png: bytes | None = None
    common_translation: dict[str, str] = field(default_factory=dict)


@dataclass
class InvestigationState:
    current_question: str = ""
    history: list[InvestigationStep] = field(default_factory=list)
    visualizations_seen: list[str] = field(default_factory=list)
    units_examined: list[str] = field(default_factory=list)
    intents_examined: list[str] = field(default_factory=list)

    def seen_ids(self) -> set[str]:
        return {step.investigation_id for step in self.history}


@dataclass
class DiveSuggestion:
    investigation_id: str
    title: str
    rationale: str
    prompt: str
    educational_value: float
    estimated_value_label: str
    next_visualization_hint: str
    applicable_roles: list[str] = field(default_factory=list)
    applicable_profiles: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    unit_id: str | None = None
    intent_id: str | None = None
    priority_reasons: list[str] = field(default_factory=list)


def _role_columns(profile: DatasetProfileResult, *role_names: str) -> list[str]:
    values: list[str] = []
    for role_name in role_names:
        raw = profile.detected_roles.get(role_name, "")
        values.extend(
            part.strip()
            for part in str(raw).split(",")
            if part.strip()
        )
    return list(dict.fromkeys(values))


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]


def _time_columns(df: pd.DataFrame, profile: DatasetProfileResult) -> list[str]:
    role_cols = _role_columns(
        profile,
        "time",
        "year",
        "school_year",
        "window",
        "date",
    )
    inferred = [
        str(column)
        for column in df.columns
        if any(
            token in str(column).lower()
            for token in ["year", "date", "window", "benchmark", "week", "month"]
        )
    ]
    return list(dict.fromkeys(role_cols + inferred))


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    existing = set(df.columns.astype(str))
    return all(column in existing for column in columns)


def _suggestion(
    investigation_id: str,
    title: str,
    rationale: str,
    prompt: str,
    value: float,
    viz_hint: str,
    *,
    roles: list[str] | None = None,
    profiles: list[str] | None = None,
    required: list[str] | None = None,
    unit_id: str | None = None,
    intent_id: str | None = None,
) -> DiveSuggestion:
    if value >= 90:
        label = "Very high value"
    elif value >= 78:
        label = "High value"
    elif value >= 65:
        label = "Useful next step"
    else:
        label = "Optional exploration"
    return DiveSuggestion(
        investigation_id=investigation_id,
        title=title,
        rationale=rationale,
        prompt=prompt,
        educational_value=value,
        estimated_value_label=label,
        next_visualization_hint=viz_hint,
        applicable_roles=roles or [],
        applicable_profiles=profiles or [],
        required_columns=required or [],
        unit_id=unit_id,
        intent_id=intent_id,
    )


def build_investigation_library(
    df: pd.DataFrame,
    profile: DatasetProfileResult,
) -> list[DiveSuggestion]:
    """Create profile-aware candidate investigations using exact dataset fields."""
    library: list[DiveSuggestion] = []
    student_cols = _role_columns(profile, "student", "student_id")
    grade_cols = _role_columns(profile, "grade")
    subject_cols = _role_columns(profile, "subject")
    school_cols = _role_columns(profile, "school")
    subgroup_cols = _role_columns(profile, "subgroup")
    measure_cols = [
        column
        for column in _role_columns(profile, "assessment_measures")
        if column in df.columns
    ]
    inferred_assessment_cols = []
    for column in df.columns:
        column_name = str(column)
        converted = pd.to_numeric(df[column], errors="coerce")
        numeric_ratio = float(converted.notna().mean()) if len(df) else 0.0
        assessment_name = any(
            token in column_name.lower()
            for token in [
                "score",
                "math",
                "reading",
                "ela",
                "science",
                "operation",
                "number",
                "fraction",
                "geometry",
                "measurement",
                "algebra",
                "statistics",
                "probability",
            ]
        )
        if (
            assessment_name
            and numeric_ratio >= 0.70
            and column_name not in student_cols
            and column_name not in grade_cols
        ):
            inferred_assessment_cols.append(column_name)
    measure_cols = list(dict.fromkeys(measure_cols + inferred_assessment_cols))
    time_cols = _time_columns(df, profile)
    numeric_cols = _numeric_columns(df)

    student_col = student_cols[0] if student_cols else None
    grade_col = grade_cols[0] if grade_cols else None
    subject_col = subject_cols[0] if subject_cols else None
    school_col = school_cols[0] if school_cols else None
    subgroup_col = subgroup_cols[0] if subgroup_cols else None
    time_col = time_cols[0] if time_cols else None

    if profile.profile_id in {"student_assessment", "cbm_progress_monitoring"}:
        if student_col and len(measure_cols) >= 2:
            library.append(_suggestion(
                "compare_student_profiles",
                "Compare Student Profiles",
                "Students with similar overall performance may have very different patterns across measures.",
                f"Compare individual {student_col} profiles across {', '.join(measure_cols[:4])}.",
                96,
                "Student heatmap or profile comparison",
                roles=["teacher", "coach", "assessment", "researcher"],
                profiles=[profile.profile_id],
                required=[student_col, *measure_cols[:2]],
                unit_id="student",
                intent_id="comparison",
            ))
        if len(measure_cols) >= 3:
            library.append(_suggestion(
                "investigate_skills",
                "Investigate Skills and Subscores",
                "A broad score can conceal the specific concepts contributing to a relative strength or need.",
                f"Compare {', '.join(measure_cols[:6])} to identify the strongest and weakest skill areas.",
                99,
                "Skill heatmap",
                roles=["teacher", "coach", "principal", "assessment", "researcher"],
                profiles=[profile.profile_id],
                required=measure_cols[:3],
                unit_id="skill",
                intent_id="strengths_needs",
            ))
        if time_col and student_col:
            library.append(_suggestion(
                "examine_growth",
                "Examine Growth",
                "Current performance and growth provide different evidence for instructional decisions.",
                f"Show change over {time_col} for each {student_col} using the available assessment measures.",
                95,
                "Growth line or slope view",
                roles=["teacher", "coach", "principal", "assessment", "researcher"],
                profiles=[profile.profile_id],
                required=[student_col, time_col],
                unit_id="student",
                intent_id="trend",
            ))
        if grade_col:
            library.append(_suggestion(
                "compare_grades",
                "Compare Grade-Level Patterns",
                "This checks whether the pattern is isolated to one grade or appears more broadly.",
                f"Compare the available assessment measures by {grade_col}.",
                79,
                "Grouped comparison or heatmap",
                roles=["teacher", "coach", "principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[grade_col],
                unit_id="grade",
                intent_id="comparison",
            ))
        if subgroup_col:
            library.append(_suggestion(
                "compare_subgroups",
                "Examine Subgroup Patterns",
                "Overall averages may conceal meaningful differences among student groups.",
                f"Compare performance and growth by {subgroup_col}.",
                88,
                "Grouped comparison",
                roles=["coach", "principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[subgroup_col],
                unit_id="subgroup",
                intent_id="comparison",
            ))
        if student_col and numeric_cols:
            library.append(_suggestion(
                "find_unusual_profiles",
                "Find Unusual Student Profiles",
                "Outliers and uneven patterns may warrant additional evidence or a closer instructional review.",
                f"Identify {student_col} records with unusual combinations across {', '.join((measure_cols or numeric_cols)[:4])}.",
                84,
                "Scatter or diagnostic view",
                roles=["teacher", "coach", "assessment", "researcher"],
                profiles=[profile.profile_id],
                required=[student_col],
                unit_id="student",
                intent_id="distribution",
            ))

    if profile.profile_id.startswith("wytopp"):
        if time_col:
            library.append(_suggestion(
                "persistent_patterns",
                "Find Persistent Strengths and Needs",
                "A sustained pattern carries more decision value than a single-year fluctuation.",
                f"Identify grades and subjects with persistent patterns across {time_col}.",
                97,
                "Longitudinal trend",
                roles=["principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[time_col],
                unit_id="grade_subject",
                intent_id="trend",
            ))
        if subject_col:
            library.append(_suggestion(
                "compare_subjects",
                "Compare Subjects",
                "Subject comparisons can distinguish a broad system pattern from a content-specific priority.",
                f"Compare performance across {subject_col}.",
                91,
                "Ranked comparison or heatmap",
                roles=["principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[subject_col],
                unit_id="subject",
                intent_id="comparison",
            ))
        if grade_col:
            library.append(_suggestion(
                "compare_grades",
                "Compare Grade Levels",
                "Grade-level comparisons help locate where a systemwide pattern begins or becomes most pronounced.",
                f"Compare performance by {grade_col}.",
                89,
                "Grade comparison",
                roles=["principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[grade_col],
                unit_id="grade",
                intent_id="comparison",
            ))
        if school_col:
            library.append(_suggestion(
                "compare_schools",
                "Compare Schools",
                "School comparisons can reveal whether a trend is widespread or concentrated.",
                f"Compare performance and trends by {school_col}.",
                92,
                "School comparison",
                roles=["superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[school_col],
                unit_id="school",
                intent_id="comparison",
            ))
        if subgroup_col:
            library.append(_suggestion(
                "compare_subgroups",
                "Examine Subgroup Patterns",
                "Aggregate results may hide persistent differences among student groups.",
                f"Compare proficiency and trend patterns by {subgroup_col}.",
                90,
                "Subgroup comparison",
                roles=["principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[subgroup_col],
                unit_id="subgroup",
                intent_id="comparison",
            ))

    if profile.profile_id == "likert_survey":
        item_cols = _role_columns(profile, "likert_items")
        if not item_cols:
            item_cols = [
                str(column)
                for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])
            ]
        if item_cols:
            library.extend([
                _suggestion(
                    "rank_constructs",
                    "Rank Survey Strengths and Needs",
                    "Ranking constructs or items identifies the most favorable and concerning areas.",
                    f"Rank the survey items or constructs using {', '.join(item_cols[:8])}.",
                    96,
                    "Ranked construct summary",
                    roles=["teacher", "coach", "principal", "superintendent", "assessment", "board", "researcher"],
                    profiles=[profile.profile_id],
                    required=item_cols[:1],
                    unit_id="construct",
                    intent_id="ranking",
                ),
                _suggestion(
                    "check_polarization",
                    "Check for Polarization",
                    "Similar averages can conceal very different levels of agreement and disagreement.",
                    f"Examine response polarization and disagreement across {', '.join(item_cols[:8])}.",
                    94,
                    "Divergent Likert view",
                    roles=["coach", "principal", "superintendent", "assessment", "board", "researcher"],
                    profiles=[profile.profile_id],
                    required=item_cols[:1],
                    unit_id="survey_item",
                    intent_id="distribution",
                ),
                _suggestion(
                    "inspect_neutral_responses",
                    "Inspect Neutral and Missing Responses",
                    "High neutral or missing response rates may indicate uncertainty, limited experience, or unclear items.",
                    f"Compare neutral and missing response patterns across {', '.join(item_cols[:8])}.",
                    82,
                    "Response composition",
                    roles=["coach", "principal", "assessment", "researcher"],
                    profiles=[profile.profile_id],
                    required=item_cols[:1],
                    unit_id="survey_item",
                    intent_id="composition",
                ),
            ])
        if subgroup_col:
            library.append(_suggestion(
                "compare_survey_groups",
                "Compare Respondent Groups",
                "Group comparisons can reveal whether experiences differ across roles or populations.",
                f"Compare survey results by {subgroup_col}.",
                91,
                "Grouped Likert comparison",
                roles=["coach", "principal", "superintendent", "assessment", "board", "researcher"],
                profiles=[profile.profile_id],
                required=[subgroup_col],
                unit_id="subgroup",
                intent_id="comparison",
            ))

    # Generic, broadly valid investigations.
    if numeric_cols:
        library.append(_suggestion(
            "inspect_variation",
            "Inspect Variation and Outliers",
            "Averages alone can hide unusually high, low, or inconsistent observations.",
            f"Examine the distribution and outliers in {', '.join(numeric_cols[:4])}.",
            68,
            "Distribution view",
            roles=["teacher", "coach", "principal", "superintendent", "assessment", "board", "researcher"],
            profiles=[profile.profile_id],
            required=numeric_cols[:1],
            unit_id=None,
            intent_id="distribution",
        ))
    if len(numeric_cols) >= 2:
        library.append(_suggestion(
            "examine_relationships",
            "Examine Relationships Between Measures",
            "Relationships can reveal whether two measures tend to move together while avoiding causal claims.",
            f"Examine the relationship between {numeric_cols[0]} and {numeric_cols[1]}.",
            72,
            "Scatterplot",
            roles=["teacher", "coach", "principal", "superintendent", "assessment", "board", "researcher"],
            profiles=[profile.profile_id],
            required=numeric_cols[:2],
            unit_id=None,
            intent_id="relationship",
        ))

    return library


def score_dive_suggestions(
    suggestions: list[DiveSuggestion],
    *,
    df: pd.DataFrame,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
    current_spec: dict[str, Any],
    state: InvestigationState,
    evidence_summary: Any | None = None,
    analytical_objective: AnalyticalObjective | None = None,
) -> list[DiveSuggestion]:
    """Filter and rank investigations according to context and investigation history."""
    seen = state.seen_ids()
    current_chart = str(current_spec.get("chart_type") or "")
    current_mode = str(current_spec.get("special_mode") or "")
    ranked: list[DiveSuggestion] = []

    for original in suggestions:
        suggestion = DiveSuggestion(**{
            field_name: getattr(original, field_name)
            for field_name in original.__dataclass_fields__
        })
        score = float(suggestion.educational_value)
        if analytical_objective is not None:
            if suggestion.intent_id == analytical_objective.comparison_mode:
                score += 14
                suggestion.priority_reasons.append(
                    "Extends the analytical objective established by the original question."
                )
            objective_terms = {
                analytical_objective.objective_id,
                analytical_objective.statistic or "",
                analytical_objective.comparison_mode or "",
            }
            suggestion_text = f"{suggestion.title} {suggestion.rationale} {suggestion.prompt}".lower()
            if any(term and term.replace("_", " ") in suggestion_text for term in objective_terms):
                score += 10
        reasons: list[str] = []

        if suggestion.applicable_profiles and profile.profile_id not in suggestion.applicable_profiles:
            continue
        if suggestion.applicable_roles and role.role_id not in suggestion.applicable_roles:
            continue
        if not _has_columns(df, suggestion.required_columns):
            continue
        if suggestion.investigation_id in seen:
            continue

        if suggestion.unit_id and suggestion.unit_id != target_unit.unit_id:
            score += 5
            reasons.append("Moves the investigation to a complementary level of evidence.")
        if suggestion.unit_id == target_unit.unit_id:
            score -= 4
            reasons.append("Stays at the current unit and adds detail.")

        if suggestion.intent_id == intent.intent_id:
            score -= 3
        else:
            score += 4
            reasons.append("Adds a different analytical lens.")

        if current_mode == "student_support_map":
            if suggestion.investigation_id == "investigate_skills":
                score += 18
                reasons.append("Skill evidence is a strong next step after identifying student patterns.")
            if suggestion.investigation_id == "examine_growth":
                score += 14
                reasons.append("Growth distinguishes current need from improving performance.")
            if suggestion.investigation_id == "compare_grades":
                score -= 5

        if current_chart == "line":
            if suggestion.investigation_id in {"compare_subgroups", "compare_schools", "compare_subjects"}:
                score += 12
                reasons.append("Disaggregates the trend to locate where it is occurring.")
            if suggestion.intent_id == "trend":
                score -= 10

        if current_chart == "heatmap":
            if suggestion.investigation_id in {"examine_growth", "compare_subgroups", "find_unusual_profiles"}:
                score += 9
                reasons.append("Tests whether the heatmap pattern persists in another form of evidence.")

        if profile.profile_id == "likert_survey" and current_mode == "likert_construct_summary":
            if suggestion.investigation_id == "check_polarization":
                score += 17
                reasons.append("Checks whether construct averages conceal divided responses.")
            if suggestion.investigation_id == "compare_survey_groups":
                score += 12

        if decision_basis.basis_id == "distribution_relative":
            if suggestion.investigation_id in {"examine_growth", "investigate_skills", "compare_subgroups"}:
                score += 6
                reasons.append("Adds evidence beyond relative standing in the current sample.")

        if role.role_id == "teacher":
            if suggestion.investigation_id in {
                "investigate_skills",
                "examine_growth",
                "compare_student_profiles",
                "find_unusual_profiles",
            }:
                score += 9
        elif role.role_id == "coach":
            if suggestion.investigation_id in {
                "investigate_skills",
                "compare_grades",
                "compare_subgroups",
                "examine_growth",
            }:
                score += 8
        elif role.role_id == "principal":
            if suggestion.investigation_id in {
                "compare_grades",
                "compare_subjects",
                "compare_subgroups",
                "persistent_patterns",
            }:
                score += 9
        elif role.role_id == "superintendent":
            if suggestion.investigation_id in {
                "compare_schools",
                "compare_subjects",
                "persistent_patterns",
                "compare_subgroups",
            }:
                score += 10
        elif role.role_id == "board":
            if suggestion.unit_id == "student":
                continue
            if suggestion.investigation_id in {
                "persistent_patterns",
                "compare_schools",
                "compare_subjects",
                "compare_subgroups",
            }:
                score += 10
        elif role.role_id in {"assessment", "researcher"}:
            if suggestion.investigation_id in {
                "check_polarization",
                "inspect_variation",
                "examine_relationships",
                "compare_subgroups",
                "persistent_patterns",
            }:
                score += 7

        if evidence_summary is not None:
            hint = getattr(evidence_summary, "recommendation_hints", {}).get(
                suggestion.investigation_id
            )
            if hint:
                score += float(hint.get("boost", 0))
                evidence_reason = str(hint.get("reason", "")).strip()
                if evidence_reason:
                    reasons.insert(0, evidence_reason)
                focus_label = str(hint.get("focus_label", "")).strip()
                measure = str(hint.get("measure", "")).strip()

                if suggestion.investigation_id == "investigate_skills" and focus_label:
                    suggestion.title = f"Investigate {focus_label}"
                    suggestion.rationale = (
                        f"{focus_label} emerged as a comparatively low area in the current evidence. "
                        "A focused analysis can show whether this pattern is widespread or concentrated."
                    )
                    suggestion.prompt = (
                        f"Investigate {focus_label} in greater detail and compare it with the other "
                        "available skills or measures."
                    )
                elif suggestion.investigation_id in {
                    "compare_grades", "compare_subjects", "compare_schools", "compare_subgroups"
                } and focus_label:
                    suggestion.rationale = (
                        f"The current evidence highlights {focus_label}. A disaggregated comparison can "
                        "help determine whether the pattern is broad or localized."
                    )
                elif suggestion.investigation_id in {
                    "find_unusual_profiles", "inspect_variation"
                } and measure:
                    suggestion.title = f"Review Variation in {measure}"
                    suggestion.rationale = (
                        f"The Evidence Engine detected unusual variation in {measure}. "
                        "Reviewing the distribution can distinguish meaningful cases from data issues."
                    )
                    suggestion.prompt = f"Examine the distribution and unusual observations in {measure}."

        if suggestion.investigation_id in state.seen_ids():
            score -= 100
        if suggestion.intent_id in state.intents_examined:
            score -= 8
        if suggestion.unit_id in state.units_examined:
            score -= 5

        suggestion.educational_value = max(0.0, min(120.0, score))
        if suggestion.educational_value >= 90:
            suggestion.estimated_value_label = "Very high value"
        elif suggestion.educational_value >= 78:
            suggestion.estimated_value_label = "High value"
        elif suggestion.educational_value >= 65:
            suggestion.estimated_value_label = "Useful next step"
        else:
            suggestion.estimated_value_label = "Optional exploration"
        suggestion.priority_reasons = reasons
        ranked.append(suggestion)

    ranked.sort(key=lambda item: item.educational_value, reverse=True)
    return ranked


def generate_dive_deeper_suggestions(
    *,
    df: pd.DataFrame,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
    current_spec: dict[str, Any],
    state: InvestigationState,
    evidence_summary: Any | None = None,
    analytical_objective: AnalyticalObjective | None = None,
    limit: int = 4,
) -> list[DiveSuggestion]:
    library = build_investigation_library(df, profile)
    return score_dive_suggestions(
        library,
        df=df,
        profile=profile,
        intent=intent,
        role=role,
        target_unit=target_unit,
        decision_basis=decision_basis,
        current_spec=current_spec,
        state=state,
        evidence_summary=evidence_summary,
        analytical_objective=analytical_objective,
    )[:limit]



def _analysis_fingerprint(prompt: str, spec: dict[str, Any]) -> str:
    """Create a stable identity for one completed analytical view."""
    payload = {
        "prompt": str(prompt or "").strip().lower(),
        "chart_type": spec.get("chart_type"),
        "x": spec.get("x"),
        "y": spec.get("y"),
        "group": spec.get("group"),
        "facet": spec.get("facet"),
        "filters": spec.get("filters") or {},
        "special_mode": spec.get("special_mode"),
    }
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def record_completed_analysis(
    state: InvestigationState,
    *,
    prompt: str,
    current_spec: dict[str, Any],
    insights: str = "",
    evidence_summary: Any | None = None,
    scope_text: str = "",
    image_png: bytes | None = None,
) -> InvestigationState:
    """Upsert a completed chart into durable investigation memory.

    Streamlit reruns the page when controls change. The fingerprint prevents
    toggles, downloads, and other reruns from duplicating the same analysis,
    while a genuinely new prompt or analytical scope creates a new record.
    """
    fingerprint = _analysis_fingerprint(prompt, current_spec)
    investigation_id = f"analysis:{fingerprint}"
    findings = (
        [finding.observation for finding in getattr(evidence_summary, "findings", [])]
        if evidence_summary else []
    )
    overview = getattr(evidence_summary, "overview", "") if evidence_summary else ""

    existing = next(
        (
            step for step in state.history
            if step.investigation_id == investigation_id
            and getattr(step, "step_type", "") == "completed_analysis"
        ),
        None,
    )

    if existing is not None:
        existing.insights = insights
        existing.evidence_overview = overview
        existing.evidence_findings = findings
        existing.scope_text = scope_text
        if image_png is not None:
            existing.image_png = image_png
        return state

    title = str(prompt or "Completed analysis").strip()
    if len(title) > 82:
        title = title[:79].rstrip() + "..."

    step = InvestigationStep(
        title=title,
        prompt=str(prompt or "").strip(),
        investigation_id=investigation_id,
        chart_type=str(current_spec.get("chart_type") or ""),
        special_mode=current_spec.get("special_mode"),
        insights=insights,
        evidence_overview=overview,
        evidence_findings=findings,
        step_type="completed_analysis",
        scope_text=scope_text,
        image_png=image_png,
    )
    state.history.append(step)

    if step.chart_type and step.chart_type not in state.visualizations_seen:
        state.visualizations_seen.append(step.chart_type)
    state.current_question = step.prompt
    return state


def set_current_analysis_translation(
    state: InvestigationState,
    *,
    prompt: str,
    translation: dict[str, str],
) -> InvestigationState:
    """Attach a translation to the most recent matching completed analysis."""
    normalized_prompt = str(prompt or "").strip().lower()
    for step in reversed(state.history):
        if (
            getattr(step, "step_type", "completed_analysis") != "transition"
            and str(getattr(step, "prompt", "")).strip().lower() == normalized_prompt
        ):
            step.common_translation = dict(translation or {})
            break
    return state


def completed_analysis_steps(state: InvestigationState) -> list[InvestigationStep]:
    """Return only analyses that were actually rendered and interpreted."""
    return [
        step for step in state.history
        if getattr(step, "step_type", "completed_analysis") != "transition"
    ]


def add_investigation_step(
    state: InvestigationState,
    suggestion: DiveSuggestion,
    current_spec: dict[str, Any],
    insights: str = "",
    evidence_summary: Any | None = None,
) -> InvestigationState:
    step = InvestigationStep(
        title=suggestion.title,
        prompt=suggestion.prompt,
        investigation_id=suggestion.investigation_id,
        chart_type=str(current_spec.get("chart_type") or ""),
        special_mode=current_spec.get("special_mode"),
        unit_id=suggestion.unit_id,
        insights=insights,
        evidence_overview=(getattr(evidence_summary, "overview", "") if evidence_summary else ""),
        evidence_findings=(
            [finding.observation for finding in getattr(evidence_summary, "findings", [])]
            if evidence_summary else []
        ),
        step_type="transition",
    )
    state.history.append(step)
    if step.chart_type and step.chart_type not in state.visualizations_seen:
        state.visualizations_seen.append(step.chart_type)
    if suggestion.unit_id and suggestion.unit_id not in state.units_examined:
        state.units_examined.append(suggestion.unit_id)
    if suggestion.intent_id and suggestion.intent_id not in state.intents_examined:
        state.intents_examined.append(suggestion.intent_id)
    state.current_question = suggestion.prompt
    return state


def add_coach_investigation_step(
    state: InvestigationState,
    suggestion: Any,
    current_spec: dict[str, Any],
    insights: str = "",
    evidence_summary: Any | None = None,
) -> InvestigationState:
    """Record a Data Analyst Coach continuation for the Phase 4 investigation summary."""
    step = InvestigationStep(
        title=suggestion.title,
        prompt=suggestion.prompt,
        investigation_id=suggestion.suggestion_id,
        chart_type=current_spec.get("chart_type"),
        special_mode=current_spec.get("special_mode"),
        unit_id=None,
        insights=insights,
        evidence_overview=(
            getattr(evidence_summary, "overview", "") if evidence_summary else ""
        ),
        evidence_findings=(
            [finding.observation for finding in getattr(evidence_summary, "findings", [])]
            if evidence_summary else []
        ),
        step_type="transition",
    )
    state.history.append(step)
    return state
