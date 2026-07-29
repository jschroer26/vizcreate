"""Profile-aware unit-of-analysis and decision-basis recognition."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd

from profiles.base_profile import DatasetProfileResult


@dataclass
class TargetUnitResult:
    unit_id: str
    display_name: str
    confidence: float
    source: str
    exact_column: str | None = None

    @property
    def confidence_percent(self) -> int:
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))


@dataclass
class DecisionBasisResult:
    basis_id: str
    display_name: str
    rationale: str
    criterion_column: str | None = None
    caution: str = ""


UNIT_PATTERNS = [
    ("student", "Student", [r"\bwhich students?\b", r"\bwho\b", r"\bstudents?\b"]),
    ("skill", "Skill or Subscore", [r"\bskills?\b", r"\bsubscores?\b", r"\bstandards?\b"]),
    ("grade", "Grade Level", [r"\bwhich grades?\b", r"\bgrade levels?\b", r"\bby grade\b"]),
    ("school", "School", [r"\bwhich schools?\b", r"\bschools?\b"]),
    ("subject", "Subject", [r"\bwhich subjects?\b", r"\bsubjects?\b"]),
    ("subgroup", "Student Subgroup", [r"\bsubgroups?\b", r"\bdemographic groups?\b"]),
    ("survey_item", "Survey Item", [r"\bwhich items?\b", r"\bsurvey items?\b", r"\bquestions?\b"]),
    ("construct", "Survey Construct", [r"\bconstructs?\b", r"\bdomains?\b", r"\bthemes?\b"]),
    ("time_period", "Time Period", [r"\bwhich years?\b", r"\byears?\b", r"\bwindows?\b"]),
]


def _first_role_column(profile: DatasetProfileResult, role_names: list[str]) -> str | None:
    for role_name in role_names:
        value = profile.detected_roles.get(role_name)
        if value:
            return str(value).split(",")[0].strip()
    return None


def detect_target_unit(
    df: pd.DataFrame,
    user_prompt: str,
    profile: DatasetProfileResult,
) -> TargetUnitResult:
    """Identify who or what the user wants evaluated."""
    prompt_lower = user_prompt.lower()

    role_columns = {
        "student": _first_role_column(profile, ["student", "student_id"]),
        "grade": _first_role_column(profile, ["grade"]),
        "school": _first_role_column(profile, ["school"]),
        "subject": _first_role_column(profile, ["subject"]),
        "subgroup": _first_role_column(profile, ["subgroup"]),
        "time_period": _first_role_column(
            profile,
            ["time", "year", "school_year", "window", "date"],
        ),
    }

    for unit_id, display_name, patterns in UNIT_PATTERNS:
        if any(re.search(pattern, prompt_lower) for pattern in patterns):
            return TargetUnitResult(
                unit_id=unit_id,
                display_name=display_name,
                confidence=0.95,
                source="explicitly named in prompt",
                exact_column=role_columns.get(unit_id),
            )

    # Profile defaults are used only when the prompt does not name a unit.
    defaults = {
        "student_assessment": ("student", "Student", ["student", "student_id"]),
        "cbm_progress_monitoring": ("student", "Student", ["student", "student_id"]),
        "likert_survey": ("construct", "Survey Construct", []),
        "wytopp_longitudinal": ("grade_subject", "Grade and Subject", []),
        "wytopp_current_year": ("grade_subject", "Grade and Subject", []),
    }
    if profile.profile_id in defaults:
        unit_id, display_name, role_names = defaults[profile.profile_id]
        return TargetUnitResult(
            unit_id=unit_id,
            display_name=display_name,
            confidence=0.62,
            source="profile default",
            exact_column=_first_role_column(profile, role_names),
        )

    return TargetUnitResult(
        unit_id="category",
        display_name="Available Category",
        confidence=0.40,
        source="generic default",
        exact_column=None,
    )


def detect_decision_basis(
    df: pd.DataFrame,
    user_prompt: str,
    profile: DatasetProfileResult,
    target_unit: TargetUnitResult,
) -> DecisionBasisResult:
    """Identify the evidence standard used to call something a strength or need."""
    prompt_lower = user_prompt.lower()
    columns = [str(column) for column in df.columns]
    lower_columns = {column.lower(): column for column in columns}

    goal_column = next(
        (
            original
            for lower, original in lower_columns.items()
            if any(token in lower for token in ["goal", "aimline", "benchmark", "target"])
        ),
        None,
    )
    criterion_column = next(
        (
            original
            for lower, original in lower_columns.items()
            if any(
                token in lower
                for token in [
                    "proficiency",
                    "performance level",
                    "risk level",
                    "status",
                    "basic and below",
                    "proficient and advanced",
                ]
            )
        ),
        None,
    )
    time_column = next(
        (
            original
            for lower, original in lower_columns.items()
            if any(token in lower for token in ["year", "date", "window", "time", "week"])
        ),
        None,
    )

    if goal_column:
        return DecisionBasisResult(
            basis_id="goal_referenced",
            display_name="Goal-referenced",
            rationale=f"Performance can be compared with the available goal field, {goal_column}.",
            criterion_column=goal_column,
        )

    if criterion_column or profile.profile_id.startswith("wytopp"):
        return DecisionBasisResult(
            basis_id="criterion_referenced",
            display_name="Criterion-referenced",
            rationale=(
                "The dataset contains performance categories or proficiency measures "
                "that provide an external decision criterion."
            ),
            criterion_column=criterion_column,
        )

    if profile.profile_id == "cbm_progress_monitoring" and time_column:
        return DecisionBasisResult(
            basis_id="trend_referenced",
            display_name="Trend-referenced",
            rationale=(
                "Repeated observations support evaluating growth and change, even when "
                "a separate goal field is unavailable."
            ),
            criterion_column=time_column,
        )

    if any(term in prompt_lower for term in ["compared with", "relative", "highest", "lowest"]):
        return DecisionBasisResult(
            basis_id="peer_relative",
            display_name="Peer-relative",
            rationale="Strengths and needs are defined relative to other observations in this dataset.",
            caution=(
                "Relative standing does not show whether an external benchmark has been met."
            ),
        )

    if profile.profile_id in {"student_assessment", "likert_survey"}:
        return DecisionBasisResult(
            basis_id="distribution_relative",
            display_name="Distribution-relative",
            rationale=(
                "The dataset supports identifying comparatively high, low, or unusual "
                "values within the uploaded sample."
            ),
            caution=(
                "No benchmark or decision threshold was detected. Use this as a screening "
                "view, not as a stand-alone intervention or proficiency decision."
            ),
        )

    return DecisionBasisResult(
        basis_id="descriptive_only",
        display_name="Descriptive only",
        rationale=(
            "The available fields support description, but do not clearly establish "
            "whether higher or lower values should be interpreted as better."
        ),
        caution=(
            "Confirm the desired direction and an appropriate criterion before labeling "
            "a value as a strength or need."
        ),
    )


def decision_context_prompt_block(
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
) -> str:
    exact_column = target_unit.exact_column or "No exact unit column detected"
    caution = decision_basis.caution or "None"
    return f"""
DECISION CONTEXT
----------------
Requested unit of analysis: {target_unit.display_name}
Unit confidence: {target_unit.confidence_percent}%
Unit source: {target_unit.source}
Exact unit column: {exact_column}

Decision basis: {decision_basis.display_name}
Rationale: {decision_basis.rationale}
Caution: {caution}

UNIT-PRESERVATION RULES
- The first analysis must preserve an explicitly requested unit.
- "Which students" requires a student-level first output when a student identifier exists.
- "Which grades" requires a grade-level first output.
- "Which schools" requires a school-level first output.
- "Which skills" requires a skill, item, standard, or subscore-level first output.
- Do not replace an explicitly requested student-level analysis with an aggregated grade-level chart.
- When no external criterion exists, use comparative language such as "relatively high"
  and "relatively low"; do not claim proficiency, risk status, or intervention need.
""".strip()
