"""VizCreate Planner 2.0: prompt interpretation, candidate generation, and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any

import pandas as pd

from core.decision_context import DecisionBasisResult, TargetUnitResult
from core.intent_recognition import AnalysisIntentResult
from core.role_coach import AnalystCoachPlan, UserRoleResult
from core.visualization_decision import (
    AnalyticalObjective,
    detect_analytical_objective,
    direct_answer_score,
)
from profiles.base_profile import DatasetProfileResult


SUPPORTED_CHART_TYPES = {
    "bar",
    "stacked_bar",
    "line",
    "heatmap",
    "box",
    "scatter",
}

SUPPORTED_SPECIAL_MODES = {
    None,
    "wytopp_stacked",
    "likert_construct_summary",
    "student_support_map",
    "student_ranked_scores",
    "multi_measure_box",
    "multi_measure_bar",
}


@dataclass
class PlannerCandidate:
    analysis_id: str
    title: str
    decision_value: str
    chart_type: str
    special_mode: str | None
    spec: dict[str, Any]
    llm_score: float
    statistical_suitability: float = 0.0
    direct_answer_score: float = 0.0
    directly_answers_question: bool = True
    direct_answer_reason: str = ""
    deterministic_score: float = 0.0
    total_score: float = 0.0
    strengths: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    validation_notes: list[str] = field(default_factory=list)
    is_valid: bool = True


@dataclass
class PlannerResult:
    educational_decision: str
    interpreted_request: str
    ambiguities: list[str]
    assumptions: list[str]
    candidates: list[PlannerCandidate]
    recommended_analysis_id: str
    recommendation_confidence: float
    confidence_reason: str
    final_spec: dict[str, Any]
    suggested_follow_up_questions: list[str]
    analytical_objective: AnalyticalObjective | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_percent(self) -> int:
        return int(round(max(0.0, min(100.0, self.recommendation_confidence))))


def planner_schema_text() -> str:
    return """
Return exactly one JSON object using this structure:
{
  "prompt_interpretation": {
    "educational_decision": "brief decision label",
    "interpreted_request": "one plain-language sentence",
    "ambiguities": ["only genuine unresolved ambiguities"],
    "assumptions": ["only assumptions necessary to proceed"]
  },
  "candidate_analyses": [
    {
      "analysis_id": "candidate_1",
      "title": "short analysis title",
      "decision_value": "how this helps the user make the stated decision",
      "chart_type": "bar | stacked_bar | line | heatmap | box | scatter",
      "special_mode": "wytopp_stacked | likert_construct_summary | student_support_map | student_ranked_scores | null",
      "x": "exact column name or null",
      "y": "exact column name or null",
      "group": "exact column name or null",
      "row": "exact column name or null",
      "col": "exact column name or null",
      "label": "exact identifier column or null",
      "filters": {},
      "item_columns": [],
      "aggregation": "mean | sum | count | none",
      "sort_x": "none | ascending | descending",
      "facets": null,
      "notes": "brief statistically cautious interpretation",
      "fit_score": 0,
      "strengths": ["specific reason this analysis fits"],
      "limitations": ["specific limitation"]
    }
  ],
  "recommended_analysis_id": "candidate_1",
  "recommendation_confidence": 0,
  "confidence_reason": "brief evidence-based explanation",
  "suggested_follow_up_questions": ["three concise next questions"]
}

PLANNING REQUIREMENTS
- Produce 2 or 3 genuinely different candidate analyses.
- Candidate 1 should usually be the strongest option, but all candidates must be executable.
- Use only exact dataset column names.
- Treat explicit scope language such as only, just, specifically, within, for Grade 3,
  Math, a named subgroup, school, or year as an executable data constraint.
- Put every explicit constraint into filters using exact dataset values.
- Resolve relative time language such as "most recent year", "latest year",
  and "last three years" to the actual chronological values in the dataset.
- Do not leave a requested filter dimension on the x-axis merely so the user can
  manually remove unwanted values.
- Do not invent benchmarks, cut scores, categories, variables, or causal claims.
- Preserve an explicitly requested unit of analysis.
- Distinguish the user's educational decision from the statistical operation.
- Rank candidates by decision usefulness, statistical validity, role fit, unit fit,
  decision-basis fit, interpretability, and data support.
- A visually familiar chart is not automatically the best chart.
- If a table would be more useful but is not supported, choose the closest supported
  visualization and state the limitation.
- recommendation_confidence is 0 to 100.
- Return JSON only.
""".strip()


def build_planner_prompt(
    *,
    schema_text: str,
    user_prompt: str,
    profile_text: str,
    intent_text: str,
    role_text: str,
    decision_context_text: str,
    coach_plan_text: str,
) -> str:
    return f"""
You are the VizCreate LLM Planner.

Your task is not merely to choose a chart. First determine the educational
decision the user is trying to make. Then generate multiple valid analyses,
compare them, and recommend the analysis whose visual evidence best supports
that decision.

The deterministic portions of VizCreate have already inspected the dataset.
Treat their findings as governing context. Do not contradict them without
explicitly naming a genuine ambiguity.

{profile_text}

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

{planner_schema_text()}
""".strip()


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    raw_text = raw_text.strip()
    try:
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Planner response must be a JSON object.")
        return parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object was found in the planner response.")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Planner response must be a JSON object.")
        return parsed


def _candidate_spec(candidate: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "chart_type",
        "special_mode",
        "x",
        "y",
        "group",
        "row",
        "col",
        "label",
        "filters",
        "item_columns",
        "aggregation",
        "sort_x",
        "facets",
        "notes",
    ]
    spec = {field: candidate.get(field) for field in fields}
    spec["filters"] = spec.get("filters") if isinstance(spec.get("filters"), dict) else {}
    spec["item_columns"] = (
        spec.get("item_columns")
        if isinstance(spec.get("item_columns"), list)
        else []
    )
    return spec


def _referenced_columns(spec: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for field in ["x", "y", "group", "row", "col", "label"]:
        value = spec.get(field)
        if isinstance(value, str) and value:
            columns.append(value)
    for value in spec.get("item_columns", []):
        if isinstance(value, str) and value:
            columns.append(value)
    filters = spec.get("filters", {})
    if isinstance(filters, dict):
        columns.extend(str(column) for column in filters)
    return list(dict.fromkeys(columns))


def _numeric_columns(df: pd.DataFrame) -> set[str]:
    return {
        str(column)
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    }


def _validate_candidate(
    df: pd.DataFrame,
    candidate: PlannerCandidate,
) -> None:
    spec = candidate.spec
    chart_type = spec.get("chart_type")
    special_mode = spec.get("special_mode")
    dataset_columns = set(df.columns.astype(str))
    numeric_columns = _numeric_columns(df)

    if chart_type not in SUPPORTED_CHART_TYPES:
        candidate.is_valid = False
        candidate.validation_notes.append(f"Unsupported chart type: {chart_type}")

    if special_mode not in SUPPORTED_SPECIAL_MODES:
        candidate.is_valid = False
        candidate.validation_notes.append(f"Unsupported special mode: {special_mode}")

    missing = [
        column
        for column in _referenced_columns(spec)
        if column not in dataset_columns
    ]
    if missing:
        candidate.is_valid = False
        candidate.validation_notes.append(
            "Missing dataset columns: " + ", ".join(missing)
        )

    if chart_type == "scatter":
        if spec.get("x") not in numeric_columns or spec.get("y") not in numeric_columns:
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Scatterplots require numeric x and y fields."
            )
        if spec.get("x") == spec.get("y"):
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Scatterplot x and y must be different fields."
            )

    if chart_type == "heatmap":
        if not all(spec.get(field) for field in ["row", "col", "y"]):
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Heatmaps require row, col, and y."
            )

    if chart_type == "bar" and special_mode == "multi_measure_bar":
        item_columns = spec.get("item_columns", [])
        if not spec.get("x") or not item_columns:
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Multi-measure bar charts require x and one or more item_columns."
            )
        elif any(column not in numeric_columns for column in item_columns):
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Every multi-measure bar-chart item must be numeric."
            )

    if chart_type == "box" and special_mode == "multi_measure_box":
        item_columns = spec.get("item_columns", [])
        if not spec.get("x") or not item_columns:
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Multi-measure box plots require x and one or more item_columns."
            )
        elif any(column not in numeric_columns for column in item_columns):
            candidate.is_valid = False
            candidate.validation_notes.append(
                "Every multi-measure box-plot item must be numeric."
            )

    if chart_type in {"bar", "line", "box"} and special_mode not in {
        "likert_construct_summary",
        "multi_measure_box",
        "multi_measure_bar",
    }:
        if spec.get("y") not in numeric_columns:
            candidate.is_valid = False
            candidate.validation_notes.append(
                f"{chart_type} requires a numeric y field."
            )


def _deterministic_fit_score(
    candidate: PlannerCandidate,
    *,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
) -> float:
    """Score decision fit without trying to replace the LLM's semantic judgment."""
    spec = candidate.spec
    chart_type = spec.get("chart_type")
    special_mode = spec.get("special_mode")
    score = 0.0

    if candidate.is_valid:
        score += 25.0
    else:
        return -100.0

    exact_unit_column = target_unit.exact_column
    referenced = set(_referenced_columns(spec))

    if exact_unit_column and exact_unit_column in referenced:
        score += 18.0
        candidate.validation_notes.append("Preserves the requested unit column.")
    elif target_unit.source == "explicitly named in prompt":
        score -= 22.0
        candidate.validation_notes.append("Does not directly preserve the requested unit.")

    if intent.intent_id == "strengths_needs":
        if target_unit.unit_id == "student":
            if special_mode == "student_support_map":
                score += 28.0
            elif special_mode == "student_ranked_scores":
                score += 23.0
            elif chart_type == "box":
                score -= 25.0
                candidate.validation_notes.append(
                    "A box plot summarizes groups but does not identify students clearly."
                )
        elif chart_type in {"bar", "heatmap", "line"}:
            score += 10.0

    if intent.intent_id == "relationship":
        score += 25.0 if chart_type == "scatter" else -18.0
    elif intent.intent_id == "trend":
        score += 22.0 if chart_type == "line" else -8.0
    elif intent.intent_id == "distribution":
        score += 18.0 if chart_type == "box" else 0.0
    elif intent.intent_id == "composition":
        score += 20.0 if chart_type == "stacked_bar" else -5.0
    elif intent.intent_id == "ranking":
        score += 18.0 if chart_type == "bar" else 0.0

    if decision_basis.basis_id in {"distribution_relative", "peer_relative"}:
        notes = str(spec.get("notes", "")).lower()
        if any(term in notes for term in ["relative", "compar", "screening"]):
            score += 7.0

    if profile.profile_id.startswith("wytopp") and special_mode == "wytopp_stacked":
        score += 14.0
    if profile.profile_id == "likert_survey" and special_mode == "likert_construct_summary":
        score += 14.0

    if role.role_id == "board" and target_unit.unit_id == "student":
        score -= 12.0
        candidate.validation_notes.append(
            "Student-level detail may be too operational for a governance perspective."
        )
    if role.role_id == "teacher" and target_unit.unit_id == "student":
        if special_mode in {"student_support_map", "student_ranked_scores"}:
            score += 10.0

    return score


def parse_and_rank_planner_response(
    raw: dict[str, Any],
    *,
    df: pd.DataFrame,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
    user_prompt: str = "",
) -> PlannerResult:
    interpretation = raw.get("prompt_interpretation", {})
    if not isinstance(interpretation, dict):
        interpretation = {}

    raw_candidates = raw.get("candidate_analyses", [])
    if not isinstance(raw_candidates, list):
        raw_candidates = []

    objective_text = user_prompt or str(interpretation.get("interpreted_request") or "")
    analytical_objective = detect_analytical_objective(objective_text, intent.intent_id)

    candidates: list[PlannerCandidate] = []
    for index, raw_candidate in enumerate(raw_candidates[:3], start=1):
        if not isinstance(raw_candidate, dict):
            continue
        candidate = PlannerCandidate(
            analysis_id=str(raw_candidate.get("analysis_id") or f"candidate_{index}"),
            title=str(raw_candidate.get("title") or f"Candidate analysis {index}"),
            decision_value=str(raw_candidate.get("decision_value") or ""),
            chart_type=str(raw_candidate.get("chart_type") or ""),
            special_mode=raw_candidate.get("special_mode"),
            spec=_candidate_spec(raw_candidate),
            llm_score=float(raw_candidate.get("fit_score") or 0.0),
            strengths=[
                str(value)
                for value in raw_candidate.get("strengths", [])
                if value is not None
            ],
            limitations=[
                str(value)
                for value in raw_candidate.get("limitations", [])
                if value is not None
            ],
        )
        _validate_candidate(df, candidate)
        candidate.deterministic_score = _deterministic_fit_score(
            candidate,
            profile=profile,
            intent=intent,
            role=role,
            target_unit=target_unit,
            decision_basis=decision_basis,
        )
        candidate.statistical_suitability = max(0.0, min(100.0, candidate.llm_score))
        (
            candidate.direct_answer_score,
            candidate.directly_answers_question,
            candidate.direct_answer_reason,
        ) = direct_answer_score(
            analytical_objective,
            candidate.chart_type,
            candidate.spec,
        )
        if not candidate.directly_answers_question:
            candidate.validation_notes.append(candidate.direct_answer_reason)
        support_score = max(0.0, min(100.0, 50.0 + candidate.deterministic_score))
        candidate.total_score = (
            0.60 * candidate.direct_answer_score
            + 0.25 * candidate.statistical_suitability
            + 0.15 * support_score
        )
        candidates.append(candidate)

    valid_candidates = [candidate for candidate in candidates if candidate.is_valid]
    if not valid_candidates:
        raise ValueError("The planner did not return any executable candidates.")

    directly_answering = [
        candidate for candidate in valid_candidates
        if candidate.directly_answers_question
    ]
    ranking_pool = directly_answering or valid_candidates
    ranking_pool.sort(key=lambda item: item.total_score, reverse=True)
    winner = ranking_pool[0]

    llm_confidence = float(raw.get("recommendation_confidence") or 0.0)
    gap = (
        winner.total_score - valid_candidates[1].total_score
        if len(valid_candidates) > 1
        else 20.0
    )
    confidence = max(35.0, min(99.0, 0.65 * llm_confidence + 0.35 * (60.0 + gap)))

    return PlannerResult(
        educational_decision=str(
            interpretation.get("educational_decision") or "Educational data exploration"
        ),
        interpreted_request=str(
            interpretation.get("interpreted_request") or ""
        ),
        ambiguities=[
            str(value)
            for value in interpretation.get("ambiguities", [])
            if value is not None
        ],
        assumptions=[
            str(value)
            for value in interpretation.get("assumptions", [])
            if value is not None
        ],
        candidates=candidates,
        recommended_analysis_id=winner.analysis_id,
        recommendation_confidence=confidence,
        confidence_reason=str(
            raw.get("confidence_reason")
            or "The selected analysis had the strongest combined semantic and deterministic fit."
        ),
        final_spec=dict(winner.spec),
        analytical_objective=analytical_objective,
        suggested_follow_up_questions=[
            str(value)
            for value in raw.get("suggested_follow_up_questions", [])
            if value is not None
        ][:3],
        raw_response=raw,
    )


def fallback_planner_result(
    *,
    spec: dict[str, Any],
    coach_plan: AnalystCoachPlan,
    intent: AnalysisIntentResult,
    target_unit: TargetUnitResult,
    decision_basis: DecisionBasisResult,
    error_message: str,
) -> PlannerResult:
    candidate = PlannerCandidate(
        analysis_id="deterministic_fallback",
        title=coach_plan.alternatives[0].title if coach_plan.alternatives else "Recommended analysis",
        decision_value=coach_plan.recommendation,
        chart_type=str(spec.get("chart_type") or ""),
        special_mode=spec.get("special_mode"),
        spec=dict(spec),
        llm_score=0.0,
        statistical_suitability=70.0,
        direct_answer_score=85.0,
        directly_answers_question=True,
        direct_answer_reason="Deterministic fallback selected a supported chart for the detected intent.",
        deterministic_score=80.0,
        total_score=80.0,
        strengths=["Uses VizCreate's deterministic profile, intent, unit, and decision rules."],
        limitations=["The richer LLM planning step was unavailable."],
        validation_notes=[error_message],
        is_valid=True,
    )
    return PlannerResult(
        educational_decision=intent.display_name,
        interpreted_request=coach_plan.recommendation,
        ambiguities=[],
        assumptions=[],
        candidates=[candidate],
        recommended_analysis_id=candidate.analysis_id,
        recommendation_confidence=70.0,
        confidence_reason="VizCreate used its deterministic analyst-coach fallback.",
        final_spec=dict(spec),
        suggested_follow_up_questions=coach_plan.next_questions[:3],
        analytical_objective=detect_analytical_objective(
            coach_plan.recommendation,
            intent.intent_id,
        ),
        raw_response={},
    )
