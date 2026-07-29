"""Deterministic statistical-intent recognition for VizCreate."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd

from profiles.base_profile import DatasetProfileResult


@dataclass
class AnalysisIntentResult:
    """Detected purpose of the user's analytical question."""

    intent_id: str
    display_name: str
    confidence: float
    description: str
    exact_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    preferred_charts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @property
    def confidence_percent(self) -> int:
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))


INTENT_PATTERNS = {
    "strengths_needs": [
        r"\bdoing great\b",
        r"\bdoing well\b",
        r"\bneed(?:s|ing)? (?:more )?help\b",
        r"\bneed(?:s|ing)? support\b",
        r"\bstrengths?\b",
        r"\bweakness(?:es)?\b",
        r"\bareas? of concern\b",
        r"\bpriority areas?\b",
        r"\bstruggl",
        r"\bintervention\b",
        r"\benrichment\b",
        r"\bat risk\b",
        r"\bwhere should .* focus\b",
    ],
    "relationship": [
        r"\brelated\b",
        r"\brelationship\b",
        r"\bcorrelat",
        r"\bassociation\b",
        r"\bpredict",
        r"\bconnected\b",
        r"\bmove together\b",
    ],
    "trend": [
        r"\bover time\b",
        r"\btrend",
        r"\bchange over\b",
        r"\bgrowth\b",
        r"\bdecline\b",
        r"\bprogress\b",
        r"\bfrom .* to\b",
    ],
    "distribution": [
        r"\bdistribution\b",
        r"\bspread\b",
        r"\bvariability\b",
        r"\bvariation\b",
        r"\boutlier",
        r"\bmedian\b",
        r"\brange\b",
    ],
    "ranking": [
        r"\bhighest\b",
        r"\blowest\b",
        r"\bstrongest\b",
        r"\bweakest\b",
        r"\bbest\b",
        r"\bworst\b",
        r"\brank",
        r"\bmost\b",
        r"\bleast\b",
    ],
    "composition": [
        r"\bproportion\b",
        r"\bpercentage breakdown\b",
        r"\bparts? of\b",
        r"\bcomposition\b",
        r"\bshare\b",
        r"\bstack",
    ],
    "comparison": [
        r"\bcompare\b",
        r"\bdifference\b",
        r"\bdiffer\b",
        r"\bversus\b",
        r"\bvs\.?\b",
        r"\bacross\b",
        r"\bby grade\b",
        r"\bby group\b",
    ],
    "status": [
        r"\boverview\b",
        r"\blook at first\b",
        r"\bwhat should .* look",
        r"\bcurrent status\b",
        r"\bwhere should .* start\b",
    ],
}


def _columns_named_in_prompt(
    df: pd.DataFrame,
    user_prompt: str,
) -> list[str]:
    """Return exact dataframe columns explicitly named in the prompt."""
    prompt_lower = user_prompt.lower()
    matches = [
        str(column)
        for column in df.columns
        if str(column).lower() in prompt_lower
    ]
    return sorted(matches, key=len, reverse=True)


def _role_columns_from_profile(
    df: pd.DataFrame,
    profile: DatasetProfileResult,
) -> list[str]:
    """Extract exact columns embedded in detected role metadata."""
    found: list[str] = []
    for value in profile.detected_roles.values():
        for part in str(value).split(","):
            candidate = part.strip()
            if candidate in df.columns and candidate not in found:
                found.append(candidate)
    return found


def detect_analysis_intent(
    df: pd.DataFrame,
    user_prompt: str,
    profile: DatasetProfileResult,
) -> AnalysisIntentResult:
    """Classify the user's question independently of dataset recognition."""
    prompt_lower = user_prompt.lower().strip()
    exact_columns = _columns_named_in_prompt(df, user_prompt)

    numeric_columns = [
        column
        for column in exact_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]
    categorical_columns = [
        column
        for column in exact_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    scores = {intent: 0.0 for intent in INTENT_PATTERNS}
    evidence_map: dict[str, list[str]] = {
        intent: [] for intent in INTENT_PATTERNS
    }

    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, prompt_lower):
                scores[intent] += 0.25
                evidence_map[intent].append(
                    f'Prompt matched intent phrase: "{pattern}".'
                )

    # Strong decision-language evidence for strengths and needs.
    strength_language = any(
        re.search(pattern, prompt_lower)
        for pattern in INTENT_PATTERNS["strengths_needs"]
    )
    explicit_unit_language = any(
        phrase in prompt_lower
        for phrase in [
            "which student",
            "which grade",
            "which school",
            "which subject",
            "which skill",
            "which item",
            "which construct",
            "who is",
            "who needs",
        ]
    )
    if strength_language:
        scores["strengths_needs"] += 0.55
        evidence_map["strengths_needs"].append(
            "The prompt asks to identify relative strengths, needs, support, or enrichment."
        )
    if strength_language and explicit_unit_language:
        scores["strengths_needs"] = max(scores["strengths_needs"], 0.95)
        evidence_map["strengths_needs"].append(
            "The prompt also names or implies a target unit such as student, grade, or skill."
        )

    # Strong structural evidence for relationships.
    if len(numeric_columns) >= 2:
        scores["relationship"] += 0.45
        evidence_map["relationship"].append(
            "At least two exact numeric columns were named."
        )

    # Profile-aware modifiers.
    if profile.profile_id == "likert_survey":
        if any(word in prompt_lower for word in ["construct", "item", "ratings"]):
            scores["ranking"] += 0.20
            evidence_map["ranking"].append(
                "Survey construct language supports a ranking or comparison."
            )

    if profile.profile_id == "cbm_progress_monitoring":
        if any(word in prompt_lower for word in ["progress", "goal", "growth"]):
            scores["trend"] += 0.25
            evidence_map["trend"].append(
                "CBM language supports repeated-measure trend analysis."
            )

    if profile.profile_id.startswith("wytopp"):
        if "basic" in prompt_lower and "proficient" in prompt_lower:
            scores["composition"] += 0.30
            evidence_map["composition"].append(
                "Complementary proficiency categories support composition."
            )

    # Exact numeric relationship takes precedence over broad comparison wording.
    if len(numeric_columns) >= 2 and any(
        phrase in prompt_lower
        for phrase in [
            "related",
            "relationship",
            "correlation",
            "associated",
            "predict",
        ]
    ):
        scores["relationship"] = max(scores["relationship"], 0.95)

    best_intent = max(scores, key=scores.get)
    best_score = scores[best_intent]

    if best_score <= 0:
        best_intent = "status"
        best_score = 0.30
        evidence_map["status"].append(
            "No specific statistical intent phrase was detected."
        )

    definitions = {
        "strengths_needs": (
            "Strengths and Needs Identification",
            "Identify comparatively strong, weak, concerning, or promising units using an appropriate decision basis.",
            ["scatter", "bar", "heatmap"],
        ),
        "relationship": (
            "Relationship",
            "Examine whether two numeric variables vary together.",
            ["scatter"],
        ),
        "trend": (
            "Trend or Change",
            "Examine ordered change across dates, years, or screening windows.",
            ["line"],
        ),
        "distribution": (
            "Distribution",
            "Examine spread, center, variation, or outliers.",
            ["box"],
        ),
        "ranking": (
            "Ranking",
            "Order categories or constructs from highest to lowest.",
            ["bar"],
        ),
        "composition": (
            "Composition",
            "Show how categories contribute to a whole.",
            ["stacked_bar"],
        ),
        "comparison": (
            "Group Comparison",
            "Compare values across categories or groups.",
            ["bar", "box"],
        ),
        "status": (
            "Overview or Status",
            "Provide a defensible first view of the available data.",
            ["bar", "heatmap", "line"],
        ),
    }

    display_name, description, preferred = definitions[best_intent]

    return AnalysisIntentResult(
        intent_id=best_intent,
        display_name=display_name,
        confidence=min(1.0, best_score),
        description=description,
        exact_columns=exact_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preferred_charts=preferred,
        evidence=evidence_map[best_intent],
    )


def intent_prompt_block(intent: AnalysisIntentResult) -> str:
    """Create explicit intent guidance for the LLM planner."""
    exact_text = ", ".join(intent.exact_columns) or "None explicitly named"
    numeric_text = ", ".join(intent.numeric_columns) or "None explicitly named"
    preferred_text = ", ".join(intent.preferred_charts)

    special_rules = ""
    if intent.intent_id == "strengths_needs":
        special_rules = """
STRENGTHS AND NEEDS EXECUTION RULES
- Preserve the explicitly requested unit in the first analysis.
- When the user asks "which students" and a student identifier exists, use row-level data.
- Do not begin with a grade-level box plot for a student-identification request.
- When two meaningful student-level score columns exist, prefer a student support scatter map.
- When only one student-level score exists, prefer a student-level ranked bar chart.
- Without benchmarks or cut scores, describe results as relatively high or low within
  the uploaded dataset rather than proficient, at risk, or needing intervention.
"""
    elif intent.intent_id == "relationship":
        special_rules = """
RELATIONSHIP EXECUTION RULES
- When two exact numeric variables are named, use chart_type "scatter".
- Set x and y to the two exact numeric column names.
- Use aggregation "none".
- Do not use a heatmap for a row-level relationship between two continuous scores.
- A heatmap is for two categorical dimensions summarized by a third numeric measure.
"""
    elif intent.intent_id == "trend":
        special_rules = """
TREND EXECUTION RULES
- Use an exact ordered time or screening-window column as x.
- Use a numeric measure as y.
- Use line charts for repeated or longitudinal observations.
"""
    elif intent.intent_id == "distribution":
        special_rules = """
DISTRIBUTION EXECUTION RULES
- Use a box plot when comparing observation-level numeric distributions across groups.
- Do not use a box plot for aggregated percentage summaries.
"""
    elif intent.intent_id == "ranking":
        special_rules = """
RANKING EXECUTION RULES
- Prefer a sorted bar chart or an approved special summary mode.
- Use descending order for questions asking highest, strongest, or most.
"""
    elif intent.intent_id == "composition":
        special_rules = """
COMPOSITION EXECUTION RULES
- Prefer stacked bars only when categories represent meaningful parts of a whole.
"""

    return f"""
STATISTICAL INTENT
------------------
Intent: {intent.display_name}
Confidence: {intent.confidence_percent}%
Purpose: {intent.description}
Exact columns named: {exact_text}
Exact numeric columns named: {numeric_text}
Preferred chart families: {preferred_text}
{special_rules}
""".strip()



def _role_columns(profile, role_name: str) -> list[str]:
    value = profile.detected_roles.get(role_name, "") if profile else ""
    return [
        part.strip()
        for part in str(value).split(",")
        if part.strip()
    ]


def repair_spec_for_intent(
    df: pd.DataFrame,
    spec: dict,
    intent: AnalysisIntentResult,
    profile=None,
    target_unit=None,
    decision_basis=None,
    analytical_objective=None,
) -> dict:
    """Repair a returned chart spec using deterministic statistical intent."""
    repaired = dict(spec or {})
    repaired.setdefault("filters", {})

    # Phase 2G governs chart-family selection when the question explicitly
    # identifies a statistic or analytical objective. Older intent-repair
    # rules may refine fields, but must not replace the winning chart family.
    objective_id = getattr(analytical_objective, "objective_id", None)
    objective_statistic = getattr(analytical_objective, "statistic", None)
    if objective_statistic == "median" or objective_id in {"rank_median", "compare_median"}:
        if repaired.get("chart_type") == "box":
            repaired["aggregation"] = "none"
            repaired["special_mode"] = None
            return repaired
        if repaired.get("chart_type") == "bar":
            repaired["aggregation"] = "median"
            repaired["special_mode"] = None
            return repaired

    if intent.intent_id == "strengths_needs":
        unit_id = getattr(target_unit, "unit_id", None)
        student_columns = _role_columns(profile, "student") + _role_columns(profile, "student_id")
        grade_columns = _role_columns(profile, "grade")
        measure_columns = [
            column
            for column in _role_columns(profile, "assessment_measures")
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column])
        ]

        # A student-identification question must remain at the student level.
        if unit_id == "student" and student_columns:
            student_col = student_columns[0]
            if len(measure_columns) >= 2:
                x_col, y_col = measure_columns[:2]
                repaired.update(
                    {
                        "chart_type": "scatter",
                        "special_mode": "student_support_map",
                        "x": x_col,
                        "y": y_col,
                        "group": grade_columns[0] if grade_columns else None,
                        "row": None,
                        "col": None,
                        "label": student_col,
                        "aggregation": "none",
                        "sort_x": "none",
                        "facets": None,
                    }
                )
                repaired["notes"] = (
                    f"Student-level support map using {x_col} and {y_col}. "
                    "Each point represents one student. Interpret higher and lower "
                    "positions comparatively unless benchmarks are supplied."
                )
            elif len(measure_columns) == 1:
                score_col = measure_columns[0]
                repaired.update(
                    {
                        "chart_type": "bar",
                        "special_mode": "student_ranked_scores",
                        "x": student_col,
                        "y": score_col,
                        "group": None,
                        "row": None,
                        "col": None,
                        "aggregation": "none",
                        "sort_x": "descending",
                        "facets": None,
                    }
                )
                repaired["notes"] = (
                    f"Student-level ranking of {score_col}. Use this as a comparative "
                    "screening view unless an external benchmark is supplied."
                )
            return repaired

        # Non-student strengths/needs requests should preserve the named unit when possible.
        unit_column = getattr(target_unit, "exact_column", None)
        numeric_columns = [
            column for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        if unit_column in df.columns and numeric_columns:
            current_y = repaired.get("y")
            y_col = current_y if current_y in numeric_columns else numeric_columns[0]
            repaired.update(
                {
                    "chart_type": "bar",
                    "special_mode": None,
                    "x": unit_column,
                    "y": y_col,
                    "aggregation": "mean",
                    "sort_x": "descending",
                    "row": None,
                    "col": None,
                }
            )

    if intent.intent_id == "relationship" and len(intent.numeric_columns) >= 2:
        x_col, y_col = intent.numeric_columns[:2]
        repaired.update(
            {
                "chart_type": "scatter",
                "special_mode": None,
                "x": x_col,
                "y": y_col,
                "group": None,
                "row": None,
                "col": None,
                "aggregation": "none",
                "sort_x": "none",
                "facets": None,
            }
        )

        old_notes = str(repaired.get("notes", "")).strip()
        relationship_note = (
            f"Scatterplot of {y_col} against {x_col} to examine their row-level relationship."
        )
        repaired["notes"] = (
            f"{old_notes} {relationship_note}".strip()
            if old_notes
            else relationship_note
        )

    return repaired
