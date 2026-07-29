"""Role-aware analysis planning for the VizCreate Data Analyst Coach."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd

from core.intent_recognition import AnalysisIntentResult
from core.decision_context import DecisionBasisResult, TargetUnitResult
from profiles.base_profile import DatasetProfileResult


ROLE_OPTIONS = [
    "Auto-detect from prompt",
    "Classroom Teacher",
    "Instructional Coach",
    "School Leader / Principal",
    "District Leader / Superintendent",
    "Assessment / Data Coordinator",
    "School Board / Governance",
    "Researcher / Faculty",
]


@dataclass
class UserRoleResult:
    role_id: str
    display_name: str
    confidence: float
    source: str
    perspective: str

    @property
    def confidence_percent(self) -> int:
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))


@dataclass
class AnalysisPlanItem:
    title: str
    rationale: str
    priority: int
    chart_family: str


@dataclass
class AnalystCoachPlan:
    role: UserRoleResult
    headline: str
    recommendation: str
    why_this_first: str
    target_unit: TargetUnitResult | None = None
    decision_basis: DecisionBasisResult | None = None
    can_answer: list[str] = field(default_factory=list)
    cannot_answer: list[str] = field(default_factory=list)
    next_questions: list[str] = field(default_factory=list)
    alternatives: list[AnalysisPlanItem] = field(default_factory=list)
    caution: str = ""


ROLE_DEFINITIONS = {
    "teacher": {
        "display_name": "Classroom Teacher",
        "perspective": (
            "Focus on students, instructional response, near-term support, and "
            "classroom-level patterns."
        ),
        "patterns": [
            r"\bteacher\b",
            r"\bmy class\b",
            r"\bmy students\b",
            r"\bclassroom\b",
        ],
    },
    "coach": {
        "display_name": "Instructional Coach",
        "perspective": (
            "Focus on instructional patterns, teacher support, grade-level teams, "
            "and questions that can guide collaborative inquiry."
        ),
        "patterns": [
            r"\binstructional coach\b",
            r"\bcoach\b",
            r"\bplc\b",
            r"\bprofessional learning\b",
        ],
    },
    "principal": {
        "display_name": "School Leader / Principal",
        "perspective": (
            "Focus on schoolwide priorities, grade-level variation, resource "
            "allocation, and areas requiring instructional follow-up."
        ),
        "patterns": [
            r"\bprincipal\b",
            r"\bschool leader\b",
            r"\bhead of school\b",
            r"\bbuilding leader\b",
        ],
    },
    "district": {
        "display_name": "District Leader / Superintendent",
        "perspective": (
            "Focus on systemwide trends, cross-school consistency, strategic "
            "priorities, and allocation of district support."
        ),
        "patterns": [
            r"\bsuperintendent\b",
            r"\bdistrict leader\b",
            r"\bdistrict administrator\b",
            r"\bcentral office\b",
        ],
    },
    "assessment": {
        "display_name": "Assessment / Data Coordinator",
        "perspective": (
            "Focus on measurement quality, comparability, subgroup definitions, "
            "sample size, and defensible statistical interpretation."
        ),
        "patterns": [
            r"\bassessment coordinator\b",
            r"\bdata coordinator\b",
            r"\bdata analyst\b",
            r"\bassessment director\b",
        ],
    },
    "board": {
        "display_name": "School Board / Governance",
        "perspective": (
            "Focus on broad outcomes, strategic monitoring, accountability, and "
            "questions appropriate for governance rather than daily management."
        ),
        "patterns": [
            r"\bschool board\b",
            r"\bboard member\b",
            r"\btrustee\b",
            r"\bgovernance\b",
        ],
    },
    "researcher": {
        "display_name": "Researcher / Faculty",
        "perspective": (
            "Focus on construct validity, assumptions, uncertainty, effect sizes, "
            "and analyses that support transparent inference."
        ),
        "patterns": [
            r"\bresearcher\b",
            r"\bfaculty\b",
            r"\bprofessor\b",
            r"\bresearch question\b",
        ],
    },
}


SELECTOR_TO_ROLE = {
    "Classroom Teacher": "teacher",
    "Instructional Coach": "coach",
    "School Leader / Principal": "principal",
    "District Leader / Superintendent": "district",
    "Assessment / Data Coordinator": "assessment",
    "School Board / Governance": "board",
    "Researcher / Faculty": "researcher",
}


def resolve_user_role(
    user_prompt: str,
    selected_role: str = "Auto-detect from prompt",
) -> UserRoleResult:
    """Resolve an explicit role selection or infer a role from prompt language."""
    if selected_role in SELECTOR_TO_ROLE:
        role_id = SELECTOR_TO_ROLE[selected_role]
        definition = ROLE_DEFINITIONS[role_id]
        return UserRoleResult(
            role_id=role_id,
            display_name=definition["display_name"],
            confidence=1.0,
            source="selected by user",
            perspective=definition["perspective"],
        )

    prompt_lower = user_prompt.lower()
    for role_id, definition in ROLE_DEFINITIONS.items():
        if any(re.search(pattern, prompt_lower) for pattern in definition["patterns"]):
            return UserRoleResult(
                role_id=role_id,
                display_name=definition["display_name"],
                confidence=0.90,
                source="inferred from prompt",
                perspective=definition["perspective"],
            )

    return UserRoleResult(
        role_id="educator",
        display_name="Educator / General User",
        confidence=0.45,
        source="general default",
        perspective=(
            "Focus on a clear first analysis, accessible interpretation, and "
            "statistically responsible next questions."
        ),
    )


def role_prompt_block(role: UserRoleResult) -> str:
    return f"""
USER ROLE AND DECISION PERSPECTIVE
----------------------------------
Role: {role.display_name}
Confidence: {role.confidence_percent}%
Source: {role.source}
Perspective: {role.perspective}

ROLE-AWARE RULES
- Adapt the framing and recommendation to this decision-making role.
- Do not change the statistical meaning of the data to suit the role.
- Teachers need student- and instruction-focused framing.
- school leaders need schoolwide and grade-level priorities.
- district leaders need systemwide patterns and strategic comparisons.
- governance users need broad monitoring, not student-level management.
- assessment and research users need stronger emphasis on assumptions,
  measurement quality, uncertainty, and limitations.
""".strip()


def _base_plan_items(
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
) -> list[AnalysisPlanItem]:
    """Generate reasonable candidate analyses from profile and intent."""
    items: list[AnalysisPlanItem] = []

    if intent.intent_id == "strengths_needs":
        if profile.profile_id in {"student_assessment", "cbm_progress_monitoring"}:
            items.extend([
                AnalysisPlanItem(
                    "Identify student-level strengths and possible support needs",
                    "Preserve individual students in the first view rather than aggregating immediately by grade.",
                    5,
                    "student support map or ranked student chart",
                ),
                AnalysisPlanItem(
                    "Examine skill or measure patterns",
                    "Determine whether students have consistent performance or uneven profiles across measures.",
                    4,
                    "heatmap or paired comparison",
                ),
                AnalysisPlanItem(
                    "Compare patterns across instructional groups",
                    "After identifying individual patterns, examine whether grade or group context helps explain them.",
                    3,
                    "group comparison",
                ),
            ])
        elif profile.profile_id == "likert_survey":
            items.extend([
                AnalysisPlanItem(
                    "Rank survey constructs or items",
                    "Identify comparatively favorable and concerning areas while preserving response distributions.",
                    5,
                    "ranked bar or Likert summary",
                ),
                AnalysisPlanItem(
                    "Check disagreement and polarization",
                    "Averages alone may conceal divided views or many neutral responses.",
                    4,
                    "divergent Likert",
                ),
                AnalysisPlanItem(
                    "Compare respondent groups",
                    "Determine whether perceived strengths and needs differ across roles or groups.",
                    3,
                    "grouped comparison",
                ),
            ])
        elif profile.profile_id.startswith("wytopp"):
            items.extend([
                AnalysisPlanItem(
                    "Identify priority grades and subjects",
                    "Use proficiency and trend evidence to locate comparatively strong and concerning program areas.",
                    5,
                    "ranked bar or heatmap",
                ),
                AnalysisPlanItem(
                    "Check persistence over time",
                    "A one-year low point differs from a sustained multi-year pattern.",
                    4,
                    "line",
                ),
                AnalysisPlanItem(
                    "Review tested counts and subgroup context",
                    "Small or changing samples can alter apparent strengths and needs.",
                    3,
                    "data check",
                ),
            ])
        else:
            items.extend([
                AnalysisPlanItem(
                    "Identify comparatively high and low units",
                    "Begin with the unit named by the user and use cautious descriptive language.",
                    5,
                    "ranked comparison",
                ),
                AnalysisPlanItem(
                    "Verify whether higher or lower is desirable",
                    "Generic datasets may not reveal the intended direction of a measure.",
                    4,
                    "data check",
                ),
                AnalysisPlanItem(
                    "Inspect variation and outliers",
                    "Averages alone may conceal important patterns.",
                    3,
                    "distribution",
                ),
            ])
    elif intent.intent_id == "relationship":
        items.extend([
            AnalysisPlanItem(
                "Examine the score relationship",
                "Use paired observations to determine whether the two measures vary together.",
                5,
                "scatter",
            ),
            AnalysisPlanItem(
                "Check the distributions",
                "A relationship can be distorted by outliers, restricted ranges, or unusual score distributions.",
                4,
                "box",
            ),
            AnalysisPlanItem(
                "Compare the relationship by group",
                "Grade or subgroup patterns may differ from the overall relationship.",
                3,
                "scatter",
            ),
        ])
    elif intent.intent_id == "trend":
        items.extend([
            AnalysisPlanItem(
                "Establish the overall trend",
                "Begin with ordered change over time before drilling into individual groups.",
                5,
                "line",
            ),
            AnalysisPlanItem(
                "Identify groups driving the change",
                "Compare grades, schools, subjects, or students after establishing the overall direction.",
                4,
                "line",
            ),
            AnalysisPlanItem(
                "Check for inconsistent time coverage",
                "Apparent changes can reflect missing years or changing samples.",
                3,
                "data check",
            ),
        ])
    elif intent.intent_id == "ranking":
        items.extend([
            AnalysisPlanItem(
                "Rank the available measures",
                "A sorted view directly answers which category or construct is strongest or weakest.",
                5,
                "bar",
            ),
            AnalysisPlanItem(
                "Inspect response or score distributions",
                "Means alone can conceal polarization or uneven response patterns.",
                4,
                "distribution",
            ),
            AnalysisPlanItem(
                "Compare rankings across groups",
                "A high overall rating may not be consistent across roles or subgroups.",
                3,
                "grouped bar",
            ),
        ])
    elif intent.intent_id == "distribution":
        items.extend([
            AnalysisPlanItem(
                "Examine center and spread",
                "Start with the full distribution rather than only an average.",
                5,
                "box",
            ),
            AnalysisPlanItem(
                "Locate possible outliers",
                "Extreme observations may be substantively important or may indicate data-quality issues.",
                4,
                "scatter or table",
            ),
            AnalysisPlanItem(
                "Compare distributions across groups",
                "Group differences may involve spread or overlap rather than only different means.",
                3,
                "box",
            ),
        ])
    elif intent.intent_id == "composition":
        items.extend([
            AnalysisPlanItem(
                "Show the overall composition",
                "A stacked view makes the parts of the whole visible.",
                5,
                "stacked bar",
            ),
            AnalysisPlanItem(
                "Compare composition across groups",
                "Differences in category shares may reveal where attention is needed.",
                4,
                "stacked bar",
            ),
            AnalysisPlanItem(
                "Verify that categories form a meaningful whole",
                "Stacking is appropriate only when the displayed categories are conceptually complementary.",
                3,
                "data check",
            ),
        ])
    else:
        items.extend([
            AnalysisPlanItem(
                "Establish the overall pattern",
                "Begin with a clear high-level view before selecting a narrower question.",
                5,
                "recommended profile chart",
            ),
            AnalysisPlanItem(
                "Identify the greatest area of concern",
                "Locate the lowest outcome, largest decline, or greatest inconsistency.",
                4,
                "ranked comparison",
            ),
            AnalysisPlanItem(
                "Check variation across meaningful groups",
                "Overall averages can conceal important grade, school, or subgroup differences.",
                3,
                "group comparison",
            ),
        ])

    return items


def build_analyst_coach_plan(
    df: pd.DataFrame,
    profile: DatasetProfileResult,
    intent: AnalysisIntentResult,
    role: UserRoleResult,
    target_unit: TargetUnitResult | None = None,
    decision_basis: DecisionBasisResult | None = None,
) -> AnalystCoachPlan:
    """Create a role-aware, educationally framed first analysis plan."""
    alternatives = _base_plan_items(profile, intent)
    primary = alternatives[0]

    role_adjustments = {
        "teacher": (
            "Begin with a view that can identify students or instructional patterns "
            "that may warrant classroom follow-up."
        ),
        "coach": (
            "Begin with a pattern suitable for collaborative inquiry, then compare "
            "grade levels or instructional teams."
        ),
        "principal": (
            "Begin with the schoolwide pattern, then identify which grades or groups "
            "contribute most to it."
        ),
        "district": (
            "Begin with the systemwide pattern, then compare schools, grades, or "
            "subjects to locate variation."
        ),
        "assessment": (
            "Begin with the requested analysis, but verify sample size, missingness, "
            "measurement scale, and comparability before drawing conclusions."
        ),
        "board": (
            "Begin with a broad outcome appropriate for strategic monitoring, then "
            "refer operational follow-up questions to school leadership."
        ),
        "researcher": (
            "Begin with a descriptive analysis, then examine assumptions, uncertainty, "
            "and alternative explanations before making inferential claims."
        ),
        "educator": (
            "Begin with the clearest analysis that directly answers the question, then "
            "use follow-up comparisons to add context."
        ),
    }

    can_answer = {
        "strengths_needs": [
            "Which requested units are comparatively high, low, or unusual in the uploaded data.",
            "Whether patterns are consistent or uneven across available measures.",
            "Which observations warrant additional instructional or contextual evidence.",
        ],
        "relationship": [
            "Whether the two numeric measures tend to increase or decrease together.",
            "The direction and approximate strength of their linear association.",
            "Whether clusters or possible outliers are visible.",
        ],
        "trend": [
            "Whether the measured outcome increased, decreased, or remained stable.",
            "Which time periods show the largest changes.",
            "Whether groups appear to follow similar trajectories.",
        ],
        "ranking": [
            "Which category, construct, or group has the highest or lowest descriptive value.",
            "How large the differences are among the ranked values.",
        ],
        "distribution": [
            "Where scores are centered and how widely they vary.",
            "Whether groups overlap and whether possible outliers are present.",
        ],
        "composition": [
            "How the whole is divided among the displayed categories.",
            "Which groups have meaningfully different category shares.",
        ],
    }.get(intent.intent_id, [
        "The broad pattern visible in the uploaded dataset.",
        "Which groups or measures may deserve closer examination.",
    ])

    cannot_answer = {
        "strengths_needs": [
            "Whether a student or program definitively requires intervention without an appropriate criterion.",
            "Why the observed strength or need exists.",
            "Whether the pattern will persist beyond the available data.",
        ],
        "relationship": [
            "Whether one score causes the other.",
            "Why the relationship exists.",
            "Whether the pattern generalizes beyond the represented observations.",
        ],
        "trend": [
            "Why the change occurred without additional evidence.",
            "Whether changes reflect the same students unless the dataset tracks cohorts.",
        ],
        "ranking": [
            "Why one category received stronger ratings.",
            "Whether small differences are practically or statistically meaningful by themselves.",
        ],
        "distribution": [
            "Why individual observations differ.",
            "Whether group differences are caused by instruction or another factor.",
        ],
        "composition": [
            "Why category shares differ.",
            "Whether differences are stable across years without longitudinal data.",
        ],
    }.get(intent.intent_id, [
        "Why the observed pattern occurred.",
        "Causal conclusions without an appropriate research design.",
    ])

    next_questions = {
        "teacher": [
            "Which students or skills contribute most to this pattern?",
            "Does the pattern differ by grade, class, or instructional group?",
            "What additional evidence should inform an instructional response?",
        ],
        "coach": [
            "Which grade levels or teams show different patterns?",
            "What instructional evidence could explain the variation?",
            "Which finding is appropriate for collaborative inquiry?",
        ],
        "principal": [
            "Which grades or student groups contribute most to the schoolwide pattern?",
            "Is the pattern consistent across subjects or years?",
            "Where should leadership seek additional evidence before allocating support?",
        ],
        "district": [
            "Which schools, grades, or subjects differ most from the district pattern?",
            "Are the differences persistent across years?",
            "Where might district support or deeper review be warranted?",
        ],
        "assessment": [
            "Are the measures comparable and sufficiently complete?",
            "How sensitive is the finding to outliers or subgroup sample size?",
            "What additional statistic or diagnostic should accompany this chart?",
        ],
        "board": [
            "Is this pattern persistent enough to warrant strategic monitoring?",
            "What additional context should administration provide?",
            "Which outcome belongs on a governance-level dashboard?",
        ],
        "researcher": [
            "Which assumptions should be checked before inference?",
            "What uncertainty or effect-size information should accompany the result?",
            "Which rival explanations remain plausible?",
        ],
        "educator": [
            "Does the pattern differ across meaningful groups?",
            "Are there outliers or missing data influencing the result?",
            "What evidence would help explain the pattern?",
        ],
    }[role.role_id]

    caution = ""
    if len(df) < 10:
        caution = (
            "This dataset contains fewer than 10 rows. Treat comparisons and "
            "relationships as exploratory rather than stable findings."
        )
    elif intent.intent_id == "trend":
        date_like = [
            column for column in df.columns
            if any(token in str(column).lower() for token in ["year", "date", "window", "time"])
        ]
        if not date_like:
            caution = (
                "The prompt suggests change over time, but no clear time field was detected."
            )

    headline = f"Recommended first analysis for a {role.display_name}"
    recommendation = f"{primary.title}: {primary.rationale}"

    unit_sentence = (
        f"The requested unit is {target_unit.display_name.lower()}."
        if target_unit is not None
        else ""
    )
    basis_sentence = (
        f"The decision basis is {decision_basis.display_name.lower()}."
        if decision_basis is not None
        else ""
    )
    why_this_first = (
        f"{role_adjustments[role.role_id]} The dataset was recognized as "
        f"{profile.display_name}, and the prompt was classified as "
        f"{intent.display_name.lower()}. {unit_sentence} {basis_sentence}"
    ).strip()

    if decision_basis is not None and decision_basis.caution:
        caution = " ".join(
            part for part in [caution, decision_basis.caution] if part
        )

    return AnalystCoachPlan(
        role=role,
        headline=headline,
        target_unit=target_unit,
        decision_basis=decision_basis,
        recommendation=recommendation,
        why_this_first=why_this_first,
        can_answer=can_answer,
        cannot_answer=cannot_answer,
        next_questions=next_questions,
        alternatives=alternatives,
        caution=caution,
    )


def analyst_plan_prompt_block(plan: AnalystCoachPlan) -> str:
    alternatives = "\n".join(
        f"- Priority {item.priority}/5: {item.title} ({item.chart_family}) — {item.rationale}"
        for item in plan.alternatives
    )
    return f"""
ANALYST COACH PLAN
------------------
{plan.headline}
Primary recommendation: {plan.recommendation}
Why this comes first: {plan.why_this_first}

Candidate analyses:
{alternatives}

COACHING RULES
- Generate the primary recommended analysis unless the user explicitly asks for another.
- Use notes to explain why the analysis fits the question and role.
- Do not overstate what the chart can establish.
- Preserve statistical validity even when adapting the explanation to the user's role.
""".strip()
