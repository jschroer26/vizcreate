"""Phase 3B.1: bridge Data Analyst Coach ideas into the investigation flow."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from core.dive_deeper import InvestigationState
from core.role_coach import AnalystCoachPlan


@dataclass
class CoachBridgeSuggestion:
    suggestion_id: str
    title: str
    prompt: str
    rationale: str
    suggestion_type: str
    chart_family: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _token_set(text: str) -> set[str]:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "how",
        "in", "is", "of", "on", "or", "the", "this", "to", "using", "what",
        "which", "with",
    }
    return {
        token for token in _normalize(text).split()
        if len(token) > 2 and token not in stop_words
    }


def _is_duplicate(candidate: str, comparison_texts: list[str]) -> bool:
    normalized = _normalize(candidate)
    candidate_tokens = _token_set(candidate)
    if not normalized:
        return True

    for existing in comparison_texts:
        existing_normalized = _normalize(existing)
        if not existing_normalized:
            continue
        if normalized == existing_normalized:
            return True
        if len(normalized) >= 24 and (
            normalized in existing_normalized or existing_normalized in normalized
        ):
            return True

        existing_tokens = _token_set(existing)
        union = candidate_tokens | existing_tokens
        if union:
            overlap = len(candidate_tokens & existing_tokens) / len(union)
            if overlap >= 0.72:
                return True
    return False


def _history_texts(state: InvestigationState) -> list[str]:
    texts: list[str] = []
    for step in state.history:
        texts.extend([step.title, step.prompt])
    return texts


def _analysis_prompt(title: str, rationale: str, chart_family: str) -> str:
    chart_instruction = (
        f" Use a {chart_family} if it is appropriate for the available fields."
        if chart_family else ""
    )
    return (
        f"Investigate this question: {title}. {rationale}{chart_instruction} "
        "Choose the clearest defensible visualization and explain what the evidence can and cannot show."
    )


def generate_coach_bridge_suggestions(
    coach_plan: AnalystCoachPlan,
    state: InvestigationState,
    current_prompt: str = "",
    current_spec: dict[str, Any] | None = None,
    limit: int = 5,
) -> list[CoachBridgeSuggestion]:
    """Return non-duplicate, executable Coach ideas for the current investigation."""
    comparison_texts = [current_prompt] + _history_texts(state)
    current_chart = _normalize((current_spec or {}).get("chart_type", ""))
    suggestions: list[CoachBridgeSuggestion] = []

    # Candidate analyses are sorted by the priority already assigned by the Coach.
    ordered_analyses = sorted(
        coach_plan.alternatives,
        key=lambda item: item.priority,
        reverse=True,
    )
    for index, item in enumerate(ordered_analyses):
        candidate_text = f"{item.title} {item.rationale}"
        if _is_duplicate(candidate_text, comparison_texts):
            continue

        # A repeated chart family is still allowed when the analytical question is new,
        # but its rationale explicitly needs to offer a different perspective.
        chart_family = item.chart_family or ""
        prompt = _analysis_prompt(item.title, item.rationale, chart_family)
        suggestion = CoachBridgeSuggestion(
            suggestion_id=f"coach_analysis_{index}_{_normalize(item.title)[:32].replace(' ', '_')}",
            title=item.title,
            prompt=prompt,
            rationale=item.rationale,
            suggestion_type="alternative_analysis",
            chart_family=chart_family,
        )
        suggestions.append(suggestion)
        comparison_texts.extend([suggestion.title, suggestion.prompt])
        if len(suggestions) >= limit:
            return suggestions

    for index, question in enumerate(coach_plan.next_questions):
        if _is_duplicate(question, comparison_texts):
            continue
        prompt = (
            f"{question} Use the available dataset to select an appropriate analysis and visualization. "
            "Explain the observed pattern cautiously and identify any important limitations."
        )
        suggestion = CoachBridgeSuggestion(
            suggestion_id=f"coach_question_{index}_{_normalize(question)[:32].replace(' ', '_')}",
            title=question.rstrip(" ?"),
            prompt=prompt,
            rationale=(
                "This question broadens the investigation beyond the strongest guided path "
                "and may reveal a different explanation or decision-relevant pattern."
            ),
            suggestion_type="next_question",
        )
        suggestions.append(suggestion)
        comparison_texts.extend([suggestion.title, suggestion.prompt])
        if len(suggestions) >= limit:
            break

    return suggestions
