"""Shared dataset-profile structures for VizCreate Dataset Intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetProfileResult:
    """The result of evaluating one dataset profile."""

    profile_id: str
    display_name: str
    confidence: float
    description: str
    structure: str
    organizational_level: str | None = None
    detected_roles: dict[str, str] = field(default_factory=dict)
    recommended_charts: list[str] = field(default_factory=list)
    discouraged_charts: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    cautions: list[str] = field(default_factory=list)
    prompt_guidance: str = ""
    default_spec: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)

    @property
    def confidence_percent(self) -> int:
        return int(round(max(0.0, min(1.0, self.confidence)) * 100))


class DatasetProfile:
    """Base class implemented by every dataset profile."""

    profile_id = "base"
    display_name = "Base Profile"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        raise NotImplementedError


def clamp_score(value: float) -> float:
    """Clamp a score to the inclusive 0–1 range."""
    return max(0.0, min(1.0, float(value)))


def first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Return the first exact matching column from a candidate list."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def matching_columns(
    df: pd.DataFrame,
    terms: list[str],
) -> list[str]:
    """Return columns whose names contain one or more case-insensitive terms."""
    matches: list[str] = []
    for column in df.columns:
        text = str(column).lower()
        if any(term.lower() in text for term in terms):
            matches.append(str(column))
    return matches


def detect_organizational_level(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Detect a likely entity column and organizational level.

    Returns:
        (entity_column, organizational_level)
    """
    entity_candidates = [
        "Organization",
        "Entity",
        "School",
        "School Name",
        "District",
        "District Name",
        "Site",
        "Building",
        "Location",
        "Comparison Group",
    ]
    entity_column = first_existing_column(df, entity_candidates)

    if entity_column is None:
        return None, None

    values = {
        str(value).strip().lower()
        for value in df[entity_column].dropna().unique()
    }

    has_school = any("school" in value or "elementary" in value or "middle" in value or "high" in value for value in values)
    has_district = any("district" in value for value in values)
    has_state = any(value == "state" or "wyoming" in value or "statewide" in value for value in values)

    levels = []
    if has_school:
        levels.append("school")
    if has_district:
        levels.append("district")
    if has_state:
        levels.append("state")

    if len(levels) > 1:
        return entity_column, "comparison across " + ", ".join(levels)
    if levels:
        return entity_column, levels[0]

    return entity_column, "organizational entity"
