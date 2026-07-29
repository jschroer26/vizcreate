
"""Generic subject and measure hierarchy detection for VizCreate.

Phase 2G.3 Revised distinguishes:
- subject families (Math, ELA, Science, and other detectable subjects),
- overall measures,
- category/subscore measures,
- requested measure scope from natural-language prompts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

import pandas as pd


@dataclass
class MeasureFamily:
    subject_id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    overall: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


@dataclass
class MeasureHierarchyResult:
    families: dict[str, MeasureFamily]
    unmatched_numeric_columns: list[str] = field(default_factory=list)


@dataclass
class MeasureScopeResult:
    subject_id: str | None
    scope: str
    requested_terms: list[str] = field(default_factory=list)
    confidence: float = 0.0


SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "math": (
        "math",
        "mathematics",
        "mathematical",
    ),
    "ela": (
        "ela",
        "reading/ela",
        "reading ela",
        "english language arts",
        "language arts",
        "reading",
        "literacy",
    ),
    "science": (
        "science",
        "scientific",
        "sci",
    ),
}

SUBJECT_DISPLAY_NAMES = {
    "math": "Math",
    "ela": "ELA/Reading",
    "science": "Science",
}

OVERALL_TERMS = (
    "scale score",
    "overall",
    "total score",
    "composite",
    "combined score",
    "overall score",
    "subject score",
)

CATEGORY_SCOPE_TERMS = (
    "all categories",
    "all category",
    "categories",
    "category",
    "all subscores",
    "subscores",
    "subscore",
    "all domains",
    "domains",
    "domain",
    "reasoning skills",
    "reasoning skill",
    "strands",
    "strand",
    "standards",
    "standard areas",
    "components",
    "component scores",
)

ALL_MEASURE_TERMS = (
    "all measures",
    "every measure",
    "all scores",
    "every score",
)

OVERALL_SCOPE_TERMS = (
    "overall score",
    "scale score",
    "composite",
    "total score",
)

EXCLUDED_ROLE_TERMS = (
    "id",
    "identifier",
    "student number",
    "student #",
    "grade level",
    "testing grade",
    "school year",
    "year",
    "enrollment",
    "number tested",
    "n tested",
    "age",
)


def normalize_column_name(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[/_:|\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _contains_phrase(text: str, phrase: str) -> bool:
    phrase_norm = normalize_column_name(phrase)
    if not phrase_norm:
        return False
    return phrase_norm in text


def detect_subject_from_text(text: str) -> str | None:
    normalized = normalize_column_name(text)
    best_subject = None
    best_length = -1

    for subject_id, aliases in SUBJECT_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_column_name(alias)
            if alias_norm and alias_norm in normalized and len(alias_norm) > best_length:
                best_subject = subject_id
                best_length = len(alias_norm)

    return best_subject


def _looks_like_excluded_role(column: str) -> bool:
    normalized = normalize_column_name(column)
    return any(normalized == normalize_column_name(term) for term in EXCLUDED_ROLE_TERMS)


def is_overall_measure(column: str, subject_id: str) -> bool:
    normalized = normalize_column_name(column)
    aliases = SUBJECT_ALIASES.get(subject_id, ())
    contains_subject = any(normalize_column_name(alias) in normalized for alias in aliases)
    if not contains_subject:
        return False

    if any(normalize_column_name(term) in normalized for term in OVERALL_TERMS):
        return True

    # A bare subject label is treated as an overall measure rather than a category.
    return normalized in {normalize_column_name(alias) for alias in aliases}


def _profile_role_columns(profile, role_name: str) -> list[str]:
    if profile is None:
        return []
    detected_roles = getattr(profile, "detected_roles", {}) or {}
    value = detected_roles.get(role_name)

    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def detect_measure_families(
    df: pd.DataFrame,
    profile=None,
) -> MeasureHierarchyResult:
    families: dict[str, MeasureFamily] = {}
    unmatched: list[str] = []

    # Build families from numeric columns.
    for column in df.columns:
        column_name = str(column)
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue
        if _looks_like_excluded_role(column_name):
            continue

        subject_id = detect_subject_from_text(column_name)
        if subject_id is None:
            unmatched.append(column_name)
            continue

        family = families.setdefault(
            subject_id,
            MeasureFamily(
                subject_id=subject_id,
                display_name=SUBJECT_DISPLAY_NAMES.get(
                    subject_id,
                    subject_id.replace("_", " ").title(),
                ),
                aliases=list(SUBJECT_ALIASES.get(subject_id, ())),
            ),
        )

        if is_overall_measure(column_name, subject_id):
            if column_name not in family.overall:
                family.overall.append(column_name)
        else:
            if column_name not in family.categories:
                family.categories.append(column_name)

    # Use profile roles as supplemental evidence when available.
    role_map = {
        "math": ("math_measures", "math_columns"),
        "ela": (
            "ela_measures",
            "reading_measures",
            "literacy_measures",
            "ela_columns",
            "reading_columns",
        ),
        "science": ("science_measures", "science_columns"),
    }

    for subject_id, role_names in role_map.items():
        role_columns: list[str] = []
        for role_name in role_names:
            role_columns.extend(_profile_role_columns(profile, role_name))

        for column_name in role_columns:
            if column_name not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[column_name]):
                continue

            family = families.setdefault(
                subject_id,
                MeasureFamily(
                    subject_id=subject_id,
                    display_name=SUBJECT_DISPLAY_NAMES.get(subject_id, subject_id.title()),
                    aliases=list(SUBJECT_ALIASES.get(subject_id, ())),
                ),
            )
            if is_overall_measure(column_name, subject_id):
                if column_name not in family.overall:
                    family.overall.append(column_name)
            elif column_name not in family.categories:
                family.categories.append(column_name)

    return MeasureHierarchyResult(
        families=families,
        unmatched_numeric_columns=unmatched,
    )


def detect_measure_scope(prompt: str) -> MeasureScopeResult:
    prompt_normalized = normalize_column_name(prompt)
    subject_id = detect_subject_from_text(prompt)

    category_matches = [
        term for term in CATEGORY_SCOPE_TERMS
        if normalize_column_name(term) in prompt_normalized
    ]
    all_measure_matches = [
        term for term in ALL_MEASURE_TERMS
        if normalize_column_name(term) in prompt_normalized
    ]
    # Accept a subject name between "all" and "measures/scores",
    # as in "all Math measures" or "all Science scores."
    prompt_tokens = set(prompt_normalized.split())
    if (
        "all" in prompt_tokens
        and ("measures" in prompt_tokens or "scores" in prompt_tokens)
        and not all_measure_matches
    ):
        all_measure_matches = ["all subject measures"]
    overall_matches = [
        term for term in OVERALL_SCOPE_TERMS
        if normalize_column_name(term) in prompt_normalized
    ]

    if category_matches:
        return MeasureScopeResult(
            subject_id=subject_id,
            scope="categories",
            requested_terms=category_matches,
            confidence=0.98 if subject_id else 0.80,
        )

    if all_measure_matches:
        return MeasureScopeResult(
            subject_id=subject_id,
            scope="all_measures",
            requested_terms=all_measure_matches,
            confidence=0.90 if subject_id else 0.72,
        )

    if overall_matches:
        return MeasureScopeResult(
            subject_id=subject_id,
            scope="overall",
            requested_terms=overall_matches,
            confidence=0.95 if subject_id else 0.70,
        )

    return MeasureScopeResult(
        subject_id=subject_id,
        scope="unspecified",
        requested_terms=[],
        confidence=0.65 if subject_id else 0.0,
    )


def resolve_measure_columns(
    hierarchy: MeasureHierarchyResult,
    scope: MeasureScopeResult,
) -> list[str]:
    if scope.subject_id is None:
        return []

    family = hierarchy.families.get(scope.subject_id)
    if family is None:
        return []

    if scope.scope == "categories":
        return list(family.categories)
    if scope.scope == "overall":
        return list(family.overall)
    if scope.scope == "all_measures":
        return list(family.overall) + list(family.categories)

    # For a subject-only request, prefer an overall measure when available.
    if family.overall:
        return list(family.overall[:1])
    return list(family.categories[:1])


def measure_hierarchy_prompt_block(
    hierarchy: MeasureHierarchyResult,
) -> str:
    if not hierarchy.families:
        return "No subject-specific measure families were detected."

    lines = ["Detected subject and measure hierarchy:"]
    for family in hierarchy.families.values():
        lines.append(f"- {family.display_name}")
        if family.overall:
            lines.append("  Overall measures: " + ", ".join(family.overall))
        if family.categories:
            lines.append("  Category/subscore measures: " + ", ".join(family.categories))
    return "\n".join(lines)
