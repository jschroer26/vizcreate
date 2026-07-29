"""Intelligent schema normalization for VizCreate.

Phase 2H accepts authentic assessment exports without asking users to rename
columns.  It cleans header formatting, applies conservative semantic aliases,
and classifies common educational-data value patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd


CANONICAL_ALIASES: dict[str, str] = {
    # Time and organization
    "school year": "School Year",
    "academic year": "School Year",
    "district": "District Name",
    "district name": "District Name",
    "school": "School Name",
    "school name": "School Name",
    # Grade and subject
    "grade": "Grade",
    "specific grade": "Grade",
    "grade level": "Grade",
    "student grade": "Grade",
    "subject": "Subject",
    "content area": "Subject",
    # Counts
    "number of students tested": "Number of Students Tested",
    "no of students tested": "Number of Students Tested",
    "students tested": "Number of Students Tested",
    "n tested": "Number of Students Tested",
    # WYTOPP proficiency measures
    "percent basic and below": "Percent Basic and Below",
    "percent proficient and advanced": "Percent Proficient and Advanced",
    "percent below basic": "Percent Below Basic",
    "percent basic": "Percent Basic",
    "percent proficient": "Percent Proficient",
    "percent advanced": "Percent Advanced",
    "participation rate": "Participation Rate",
    "subgroup": "Subgroup",
}


@dataclass
class SchemaNormalizationResult:
    """Metadata describing changes made during normalization."""

    renamed_columns: dict[str, str] = field(default_factory=dict)
    column_types: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def clean_header(value: Any) -> str:
    """Collapse whitespace and trim a header while preserving readable text."""
    return re.sub(r"\s+", " ", str(value)).strip()


def matching_key(value: Any) -> str:
    """Create a punctuation-insensitive key used only for alias matching."""
    text = clean_header(value).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bno\.?\b", "number", text)
    text = re.sub(r"[%]", " percent ", text)
    text = re.sub(r"[_\-/]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonical_column_name(column: Any) -> str:
    """Return a canonical name when an approved alias is recognized."""
    cleaned = clean_header(column)
    return CANONICAL_ALIASES.get(matching_key(cleaned), cleaned)


def _nonempty_strings(series: pd.Series) -> pd.Series:
    values = series.dropna().astype(str).str.strip()
    return values[values.ne("")]


def classify_column_values(series: pd.Series, column_name: str = "") -> str:
    """Classify common educational-data value patterns conservatively."""
    values = _nonempty_strings(series)
    if values.empty:
        return "EMPTY"

    year_pattern = re.compile(r"^(?:19|20)\d{2}\s*[-–—/]\s*(?:\d{2}|(?:19|20)\d{2})$")
    year_share = values.map(lambda value: bool(year_pattern.fullmatch(value))).mean()
    if year_share >= 0.8:
        return "SCHOOL_YEAR"

    # Official Wyoming exports use ranges such as 290 - 299 when counts are
    # suppressed or banded. These must not be treated as exact sample sizes.
    # Require a count-like header so year labels are never mistaken for counts.
    range_pattern = re.compile(r"^\d+\s*[-–—]\s*\d+$")
    range_share = values.map(lambda value: bool(range_pattern.fullmatch(value))).mean()
    count_like_name = any(term in matching_key(column_name) for term in ("tested", "count", "number", " n "))
    if range_share >= 0.8 and count_like_name:
        return "SUPPRESSED_COUNT_RANGE"

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_share = numeric.notna().mean()
    if numeric_share >= 0.8:
        name_key = matching_key(column_name)
        finite = numeric.dropna()
        if "percent" in name_key or "rate" in name_key:
            return "PERCENT"
        if not finite.empty and (finite % 1 == 0).all():
            return "INTEGER"
        return "NUMERIC"

    if matching_key(column_name) == "grade":
        return "GRADE_LEVEL"
    return "TEXT"


def normalize_dataframe_schema(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, SchemaNormalizationResult]:
    """Clean and normalize a dataframe schema without altering data values.

    Duplicate canonical names are avoided.  If two source columns would map to
    the same canonical label, the first receives the canonical name and the
    later column keeps its cleaned original label.  A warning records this.
    """
    normalized = df.copy()
    result = SchemaNormalizationResult()
    used_names: set[str] = set()
    new_columns: list[str] = []

    for original in normalized.columns:
        cleaned = clean_header(original)
        candidate = canonical_column_name(cleaned)
        final = candidate
        if final in used_names:
            final = cleaned
            if final in used_names:
                index = 2
                while f"{cleaned} ({index})" in used_names:
                    index += 1
                final = f"{cleaned} ({index})"
            result.warnings.append(
                f"Multiple columns mapped to '{candidate}'. Kept '{final}' distinct."
            )
        used_names.add(final)
        new_columns.append(final)
        if str(original) != final:
            result.renamed_columns[str(original)] = final

    normalized.columns = new_columns
    result.column_types = {
        str(column): classify_column_values(normalized[column], str(column))
        for column in normalized.columns
    }

    suppressed = [
        column
        for column, value_type in result.column_types.items()
        if value_type == "SUPPRESSED_COUNT_RANGE"
    ]
    if suppressed:
        result.warnings.append(
            "Suppressed or banded student-count ranges were detected and will "
            "not be treated as exact sample sizes: " + ", ".join(suppressed)
        )

    # Keep metadata available to downstream UI without changing the public
    # loader return type.
    normalized.attrs["schema_normalization"] = {
        "renamed_columns": result.renamed_columns,
        "column_types": result.column_types,
        "warnings": result.warnings,
    }
    return normalized, result
