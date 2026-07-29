"""Phase 2F: infer executable dataset constraints from narrow user requests."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

import pandas as pd

from core.filters import apply_filters


@dataclass
class IntentFilter:
    column: str
    value: Any
    source_phrase: str
    confidence: float
    reason: str


@dataclass
class IntentFilterResult:
    filters: dict[str, Any] = field(default_factory=dict)
    inferred: list[IntentFilter] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    original_rows: int = 0
    filtered_rows: int = 0

    @property
    def applied(self) -> bool:
        return bool(self.filters)

    @property
    def scope_text(self) -> str:
        if not self.filters:
            return "No additional prompt constraints were applied."
        parts = []
        for column, value in self.filters.items():
            if isinstance(value, list):
                rendered = ", ".join(str(item) for item in value)
            else:
                rendered = str(value)
            parts.append(f"{column}: {rendered}")
        return " · ".join(parts)


GRADE_COLUMNS = (
    "grade", "grade level", "testing grade", "tested grade", "student grade",
)
SUBJECT_COLUMNS = (
    "subject", "content area", "test subject", "assessment subject",
)
YEAR_COLUMNS = (
    "school year", "academic year", "year", "test year",
)
SCHOOL_COLUMNS = (
    "school", "school name", "site", "campus",
)
SUBGROUP_COLUMNS = (
    "subgroup", "student group", "demographic group", "reporting group",
)

SUBJECT_ALIASES = {
    "math": ("math", "mathematics"),
    "mathematics": ("math", "mathematics"),
    "ela": ("ela", "english language arts", "language arts", "reading"),
    "reading": ("reading", "ela", "english language arts", "language arts"),
    "science": ("science",),
    "writing": ("writing",),
}

ORDINAL_TO_GRADE = {
    "kindergarten": 0,
    "kinder": 0,
    "first": 1,
    "1st": 1,
    "second": 2,
    "2nd": 2,
    "third": 3,
    "3rd": 3,
    "fourth": 4,
    "4th": 4,
    "fifth": 5,
    "5th": 5,
    "sixth": 6,
    "6th": 6,
    "seventh": 7,
    "7th": 7,
    "eighth": 8,
    "8th": 8,
    "ninth": 9,
    "9th": 9,
    "tenth": 10,
    "10th": 10,
    "eleventh": 11,
    "11th": 11,
    "twelfth": 12,
    "12th": 12,
}


def _normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _candidate_columns(df: pd.DataFrame, aliases: tuple[str, ...]) -> list[str]:
    ranked = []
    for column in df.columns:
        normalized = _normalize(column)
        score = 0
        for alias in aliases:
            if normalized == alias:
                score = max(score, 100)
            elif alias in normalized:
                score = max(score, 70)
        if score:
            ranked.append((score, str(column)))
    return [column for _, column in sorted(ranked, reverse=True)]


def _unique_values(df: pd.DataFrame, column: str) -> list[Any]:
    return df[column].dropna().drop_duplicates().tolist()


def _grade_from_prompt(prompt: str) -> tuple[int | None, str]:
    normalized = _normalize(prompt)
    for phrase, grade in ORDINAL_TO_GRADE.items():
        if re.search(rf"\b{re.escape(phrase)}(?:\s+grade|\s+grader|\s+graders)?\b", normalized):
            return grade, phrase

    match = re.search(r"\bgrade\s*(k|[0-9]{1,2})\b", normalized)
    if match:
        raw = match.group(1)
        return (0 if raw == "k" else int(raw)), match.group(0)

    match = re.search(r"\b(k|[0-9]{1,2})(?:st|nd|rd|th)?\s*grade\b", normalized)
    if match:
        raw = match.group(1)
        return (0 if raw == "k" else int(raw)), match.group(0)

    return None, ""


def _match_grade_value(values: list[Any], grade: int) -> Any | None:
    candidates = []
    for value in values:
        normalized = _normalize(value)
        if grade == 0 and normalized in {"k", "kg", "kindergarten", "grade k"}:
            return value
        numbers = [int(item) for item in re.findall(r"\b([0-9]{1,2})\b", normalized)]
        if grade in numbers:
            candidates.append(value)
        elif isinstance(value, (int, float)) and float(value) == float(grade):
            candidates.append(value)
    return candidates[0] if candidates else None


def _subject_from_prompt(prompt: str) -> tuple[list[str], str]:
    normalized = _normalize(prompt)
    for phrase, aliases in SUBJECT_ALIASES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", normalized):
            return list(aliases), phrase
    return [], ""


def _match_text_value(values: list[Any], aliases: list[str]) -> Any | None:
    best = None
    best_score = 0
    for value in values:
        normalized = _normalize(value)
        for alias in aliases:
            alias_normalized = _normalize(alias)
            score = 0
            if normalized == alias_normalized:
                score = 100
            elif alias_normalized in normalized:
                score = 80
            elif normalized in alias_normalized:
                score = 60
            if score > best_score:
                best = value
                best_score = score
    return best



def _year_sort_key(value: Any) -> tuple[int, int, str]:
    """Sort calendar years and school-year labels chronologically."""
    text = str(value)
    numbers = [int(item) for item in re.findall(r"\b(19\d{2}|20\d{2}|\d{2})\b", text)]
    expanded = []
    for number in numbers:
        if number < 100:
            number += 2000
        expanded.append(number)
    if expanded:
        return (expanded[0], expanded[-1], text)
    return (-1, -1, text)


def _relative_year_request(prompt: str) -> tuple[int | None, str]:
    normalized = _normalize(prompt)

    if re.search(
        r"\b(most recent|latest|current)\s+(school\s+|academic\s+|test\s+)?year\b",
        normalized,
    ):
        return 1, "most recent year"

    match = re.search(
        r"\b(?:last|most recent|latest|previous|past)\s+"
        r"(one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?:school\s+|academic\s+|test\s+)?years?\b",
        normalized,
    )
    if match:
        word = match.group(1)
        number_words = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        }
        count = number_words.get(word, int(word) if word.isdigit() else None)
        return count, match.group(0)

    return None, ""


def _resolve_recent_years(values: list[Any], count: int) -> list[Any]:
    ordered = sorted(values, key=_year_sort_key)
    return ordered[-count:] if count > 0 else []


def _explicit_value_mentions(
    df: pd.DataFrame,
    prompt: str,
    excluded_columns: set[str],
) -> list[IntentFilter]:
    """Detect exact categorical values named in the prompt.

    This safely extends filtering to schools, subgroups, years, and other
    low-cardinality dimensions without asking the LLM to invent values.
    """
    normalized_prompt = f" {_normalize(prompt)} "
    findings: list[IntentFilter] = []

    for column in df.columns:
        column = str(column)
        if column in excluded_columns:
            continue
        series = df[column]
        unique_count = int(series.nunique(dropna=True))
        if unique_count < 1 or unique_count > 40:
            continue
        if pd.api.types.is_numeric_dtype(series) and unique_count > 15:
            continue

        matches = []
        for value in _unique_values(df, column):
            value_text = _normalize(value)
            if len(value_text) < 3:
                continue
            if f" {value_text} " in normalized_prompt:
                matches.append(value)

        # Apply only an unambiguous exact value mention for a dimension.
        if len(matches) == 1:
            findings.append(IntentFilter(
                column=column,
                value=matches[0],
                source_phrase=str(matches[0]),
                confidence=0.94,
                reason="The prompt exactly names a value found in this dataset.",
            ))
    return findings


def infer_intent_filters(
    df: pd.DataFrame,
    prompt: str,
    existing_filters: dict[str, Any] | None = None,
) -> IntentFilterResult:
    """Infer defensible filters using only columns and values present in `df`."""
    existing_filters = dict(existing_filters or {})
    result = IntentFilterResult(
        filters=dict(existing_filters),
        original_rows=len(df),
        filtered_rows=len(df),
    )
    used_columns = set(existing_filters)

    grade, grade_phrase = _grade_from_prompt(prompt)
    grade_columns = _candidate_columns(df, GRADE_COLUMNS)
    if grade is not None and grade_columns:
        column = grade_columns[0]
        if column not in used_columns:
            value = _match_grade_value(_unique_values(df, column), grade)
            if value is not None:
                result.filters[column] = value
                used_columns.add(column)
                result.inferred.append(IntentFilter(
                    column=column,
                    value=value,
                    source_phrase=grade_phrase,
                    confidence=0.99,
                    reason=f"The prompt explicitly requests grade {grade}.",
                ))
            else:
                result.rejected.append(
                    f"The prompt requested grade {grade}, but no matching value was found in {column}."
                )

    subject_aliases, subject_phrase = _subject_from_prompt(prompt)
    subject_columns = _candidate_columns(df, SUBJECT_COLUMNS)
    if subject_aliases and subject_columns:
        column = subject_columns[0]
        if column not in used_columns:
            value = _match_text_value(_unique_values(df, column), subject_aliases)
            if value is not None:
                result.filters[column] = value
                used_columns.add(column)
                result.inferred.append(IntentFilter(
                    column=column,
                    value=value,
                    source_phrase=subject_phrase,
                    confidence=0.99,
                    reason=f"The prompt explicitly requests {subject_phrase}.",
                ))
            else:
                result.rejected.append(
                    f"The prompt requested {subject_phrase}, but no matching value was found in {column}."
                )

    recent_year_count, recent_year_phrase = _relative_year_request(prompt)
    year_columns = _candidate_columns(df, YEAR_COLUMNS)
    if recent_year_count is not None and year_columns:
        column = year_columns[0]
        if column not in used_columns:
            available_years = _unique_values(df, column)
            selected_years = _resolve_recent_years(
                available_years,
                min(recent_year_count, len(available_years)),
            )
            if selected_years:
                value = selected_years[0] if recent_year_count == 1 else selected_years
                result.filters[column] = value
                used_columns.add(column)
                result.inferred.append(IntentFilter(
                    column=column,
                    value=value,
                    source_phrase=recent_year_phrase,
                    confidence=0.99,
                    reason=(
                        f"The prompt requests the {recent_year_phrase}; VizCreate "
                        "resolved this against the available dataset years."
                    ),
                ))
            else:
                result.rejected.append(
                    f"The prompt requested the {recent_year_phrase}, but no usable values were found in {column}."
                )

    # Extend to exact values such as a named school, subgroup, or school year.
    for inferred in _explicit_value_mentions(df, prompt, used_columns):
        result.filters[inferred.column] = inferred.value
        used_columns.add(inferred.column)
        result.inferred.append(inferred)

    filtered = apply_filters(df, spec_filters=result.filters)
    if filtered.empty and result.inferred:
        # Never silently replace a valid visualization with an empty scope.
        result.rejected.append(
            "The combined inferred constraints returned no rows, so VizCreate retained only the planner's existing filters."
        )
        result.filters = existing_filters
        result.inferred = []
        filtered = apply_filters(df, spec_filters=result.filters)

    result.filtered_rows = len(filtered)
    return result


def enrich_spec_with_intent_filters(
    df: pd.DataFrame,
    prompt: str,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], IntentFilterResult]:
    """Merge prompt constraints into a chart specification."""
    enriched = dict(spec or {})
    existing_filters = enriched.get("filters", {})
    if not isinstance(existing_filters, dict):
        existing_filters = {}

    result = infer_intent_filters(df, prompt, existing_filters=existing_filters)
    enriched["filters"] = result.filters
    enriched["intent_filter_metadata"] = {
        "scope_text": result.scope_text,
        "original_rows": result.original_rows,
        "filtered_rows": result.filtered_rows,
        "inferred": [
            {
                "column": item.column,
                "value": item.value,
                "source_phrase": item.source_phrase,
                "confidence": item.confidence,
                "reason": item.reason,
            }
            for item in result.inferred
        ],
        "rejected": list(result.rejected),
    }
    return enriched, result
