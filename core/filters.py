"""Filtering utilities for VizCreate.

This module centralizes filters supplied by the LLM chart specification and
filters selected by the user in the Streamlit interface. UI filters take
precedence over specification filters for the same column.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import pandas as pd


LATEST_TOKEN = "__LATEST__"
EARLIEST_TOKEN = "__EARLIEST__"


def _as_filter_values(value: Any) -> list[Any]:
    """Return a filter value as a list without treating strings as iterables."""
    if isinstance(value, (list, tuple, set, pd.Series)):
        return list(value)
    return [value]


def _ordered_non_null_values(series: pd.Series) -> list[Any]:
    """Return unique, non-null values while preserving their first-seen order."""
    return series.dropna().drop_duplicates().tolist()


def resolve_special_filter_values(
    df: pd.DataFrame,
    filters: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve supported special tokens against values in the dataframe.

    Supported tokens are intentionally small in Phase 1:
    - ``__LATEST__`` resolves to the greatest available value in the column.
    - ``__EARLIEST__`` resolves to the smallest available value in the column.

    Values that cannot be resolved are preserved so validation/debugging can
    remain transparent. Ordinary filters are returned unchanged.
    """
    if not filters:
        return {}

    resolved: dict[str, Any] = {}

    for column, wanted in filters.items():
        if column not in df.columns:
            resolved[column] = wanted
            continue

        values = _ordered_non_null_values(df[column])
        if not values:
            resolved[column] = wanted
            continue

        def resolve_one(item: Any) -> Any:
            if item == LATEST_TOKEN:
                try:
                    return max(values)
                except TypeError:
                    return values[-1]
            if item == EARLIEST_TOKEN:
                try:
                    return min(values)
                except TypeError:
                    return values[0]
            return item

        if isinstance(wanted, (list, tuple, set, pd.Series)):
            resolved[column] = [resolve_one(item) for item in wanted]
        else:
            resolved[column] = resolve_one(wanted)

    return resolved


def apply_filters(
    df: pd.DataFrame,
    spec_filters: Optional[Mapping[str, Any]] = None,
    ui_filters: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Apply UI and chart-specification filters to a dataframe.

    UI filters are applied first and take precedence. When both sources mention
    the same column, the specification filter for that column is skipped.
    Missing columns are ignored to preserve the current VizCreate behavior.
    """
    filtered = df.copy()

    resolved_ui = resolve_special_filter_values(filtered, ui_filters)
    for column, wanted in resolved_ui.items():
        if column not in filtered.columns:
            continue

        values = _as_filter_values(wanted)
        filtered = filtered[filtered[column].isin(values)]

    resolved_spec = resolve_special_filter_values(filtered, spec_filters)
    for column, wanted in resolved_spec.items():
        if ui_filters and column in ui_filters:
            continue
        if column not in filtered.columns:
            continue

        values = _as_filter_values(wanted)
        filtered = filtered[filtered[column].isin(values)]

    return filtered


def apply_spec(
    df: pd.DataFrame,
    spec: Optional[Mapping[str, Any]],
    ui_filters: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    """Compatibility wrapper for the former in-app ``apply_spec`` helper."""
    spec_filters = spec.get("filters", {}) if spec else {}
    return apply_filters(
        df,
        spec_filters=spec_filters,
        ui_filters=ui_filters,
    )


def detect_filter_dimensions(
    df: pd.DataFrame,
    preferred_columns: Optional[Iterable[str]] = None,
    max_unique_values: int = 50,
) -> dict[str, list[Any]]:
    """Identify columns that are practical candidates for UI filtering.

    Preferred columns are returned first when available. Other categorical or
    low-cardinality columns are included when they have between 1 and
    ``max_unique_values`` unique non-null values.
    """
    preferred = list(
        preferred_columns
        or [
            "Grade",
            "Grade Level",
            "Testing Grade",
            "School Year",
            "Year",
            "Subject",
            "Subgroup",
        ]
    )

    ordered_columns: list[str] = []
    for column in preferred + list(df.columns):
        if column in df.columns and column not in ordered_columns:
            ordered_columns.append(column)

    dimensions: dict[str, list[Any]] = {}
    for column in ordered_columns:
        values = _ordered_non_null_values(df[column])
        if not values or len(values) > max_unique_values:
            continue

        is_preferred = column in preferred
        is_categorical = not pd.api.types.is_numeric_dtype(df[column])
        is_low_cardinality_numeric = len(values) <= 15

        if is_preferred or is_categorical or is_low_cardinality_numeric:
            dimensions[column] = values

    return dimensions


def default_ui_filters(
    df: pd.DataFrame,
    filter_columns: Optional[Iterable[str]] = None,
) -> dict[str, list[Any]]:
    """Build inclusive defaults for selected filter dimensions.

    Every available value is selected so adding the UI controls does not alter
    the data until the user changes a selection.
    """
    dimensions = detect_filter_dimensions(
        df,
        preferred_columns=filter_columns,
    )
    return {column: values.copy() for column, values in dimensions.items()}
