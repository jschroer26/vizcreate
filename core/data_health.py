"""Deterministic data-health checks for uploaded VizCreate datasets."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _check(level: str, title: str, detail: str) -> dict[str, str]:
    return {"level": level, "title": title, "detail": detail}


def assess_dataset_health(
    df: pd.DataFrame,
    *,
    family_name: str = "",
) -> dict[str, Any]:
    """
    Inspect an uploaded dataframe and return a concise health summary.

    Levels:
    - pass: the condition looks ready
    - info: useful limitation or context
    - warning: the user should review the data
    - error: the dataset cannot be analyzed reliably
    """
    checks: list[dict[str, str]] = []

    row_count = len(df)
    column_count = len(df.columns)

    if row_count == 0:
        checks.append(
            _check(
                "error",
                "No data rows",
                "The file contains column headers but no usable data rows.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "Data rows detected",
                f"{row_count:,} rows are available for analysis.",
            )
        )

    if column_count < 2:
        checks.append(
            _check(
                "error",
                "Too few columns",
                "At least two columns are usually needed to create a meaningful visualization.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "Multiple fields detected",
                f"{column_count:,} columns are available.",
            )
        )

    duplicate_columns = pd.Index(df.columns).duplicated(keep=False)
    duplicated_names = sorted(
        set(pd.Index(df.columns)[duplicate_columns].astype(str))
    )

    if duplicated_names:
        checks.append(
            _check(
                "error",
                "Duplicate column headings",
                "Rename repeated headings before analysis: "
                + ", ".join(duplicated_names),
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "Column headings are unique",
                "No duplicate column names were found.",
            )
        )

    total_cells = max(row_count * column_count, 1)
    missing_cells = int(df.isna().sum().sum())
    missing_rate = missing_cells / total_cells

    if missing_cells == 0:
        checks.append(
            _check(
                "pass",
                "No missing cells detected",
                "All displayed cells contain values.",
            )
        )
    elif missing_rate <= 0.05:
        checks.append(
            _check(
                "info",
                "A small number of values are missing",
                f"{missing_cells:,} cells are blank ({missing_rate:.1%} of the dataset).",
            )
        )
    else:
        checks.append(
            _check(
                "warning",
                "Missing data should be reviewed",
                f"{missing_cells:,} cells are blank ({missing_rate:.1%} of the dataset).",
            )
        )

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        checks.append(
            _check(
                "warning",
                "Duplicate rows detected",
                f"{duplicate_rows:,} completely duplicated rows may affect summaries.",
            )
        )
    else:
        checks.append(
            _check(
                "pass",
                "No exact duplicate rows",
                "No completely duplicated records were found.",
            )
        )

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    if numeric_columns:
        checks.append(
            _check(
                "pass",
                "Numeric measures detected",
                f"{len(numeric_columns)} numeric column(s) can support charts and summaries.",
            )
        )
    else:
        checks.append(
            _check(
                "warning",
                "No numeric columns detected",
                "Most VizCreate charts require at least one numeric measure.",
            )
        )

    constant_columns = [
        str(column)
        for column in df.columns
        if df[column].nunique(dropna=True) <= 1
    ]
    if constant_columns:
        checks.append(
            _check(
                "info",
                "Constant fields detected",
                "These fields contain one unique value and may be useful only as filters: "
                + ", ".join(constant_columns[:6])
                + ("…" if len(constant_columns) > 6 else ""),
            )
        )

    if family_name in {"WYTOPP longitudinal proficiency summary", "WYTOPP current-year proficiency summary"}:
        required = [
            "School Year",
            "Grade",
            "Subject",
            "Percent Basic and Below",
            "Percent Proficient and Advanced",
        ]
        missing_required = [
            column for column in required if column not in df.columns
        ]

        if missing_required:
            checks.append(
                _check(
                    "error",
                    "Required WYTOPP fields are missing",
                    "Missing: " + ", ".join(missing_required),
                )
            )
        else:
            checks.append(
                _check(
                    "pass",
                    "Required WYTOPP fields found",
                    "The core longitudinal proficiency columns are available.",
                )
            )

            critical_missing = int(df[required].isna().sum().sum())
            if critical_missing:
                checks.append(
                    _check(
                        "warning",
                        "Blank values in required WYTOPP fields",
                        f"{critical_missing:,} required-field cells are blank.",
                    )
                )

            year_count = int(
                df["School Year"]
                .dropna()
                .astype(str)
                .nunique()
            )
            if year_count >= 2:
                checks.append(
                    _check(
                        "pass",
                        "Longitudinal analysis is supported",
                        f"{year_count} school years are available.",
                    )
                )
            elif year_count == 1:
                checks.append(
                    _check(
                        "info",
                        "Single-year dataset",
                        "Current-status comparisons are supported, but growth cannot be calculated.",
                    )
                )
            else:
                checks.append(
                    _check(
                        "error",
                        "No school-year values",
                        "A School Year value is required for longitudinal analysis.",
                    )
                )

            percent_columns = [
                "Percent Basic and Below",
                "Percent Proficient and Advanced",
            ]
            invalid_percent_count = 0
            for column in percent_columns:
                numeric = pd.to_numeric(df[column], errors="coerce")
                invalid_percent_count += int(
                    ((numeric < 0) | (numeric > 100)).sum()
                )

            if invalid_percent_count:
                checks.append(
                    _check(
                        "warning",
                        "Percentage values outside 0–100",
                        f"{invalid_percent_count:,} proficiency values should be reviewed.",
                    )
                )
            else:
                checks.append(
                    _check(
                        "pass",
                        "Percentage ranges look valid",
                        "The two proficiency fields fall between 0 and 100.",
                    )
                )

            pair_total = (
                pd.to_numeric(
                    df["Percent Basic and Below"],
                    errors="coerce",
                )
                + pd.to_numeric(
                    df["Percent Proficient and Advanced"],
                    errors="coerce",
                )
            )
            non_hundred = int(
                (
                    pair_total.notna()
                    & ~np.isclose(pair_total, 100, atol=1.0)
                ).sum()
            )

            if non_hundred:
                checks.append(
                    _check(
                        "info",
                        "Some proficiency pairs do not total 100%",
                        f"{non_hundred:,} rows differ from 100% by more than one percentage point. "
                        "This may reflect rounding or data-entry differences.",
                    )
                )

            if "Number of Students Tested" in df.columns:
                tested = pd.to_numeric(
                    df["Number of Students Tested"],
                    errors="coerce",
                )
                invalid_n = int(
                    (tested.isna() | (tested <= 0)).sum()
                )
                if invalid_n:
                    checks.append(
                        _check(
                            "warning",
                            "Student-count values should be reviewed",
                            f"{invalid_n:,} rows have a missing, zero, or negative tested count.",
                        )
                    )
                else:
                    checks.append(
                        _check(
                            "pass",
                            "Student counts look usable",
                            "Tested-count values are positive where provided.",
                        )
                    )

    if family_name == "survey or Likert-response data":
        likely_item_columns = []
        for column in df.columns:
            series = df[column]
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if (
                    not numeric.empty
                    and ((numeric % 1).abs() < 1e-9).all()
                    and 2 <= numeric.nunique() <= 7
                    and numeric.min() >= 0
                    and numeric.max() <= 7
                    and "id" not in str(column).lower()
                ):
                    likely_item_columns.append(str(column))
        if likely_item_columns:
            checks.append(
                _check(
                    "pass",
                    "Ordered response items detected",
                    f"{len(likely_item_columns)} bounded numeric survey item(s) are available.",
                )
            )

    if family_name == "student-level assessment data":
        student_col = next(
            (
                column
                for column in ["Student ID", "Student #", "Student"]
                if column in df.columns
            ),
            None,
        )
        if student_col:
            duplicate_ids = int(
                df[student_col]
                .dropna()
                .astype(str)
                .duplicated()
                .sum()
            )
            if duplicate_ids:
                checks.append(
                    _check(
                        "info",
                        "Repeated student identifiers detected",
                        f"{duplicate_ids:,} repeated identifiers were found. "
                        "This may be expected when students have multiple records.",
                    )
                )
            else:
                checks.append(
                    _check(
                        "pass",
                        "Student identifiers are unique",
                        f"{student_col} contains no repeated nonblank values.",
                    )
                )

    levels = [item["level"] for item in checks]

    if "error" in levels:
        status = "error"
        label = "Needs correction"
        message = "The dataset has issues that may prevent reliable analysis."
    elif "warning" in levels:
        status = "warning"
        label = "Ready with cautions"
        message = "VizCreate can continue, but one or more items should be reviewed."
    else:
        status = "ready"
        label = "Ready for analysis"
        message = "The dataset passed the core structural checks."

    return {
        "status": status,
        "label": label,
        "message": message,
        "checks": checks,
        "counts": {
            "pass": levels.count("pass"),
            "info": levels.count("info"),
            "warning": levels.count("warning"),
            "error": levels.count("error"),
        },
    }


def health_icon(level: str) -> str:
    """Return a compact icon for a health-check level."""
    return {
        "pass": "✓",
        "info": "ℹ",
        "warning": "⚠",
        "error": "✕",
    }.get(level, "•")
