"""General tabular fallback profile."""

from __future__ import annotations

import pandas as pd

from .base_profile import (
    DatasetProfile,
    DatasetProfileResult,
    detect_organizational_level,
    first_existing_column,
)


class GeneralTabularProfile(DatasetProfile):
    profile_id = "general_tabular"
    display_name = "General Tabular Data"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        numeric_columns = [
            str(column)
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_columns = [
            str(column)
            for column in df.columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        time_col = first_existing_column(
            df,
            ["School Year", "Year", "Date", "Month", "Quarter"],
        )
        entity_column, org_level = detect_organizational_level(df)

        roles = {}
        if time_col:
            roles["time"] = time_col
        if numeric_columns:
            roles["numeric_measures"] = ", ".join(numeric_columns[:5])
        if categorical_columns:
            roles["categorical_dimensions"] = ", ".join(categorical_columns[:5])
        if entity_column:
            roles["entity"] = entity_column

        recommended = ["Bar chart"]
        if time_col and numeric_columns:
            recommended.append("Line chart")
        if len(categorical_columns) >= 2 and numeric_columns:
            recommended.append("Heatmap")
        if numeric_columns and categorical_columns:
            recommended.append("Box plot when rows are independent observations")

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=0.20,
            description=(
                "A general cross-sectional or longitudinal table without a strong "
                "match to a specialized educational-data profile."
            ),
            structure="general tabular structure",
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=recommended,
            discouraged_charts=[
                "Charts requiring columns not present in the file",
            ],
            suggested_questions=[
                "Which numeric measure should be compared across categories?",
                "Is there a meaningful time field for trend analysis?",
                "Which groups have the highest or lowest values?",
                "What chart best matches the structure of this file?",
            ],
            cautions=[
                "Domain-specific assumptions are limited for an unrecognized dataset.",
                "Users should verify that rows represent comparable observations.",
            ],
            prompt_guidance=(
                "Use only available categorical, numeric, and time fields. Avoid "
                "domain-specific assumptions and do not invent unsupported analyses."
            ),
            default_spec={},
            evidence=[
                f"{len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns detected."
            ],
        )
