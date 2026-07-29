"""WYTOPP current-year summary profile."""

from __future__ import annotations

import pandas as pd

from .base_profile import (
    DatasetProfile,
    DatasetProfileResult,
    clamp_score,
    detect_organizational_level,
)


class WytoppCurrentYearProfile(DatasetProfile):
    profile_id = "wytopp_current_year"
    display_name = "WYTOPP Current-Year Assessment"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        columns = set(df.columns.astype(str))
        required = {
            "Grade",
            "Subject",
            "Percent Basic and Below",
            "Percent Proficient and Advanced",
        }
        matched = len(required.intersection(columns))
        year_count = (
            df["School Year"].dropna().astype(str).nunique()
            if "School Year" in df.columns
            else 0
        )

        score = 0.0
        evidence: list[str] = []

        if matched:
            score += 0.14 * matched
            evidence.append(f"{matched} of 4 current-year WYTOPP fields detected.")
        if required.issubset(columns):
            score += 0.22
            evidence.append("All core current-year proficiency columns are present.")
        if year_count == 1:
            score += 0.18
            evidence.append("Exactly one school year is present.")
        elif "School Year" not in columns:
            score += 0.08
            evidence.append("No time field suggests a cross-sectional summary.")
        elif year_count >= 2:
            score -= 0.20
        if "Number of Students Tested" in columns:
            score += 0.05
        if "Subgroup" in columns:
            score += 0.03

        entity_column, org_level = detect_organizational_level(df)

        roles = {
            "grade": "Grade" if "Grade" in columns else "",
            "subject": "Subject" if "Subject" in columns else "",
            "primary_measure": (
                "Percent Proficient and Advanced"
                if "Percent Proficient and Advanced" in columns
                else ""
            ),
            "complementary_measure": (
                "Percent Basic and Below"
                if "Percent Basic and Below" in columns
                else ""
            ),
        }
        if "School Year" in columns:
            roles["time"] = "School Year"
        if "Number of Students Tested" in columns:
            roles["sample_size"] = "Number of Students Tested"
        if entity_column:
            roles["entity"] = entity_column
        roles = {key: value for key, value in roles.items() if value}

        default_subject = None
        if "Subject" in df.columns:
            values = df["Subject"].dropna().astype(str).unique().tolist()
            for candidate in ["ELA", "English", "English Language Arts", "Reading"]:
                if candidate in values:
                    default_subject = candidate
                    break

        default_filters = {}
        if default_subject:
            default_filters["Subject"] = default_subject
        if "Subgroup" in df.columns:
            subgroup_values = df["Subgroup"].dropna().astype(str).unique().tolist()
            if "All Students" in subgroup_values:
                default_filters["Subgroup"] = "All Students"

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=clamp_score(score),
            description=(
                "Aggregated current-year WYTOPP proficiency data for grade, "
                "subject, school, district, or state comparisons."
            ),
            structure="aggregated current-year summary",
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=[
                "WYTOPP stacked bar",
                "Grouped bar chart",
                "Heatmap",
            ],
            discouraged_charts=[
                "Growth line chart",
                "Box plot",
                "Histogram",
            ],
            suggested_questions=[
                "Which grades have the lowest Math proficiency?",
                "How does the school compare with district or state results?",
                "Which subject should receive the most attention this year?",
                "Where are current areas of relative strength?",
            ],
            cautions=[
                "One year supports current-status comparisons, not growth claims.",
                "Aggregated percentages do not show individual student distributions.",
            ],
            prompt_guidance=(
                "Treat this as current-year aggregated WYTOPP data. Prefer grade, "
                "subject, and organizational comparisons. Do not claim growth or "
                "create student-level distribution charts."
            ),
            default_spec={
                "chart_type": "bar",
                "special_mode": None,
                "x": "Grade",
                "y": "Percent Proficient and Advanced",
                "group": None,
                "row": None,
                "col": None,
                "filters": default_filters,
                "aggregation": "mean",
                "sort_x": "ascending",
                "facets": None,
                "notes": "Default current-year WYTOPP grade comparison.",
            },
            evidence=evidence,
        )
