"""WYTOPP longitudinal summary profile."""

from __future__ import annotations

import pandas as pd

from .base_profile import (
    DatasetProfile,
    DatasetProfileResult,
    clamp_score,
    detect_organizational_level,
    first_existing_column,
)


class WytoppLongitudinalProfile(DatasetProfile):
    profile_id = "wytopp_longitudinal"
    display_name = "WYTOPP Longitudinal Assessment"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        columns = set(df.columns.astype(str))
        required = {
            "School Year",
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
            score += 0.12 * matched
            evidence.append(f"{matched} of 5 core WYTOPP columns detected.")
        if required.issubset(columns):
            score += 0.22
            evidence.append("All core WYTOPP proficiency-summary columns are present.")
        if year_count >= 2:
            score += 0.18
            evidence.append(f"{year_count} school years support longitudinal analysis.")
        if "Number of Students Tested" in columns:
            score += 0.05
            evidence.append("A tested-count field is available.")
        if "Subgroup" in columns:
            score += 0.03
            evidence.append("A subgroup field is available.")

        entity_column, org_level = detect_organizational_level(df)

        roles = {
            "time": "School Year" if "School Year" in columns else "",
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
                "Aggregated Grade × School Year × Subject proficiency data "
                "designed for multi-year achievement analysis."
            ),
            structure="aggregated longitudinal summary",
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=[
                "Line chart",
                "Heatmap",
                "WYTOPP stacked bar",
                "Grouped bar chart",
            ],
            discouraged_charts=[
                "Box plot",
                "Histogram",
                "Student-level scatterplot",
            ],
            suggested_questions=[
                "How has ELA proficiency changed over time by grade?",
                "Which subject has shown the greatest growth?",
                "Where are the strongest and weakest grade-year patterns?",
                "How does the school compare with district or state results?",
            ],
            cautions=[
                "Rows are aggregated and do not represent individual student distributions.",
                "Science may include different tested grades than ELA or Math.",
                "Growth comparisons should use at least two school years.",
            ],
            prompt_guidance=(
                "Treat this as aggregated longitudinal WYTOPP data. Prefer trends, "
                "grade-year heatmaps, grouped comparisons, and the special WYTOPP "
                "stacked mode. Do not create box plots or imply student-level distributions."
            ),
            default_spec={
                "chart_type": "stacked_bar",
                "special_mode": "wytopp_stacked",
                "x": "School Year",
                "y": None,
                "group": None,
                "row": None,
                "col": None,
                "filters": default_filters,
                "aggregation": "mean",
                "sort_x": "ascending",
                "facets": None,
                "notes": "Default longitudinal WYTOPP status view.",
            },
            evidence=evidence,
        )
