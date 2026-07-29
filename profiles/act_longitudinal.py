"""ACT longitudinal summary profile for official Wyoming exports."""

from __future__ import annotations

import re

import pandas as pd

from .base_profile import DatasetProfile, DatasetProfileResult, clamp_score, detect_organizational_level


ACT_SUBJECTS = ("English", "Math", "Reading", "Science", "Composite")


class ActLongitudinalProfile(DatasetProfile):
    profile_id = "act_longitudinal"
    display_name = "ACT Longitudinal Summary"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        columns = [str(column) for column in df.columns]
        column_set = set(columns)
        average_columns = [
            column for column in columns
            if re.search(r"\bScore Average$", column, flags=re.IGNORECASE)
        ]
        tested_columns = [
            column for column in columns
            if re.search(r"\bNumber Tested$", column, flags=re.IGNORECASE)
        ]
        std_columns = [
            column for column in columns
            if re.search(r"\bScore Std\.? Dev\.?$", column, flags=re.IGNORECASE)
        ]
        subjects = [
            subject for subject in ACT_SUBJECTS
            if any(column.lower().startswith(subject.lower() + " ") for column in columns)
        ]
        year_count = (
            df["School Year"].dropna().astype(str).nunique()
            if "School Year" in column_set else 0
        )
        act_value = False
        if "Test Type" in column_set:
            act_value = df["Test Type"].dropna().astype(str).str.upper().eq("ACT").any()

        score = 0.0
        evidence: list[str] = []
        if "School Year" in column_set:
            score += 0.10
            evidence.append("School Year is available.")
        if act_value:
            score += 0.28
            evidence.append("Test Type identifies ACT records.")
        if average_columns:
            score += min(0.25, 0.05 * len(average_columns))
            evidence.append(f"{len(average_columns)} ACT average-score measures detected.")
        if tested_columns:
            score += min(0.15, 0.03 * len(tested_columns))
            evidence.append(f"{len(tested_columns)} tested-count measures detected.")
        if std_columns:
            score += min(0.08, 0.016 * len(std_columns))
            evidence.append(f"{len(std_columns)} standard-deviation measures detected.")
        if len(subjects) >= 3:
            score += 0.08
            evidence.append("Multiple ACT subject domains are present.")
        if year_count >= 2:
            score += 0.12
            evidence.append(f"{year_count} school years support longitudinal analysis.")

        entity_column, org_level = detect_organizational_level(df)
        primary = "Composite Score Average" if "Composite Score Average" in column_set else (average_columns[0] if average_columns else "")
        roles = {
            "time": "School Year" if "School Year" in column_set else "",
            "test_type": "Test Type" if "Test Type" in column_set else "",
            "grade": "Testing Grade" if "Testing Grade" in column_set else "",
            "primary_measure": primary,
            "measure_family": "ACT Score Average" if average_columns else "",
        }
        if entity_column:
            roles["entity"] = entity_column
        roles = {key: value for key, value in roles.items() if value}

        cautions = [
            "Rows contain aggregated ACT summaries rather than student-level scores.",
            "Standard deviations describe score variability but do not identify individual students.",
            "Comparisons should retain the ACT scale and should not be converted to proficiency percentages.",
        ]

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=clamp_score(score),
            description=(
                "Aggregated ACT average scores, tested counts, and standard deviations "
                "organized across school years and ACT subject domains."
            ),
            structure="aggregated longitudinal assessment summary",
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=[
                "Line chart",
                "Grouped line chart",
                "Grouped bar chart",
                "Error-bar chart",
            ],
            discouraged_charts=[
                "WYTOPP stacked proficiency bar",
                "Likert chart",
                "Student-level histogram",
            ],
            suggested_questions=[
                "How has the ACT composite average changed over time?",
                "Compare English, Math, Reading, and Science average scores.",
                "Which ACT subject improved the most across the available years?",
                "How has the number of students tested changed over time?",
            ],
            cautions=cautions,
            prompt_guidance=(
                "Treat this as aggregated longitudinal ACT summary data. Prefer trends "
                "in Score Average measures, subject comparisons, tested-count trends, "
                "and optional standard-deviation displays. Do not use WYTOPP proficiency "
                "stacking or imply that the rows are individual student records."
            ),
            default_spec={
                "chart_type": "line",
                "special_mode": None,
                "x": "School Year" if "School Year" in column_set else None,
                "y": primary or None,
                "group": None,
                "row": None,
                "col": None,
                "filters": {"Test Type": "ACT"} if act_value else {},
                "aggregation": "mean",
                "sort_x": "ascending",
                "facets": None,
                "notes": "Default ACT composite trend view.",
            },
            evidence=evidence,
        )
