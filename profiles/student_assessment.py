"""Student-level assessment dataset profile."""

from __future__ import annotations

import pandas as pd

from .base_profile import (
    DatasetProfile,
    DatasetProfileResult,
    clamp_score,
    detect_organizational_level,
    first_existing_column,
    matching_columns,
)


class StudentAssessmentProfile(DatasetProfile):
    profile_id = "student_assessment"
    display_name = "Student-Level Assessment Data"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        score = 0.0
        evidence: list[str] = []

        student_col = first_existing_column(
            df,
            [
                "Student ID",
                "Student #",
                "Student",
                "Student Name",
                "Learner ID",
            ],
        )
        grade_col = first_existing_column(
            df,
            ["Grade", "Grade Level", "Testing Grade"],
        )
        subgroup_col = first_existing_column(
            df,
            [
                "Gender",
                "Race/Ethnicity",
                "Ethnicity",
                "Program",
                "Program Track",
                "Class Standing",
                "Teacher",
                "Classroom",
            ],
        )
        time_col = first_existing_column(
            df,
            [
                "Assessment Date",
                "Date",
                "School Year",
                "Screening Period",
                "Screening Window",
                "Season",
            ],
        )

        numeric_columns = [
            str(column)
            for column in df.columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]
        score_columns = matching_columns(
            df,
            [
                "score",
                "scale",
                "subscore",
                "percentile",
                "achievement",
                "reading",
                "math",
                "ela",
                "science",
            ],
        )
        score_columns = [
            column
            for column in score_columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        repeated_students = False
        if student_col:
            score += 0.28
            evidence.append(f"Student identifier detected: {student_col}.")
            repeated_students = (
                df[student_col]
                .dropna()
                .astype(str)
                .duplicated()
                .any()
            )

        if grade_col:
            score += 0.14
            evidence.append(f"Grade field detected: {grade_col}.")

        if score_columns:
            score += min(0.30, 0.10 * len(score_columns))
            evidence.append(
                f"{len(score_columns)} assessment-score field(s) detected."
            )
        elif numeric_columns:
            score += min(0.18, 0.05 * len(numeric_columns))
            evidence.append(
                f"{len(numeric_columns)} numeric measure(s) detected."
            )

        if subgroup_col:
            score += 0.07
            evidence.append(f"Grouping field detected: {subgroup_col}.")

        # Single-record student assessment is strengthened when no repeated-measure
        # structure is present.
        if student_col and not repeated_students:
            score += 0.14
            evidence.append(
                "Student identifiers are unique, consistent with one row per student."
            )

        # Repeated students plus a real time field are better handled by CBM.
        if repeated_students and time_col:
            score -= 0.22
            evidence.append(
                "Repeated student records with time reduce the student-assessment match."
            )

        entity_column, org_level = detect_organizational_level(df)

        roles: dict[str, str] = {}
        if student_col:
            roles["student"] = student_col
        if grade_col:
            roles["grade"] = grade_col
        if score_columns:
            roles["assessment_measures"] = ", ".join(score_columns[:6])
        elif numeric_columns:
            roles["numeric_measures"] = ", ".join(numeric_columns[:6])
        if subgroup_col:
            roles["group"] = subgroup_col
        if entity_column:
            roles["entity"] = entity_column

        questions = []
        if grade_col and score_columns:
            questions.append(
                f"How do {score_columns[0]} distributions differ by {grade_col}?"
            )
            questions.append(
                f"Which {grade_col} has the highest median {score_columns[0]}?"
            )
        if subgroup_col and score_columns:
            questions.append(
                f"How does {score_columns[0]} differ by {subgroup_col}?"
            )
        if len(score_columns) >= 2:
            questions.append(
                f"How are {score_columns[0]} and {score_columns[1]} related?"
            )
        if not questions:
            questions = [
                "Which groups have the highest and lowest scores?",
                "How variable are student scores across categories?",
            ]

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=clamp_score(score),
            description=(
                "Individual student assessment records with one or more score "
                "columns, often accompanied by grade and subgroup fields."
            ),
            structure="one row per student or assessment record",
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=[
                "Box plot",
                "Grouped bar chart",
                "Heatmap",
                "Scatterplot when two numeric measures exist",
                "Histogram when distribution support is added",
            ],
            discouraged_charts=[
                "Progress-toward-goal analysis without repeated measurements",
                "Growth claims without a time field",
            ],
            suggested_questions=questions[:4],
            cautions=[
                "A single score per student supports distribution and comparison analyses, not progress monitoring.",
                "Scale scores from different assessments may not be directly comparable.",
                "Small subgroups should be interpreted cautiously.",
            ],
            prompt_guidance=(
                "Treat this as student-level cross-sectional assessment data. "
                "Use distributions, grade or subgroup comparisons, and relationships "
                "between numeric scores. Do not claim progress or goal attainment unless "
                "the dataset contains repeated measurements and an ordered time field."
            ),
            default_spec={},
            evidence=evidence,
        )
