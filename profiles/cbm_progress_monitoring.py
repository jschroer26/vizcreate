"""CBM and progress-monitoring dataset profile."""

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


class CbmProgressMonitoringProfile(DatasetProfile):
    profile_id = "cbm_progress_monitoring"
    display_name = "CBM / Progress-Monitoring Data"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        score = 0.0
        evidence: list[str] = []

        student_col = first_existing_column(
            df,
            ["Student ID", "Student #", "Student", "Student Name"],
        )
        date_col = first_existing_column(
            df,
            ["Assessment Date", "Date", "Test Date", "Administration Date"],
        )
        window_col = first_existing_column(
            df,
            [
                "Screening Period",
                "Screening Window",
                "Season",
                "Benchmark Period",
                "Assessment Window",
                "Window",
            ],
        )
        measure_col = first_existing_column(
            df,
            ["Measure", "Assessment", "Test", "Probe", "Skill"],
        )
        grade_col = first_existing_column(
            df,
            ["Grade", "Grade Level", "Testing Grade"],
        )

        score_columns = matching_columns(
            df,
            [
                "score",
                "rate",
                "roi",
                "words correct",
                "wcpm",
                "fluency",
                "accuracy",
            ],
        )
        score_columns = [
            column
            for column in score_columns
            if pd.api.types.is_numeric_dtype(df[column])
        ]

        benchmark_columns = matching_columns(
            df,
            ["benchmark", "risk", "status", "tier"],
        )
        goal_columns = matching_columns(
            df,
            ["goal", "target", "aimline"],
        )
        growth_columns = matching_columns(
            df,
            ["rate of improvement", "growth rate", "roi", "slope"],
        )

        repeated_students = False
        repeated_observation_evidence = False

        if student_col:
            repeated_students = (
                df[student_col]
                .dropna()
                .astype(str)
                .duplicated()
                .any()
            )
            if repeated_students:
                score += 0.28
                evidence.append(
                    "Repeated student identifiers indicate multiple observations."
                )
            else:
                # A unique student ID by itself is not CBM evidence.
                score += 0.02

        if date_col:
            score += 0.20
            evidence.append(f"Assessment-date field detected: {date_col}.")
        if window_col:
            score += 0.18
            evidence.append(f"Screening-window field detected: {window_col}.")
        if measure_col:
            score += 0.10
            evidence.append(f"Measure field detected: {measure_col}.")
        if score_columns:
            score += min(0.18, 0.07 * len(score_columns))
            evidence.append(f"{len(score_columns)} CBM-like score field(s) detected.")
        if benchmark_columns:
            score += min(0.10, 0.04 * len(benchmark_columns))
            evidence.append("Benchmark or risk status field detected.")
        if goal_columns:
            score += min(0.10, 0.05 * len(goal_columns))
            evidence.append("Goal or target field detected.")
        if growth_columns:
            score += min(0.08, 0.04 * len(growth_columns))
            evidence.append("Growth-rate field detected.")
        if grade_col:
            score += 0.03

        repeated_observation_evidence = (
            repeated_students
            and bool(date_col or window_col)
            and bool(score_columns)
        )

        aggregated_cbm_evidence = (
            bool(window_col or date_col)
            and bool(score_columns)
            and bool(benchmark_columns)
            and not student_col
        )

        # Hard evidence rule: without repeated or aggregated monitoring structure,
        # cap CBM confidence so student-level assessment can win.
        if repeated_observation_evidence:
            score += 0.12
            evidence.append(
                "Repeated scores and ordered time support true progress monitoring."
            )
        elif aggregated_cbm_evidence:
            score += 0.10
            evidence.append(
                "Time-window and benchmark fields support an aggregated CBM summary."
            )
        else:
            score = min(score, 0.42)
            evidence.append(
                "Insufficient repeated-measure evidence for a strong CBM classification."
            )

        entity_column, org_level = detect_organizational_level(df)

        if repeated_observation_evidence:
            structure = "student-level repeated measures"
        elif aggregated_cbm_evidence:
            structure = "aggregated screening or benchmark summary"
        else:
            structure = "possible CBM-like assessment table"

        roles: dict[str, str] = {}
        if student_col:
            roles["student"] = student_col
        if date_col:
            roles["time"] = date_col
        elif window_col:
            roles["time"] = window_col
        if grade_col:
            roles["grade"] = grade_col
        if measure_col:
            roles["measure"] = measure_col
        if score_columns:
            roles["primary_measure"] = score_columns[0]
        if benchmark_columns:
            roles["benchmark_status"] = benchmark_columns[0]
        if goal_columns:
            roles["goal"] = goal_columns[0]
        if growth_columns:
            roles["growth_rate"] = growth_columns[0]
        if entity_column:
            roles["entity"] = entity_column

        questions: list[str] = []
        if repeated_observation_evidence:
            questions.append("Which students are improving over time?")
            if goal_columns:
                questions.append(
                    "Which students are on track to meet the next goal?"
                )
            if benchmark_columns:
                questions.append(
                    "Which students remain below benchmark across multiple observations?"
                )
            if grade_col:
                questions.append(
                    f"How do progress trends differ by {grade_col}?"
                )
        elif aggregated_cbm_evidence:
            questions.extend(
                [
                    "How did benchmark status change across screening windows?",
                    "Which grades or schools have the highest percentage at risk?",
                ]
            )
        else:
            questions.extend(
                [
                    "Does this file contain repeated assessments for each student?",
                    "Which score or benchmark field should be summarized?",
                ]
            )

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=clamp_score(score),
            description=(
                "Curriculum-based measurement, screening, or progress-monitoring "
                "data with repeated observations, ordered windows, benchmark status, "
                "goals, or growth rates."
            ),
            structure=structure,
            organizational_level=org_level,
            detected_roles=roles,
            recommended_charts=[
                "Progress-monitoring line chart",
                "Student-versus-goal line chart when a goal exists",
                "Benchmark-status stacked bar",
                "Screening-window heatmap",
            ],
            discouraged_charts=[
                "Box plot as evidence of individual progress",
                "Combining unlike CBM measures on one scale",
                "Goal-attainment claims without a goal field",
            ],
            suggested_questions=questions[:4],
            cautions=[
                "Screening scores, progress-monitoring scores, percentiles, benchmark categories, and goals are not interchangeable.",
                "Fall, Winter, and Spring should be treated as ordered windows.",
                "Different CBM measures may use different score scales.",
            ],
            prompt_guidance=(
                "Use CBM reasoning only when repeated observations or an aggregated "
                "screening-window structure is present. Preserve distinctions among raw "
                "scores, benchmark status, goals, and rates of improvement. Never describe "
                "a cross-sectional box plot as progress toward a goal."
            ),
            default_spec={},
            evidence=evidence,
        )
