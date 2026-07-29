"""Rank and select VizCreate dataset profiles."""

from __future__ import annotations

import pandas as pd

from profiles import (
    ActLongitudinalProfile,
    CbmProgressMonitoringProfile,
    DatasetProfileResult,
    GeneralTabularProfile,
    LikertSurveyProfile,
    StudentAssessmentProfile,
    WytoppCurrentYearProfile,
    WytoppLongitudinalProfile,
)


PROFILE_CLASSES = [
    ActLongitudinalProfile,
    WytoppLongitudinalProfile,
    WytoppCurrentYearProfile,
    CbmProgressMonitoringProfile,
    StudentAssessmentProfile,
    LikertSurveyProfile,
    GeneralTabularProfile,
]


def rank_dataset_profiles(
    df: pd.DataFrame,
) -> list[DatasetProfileResult]:
    """Evaluate and rank all profiles from strongest to weakest."""
    results = [
        profile_class().evaluate(df)
        for profile_class in PROFILE_CLASSES
    ]
    return sorted(
        results,
        key=lambda result: result.confidence,
        reverse=True,
    )


def detect_dataset_profile(
    df: pd.DataFrame,
) -> DatasetProfileResult:
    """Return the strongest dataset-profile match."""
    return rank_dataset_profiles(df)[0]


def legacy_family_name(profile: DatasetProfileResult) -> str:
    """Map new profiles to names expected by the current app and health checks."""
    mapping = {
        "act_longitudinal": "ACT longitudinal summary",
        "wytopp_longitudinal": "WYTOPP longitudinal proficiency summary",
        "wytopp_current_year": "WYTOPP current-year proficiency summary",
        "cbm_progress_monitoring": "CBM / progress-monitoring data",
        "student_assessment": "student-level assessment data",
        "likert_survey": "survey or Likert-response data",
        "general_tabular": "general tabular data",
    }
    return mapping.get(profile.profile_id, profile.display_name)


def profile_guidance(profile: DatasetProfileResult) -> str:
    """Return concise guidance compatible with the current prompt builder."""
    return profile.prompt_guidance or profile.description
