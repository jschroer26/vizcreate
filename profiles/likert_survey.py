"""Text-coded and numeric-coded Likert survey profile."""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .base_profile import (
    DatasetProfile,
    DatasetProfileResult,
    clamp_score,
    first_existing_column,
)


LIKERT_LABELS = {
    "strongly disagree",
    "disagree",
    "neutral",
    "neither agree nor disagree",
    "agree",
    "strongly agree",
    "very dissatisfied",
    "dissatisfied",
    "satisfied",
    "very satisfied",
    "never",
    "rarely",
    "sometimes",
    "often",
    "always",
}


def _looks_like_item_name(name: str) -> bool:
    text = str(name).strip()
    lower = text.lower()

    patterns = [
        r"^q\d+([_\-\s].*)?$",
        r"^item[_\-\s]?\d+",
        r"^question[_\-\s]?\d+",
        r"^\d+[\.\)]\s*",
    ]
    if any(re.search(pattern, lower) for pattern in patterns):
        return True

    survey_terms = [
        "agree",
        "satisfaction",
        "confidence",
        "self efficacy",
        "self-efficacy",
        "engagement",
        "utility",
        "ease of use",
        "cognitive load",
        "support",
        "climate",
        "belonging",
        "perception",
    ]
    return any(term in lower for term in survey_terms)


def _humanize_construct(column_name: str) -> str:
    text = str(column_name).strip()
    text = re.sub(r"^(q|item|question)\d+[_\-\s]*", "", text, flags=re.I)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else str(column_name)


def _numeric_likert_info(series: pd.Series) -> dict | None:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None

    # Numeric Likert codes are normally small integers.
    if not ((numeric % 1).abs() < 1e-9).all():
        return None

    values = sorted({int(value) for value in numeric.unique()})
    if len(values) < 2 or len(values) > 7:
        return None

    observed_min = min(values)
    observed_max = max(values)

    likely_scale = None
    if 0 <= observed_min and observed_max <= 3:
        likely_scale = "likely 0–3 or 1–4 ordered scale"
    elif 1 <= observed_min and observed_max <= 4:
        likely_scale = "likely 1–4 ordered scale"
    elif 1 <= observed_min and observed_max <= 5:
        likely_scale = "likely 1–5 Likert scale"
    elif 1 <= observed_min and observed_max <= 6:
        likely_scale = "likely 1–6 Likert scale"
    elif 1 <= observed_min and observed_max <= 7:
        likely_scale = "likely 1–7 Likert scale"
    elif 0 <= observed_min and observed_max <= 6:
        likely_scale = "likely 0–6 ordered scale"

    if likely_scale is None:
        return None

    return {
        "observed_values": values,
        "observed_range": f"{observed_min}–{observed_max}",
        "likely_scale": likely_scale,
    }


class LikertSurveyProfile(DatasetProfile):
    profile_id = "likert_survey"
    display_name = "Likert / Survey Response Data"

    def evaluate(self, df: pd.DataFrame) -> DatasetProfileResult:
        score = 0.0
        evidence: list[str] = []

        response_col = first_existing_column(
            df,
            ["Response", "Rating", "Answer", "Likert Response"],
        )
        question_col = first_existing_column(
            df,
            ["Question", "Item", "Survey Item", "Statement"],
        )
        respondent_col = first_existing_column(
            df,
            [
                "Respondent ID",
                "Respondent_ID",
                "Response ID",
                "Participant ID",
                "Respondent",
            ],
        )

        group_candidates = [
            column
            for column in [
                "Role",
                "School",
                "Department",
                "Position",
                "Grade Band",
                "Program Track",
                "Program_Track",
                "Class Standing",
                "Class_Standing",
            ]
            if column in df.columns
        ]

        text_likert_columns: list[str] = []
        numeric_likert_columns: list[str] = []
        numeric_scale_details: dict[str, dict] = {}
        item_name_columns: list[str] = []

        for column in df.columns:
            if _looks_like_item_name(str(column)):
                item_name_columns.append(str(column))

            if pd.api.types.is_numeric_dtype(df[column]):
                info = _numeric_likert_info(df[column])
                if info is not None:
                    # Avoid treating obvious identifiers as Likert items.
                    lower = str(column).lower()
                    if not any(
                        token in lower
                        for token in ["id", "year", "grade", "age", "count", "number"]
                    ):
                        numeric_likert_columns.append(str(column))
                        numeric_scale_details[str(column)] = info
            else:
                values = {
                    str(value).strip().lower()
                    for value in df[column].dropna().unique()[:100]
                }
                if values.intersection(LIKERT_LABELS):
                    text_likert_columns.append(str(column))

        # Strong text-coded evidence.
        if text_likert_columns:
            score += min(0.48, 0.12 * len(text_likert_columns))
            evidence.append(
                f"Likert response labels detected in {len(text_likert_columns)} column(s)."
            )

        # Wide numeric survey evidence requires multiple bounded item columns.
        if len(numeric_likert_columns) >= 3:
            score += min(0.50, 0.11 * len(numeric_likert_columns))
            evidence.append(
                f"{len(numeric_likert_columns)} bounded integer response item(s) detected."
            )
        elif len(numeric_likert_columns) == 2:
            score += 0.18

        if len(item_name_columns) >= 3:
            score += 0.18
            evidence.append(
                f"{len(item_name_columns)} survey-style item names detected."
            )
        elif item_name_columns:
            score += 0.06

        if respondent_col:
            score += 0.08
            evidence.append(f"Respondent identifier detected: {respondent_col}.")
        if response_col:
            score += 0.12
            evidence.append(f"Long-format response field detected: {response_col}.")
        if question_col:
            score += 0.10
            evidence.append(f"Survey item field detected: {question_col}.")
        if group_candidates:
            score += min(0.10, 0.04 * len(group_candidates))
            evidence.append(
                f"{len(group_candidates)} respondent grouping field(s) detected."
            )

        survey_items = list(
            dict.fromkeys(
                text_likert_columns
                + numeric_likert_columns
                + [
                    column
                    for column in item_name_columns
                    if column not in text_likert_columns
                    and column not in numeric_likert_columns
                ]
            )
        )

        if response_col and question_col:
            structure = "long-format Likert survey"
        elif numeric_likert_columns:
            structure = "wide-format numerically coded Likert survey"
        elif text_likert_columns:
            structure = "wide-format text-coded Likert survey"
        else:
            structure = "possible survey response table"

        constructs = [_humanize_construct(column) for column in survey_items[:10]]

        scale_counter = Counter(
            detail["likely_scale"]
            for detail in numeric_scale_details.values()
        )
        likely_scale = scale_counter.most_common(1)[0][0] if scale_counter else None

        observed_values = sorted(
            {
                value
                for detail in numeric_scale_details.values()
                for value in detail["observed_values"]
            }
        )

        roles: dict[str, str] = {}
        if respondent_col:
            roles["respondent"] = respondent_col
        if question_col:
            roles["question"] = question_col
        if response_col:
            roles["response"] = response_col
        if survey_items:
            roles["survey_items"] = ", ".join(survey_items[:8])
        if constructs:
            roles["constructs"] = ", ".join(constructs[:8])
        if group_candidates:
            roles["grouping_dimensions"] = ", ".join(group_candidates)
        if likely_scale:
            roles["response_scale"] = (
                f"{likely_scale}; observed values "
                + ", ".join(str(value) for value in observed_values)
            )

        questions: list[str] = []
        if constructs:
            questions.append(
                "Which survey construct received the strongest ratings?"
            )
            questions.append(
                "Which construct shows the greatest variation or disagreement?"
            )
        if group_candidates and constructs:
            questions.append(
                f"How does {constructs[0]} differ by {group_candidates[0]}?"
            )
        if len(constructs) >= 2:
            questions.append(
                f"How are {constructs[0]} and {constructs[1]} related?"
            )
        if not questions:
            questions = [
                "Which items received the strongest agreement?",
                "Which items show the most divided opinions?",
            ]

        cautions = [
            "Likert responses are ordinal and should not automatically be treated as interval-level continuous data.",
            "Means may be useful descriptively, but response distributions should remain visible.",
            "Missing response categories do not prove those options were absent from the original scale.",
        ]

        return DatasetProfileResult(
            profile_id=self.profile_id,
            display_name=self.display_name,
            confidence=clamp_score(score),
            description=(
                "Survey-response data containing text-coded or numerically coded "
                "ordered response items, respondent identifiers, and grouping fields."
            ),
            structure=structure,
            organizational_level=None,
            detected_roles=roles,
            recommended_charts=[
                "Diverging stacked Likert bar",
                "Stacked response bar",
                "Agreement or favorable-response bar",
                "Question-by-group heatmap",
                "Construct comparison",
            ],
            discouraged_charts=[
                "Line chart without a true time field",
                "Box plot of text response categories",
                "Treating coded responses as unquestionably continuous",
            ],
            suggested_questions=questions[:4],
            cautions=cautions,
            prompt_guidance=(
                "Treat the detected item columns as ordered survey responses. "
                "Prefer response distributions, stacked or diverging bars, favorable "
                "response summaries, construct comparisons, and group comparisons. "
                "Do not assume equal intervals merely because responses are coded numerically."
            ),
            default_spec={},
            evidence=evidence,
        )
