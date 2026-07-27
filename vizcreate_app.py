import io
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import json
import re
from openai import OpenAI

# -------------------------
# Color palettes
# -------------------------
PALETTES = {
    "Default": None,  # use matplotlib defaults
    "Greyscale": ["#000000", "#555555", "#888888", "#BBBBBB", "#DDDDDD"],
    "UW Brown & Gold": ["#3B2314", "#FFC72C", "#6B4C3B", "#FFB81C"],
    "Blue-Orange": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "Colorblind-friendly": ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"],
    "Muted": ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
}


# -------------------------
# OpenAI client
# -------------------------
client = OpenAI()  # uses OPENAI_API_KEY from environment


# -------------------------
# LLM helper functions
# -------------------------
def summarize_dataframe_for_prompt(df: pd.DataFrame, max_examples: int = 3) -> str:
    """
    Build a human-readable schema description for the LLM:
    column names, inferred types, and a few example values.
    """
    lines = []
    lines.append("Columns:")

    for col in df.columns:
        series = df[col]
        # crude type mapping
        if pd.api.types.is_numeric_dtype(series):
            col_type = "numeric"
        else:
            col_type = "categorical"

        # get up to max_examples unique non-null values
        non_null = series.dropna().unique()
        examples = list(non_null[:max_examples])
        examples_str = ", ".join(map(lambda x: str(x), examples))

        lines.append(f"- {col} (type: {col_type}, example values: [{examples_str}])")

    return "\n".join(lines)


def detect_dataset_family(df: pd.DataFrame) -> tuple[str, str]:
    """Classify the uploaded dataset using its column structure."""
    columns = set(df.columns.astype(str))

    wytopp_required = {
        "School Year",
        "Grade",
        "Subject",
        "Percent Basic and Below",
        "Percent Proficient and Advanced",
    }
    if wytopp_required.issubset(columns):
        return (
            "WYTOPP longitudinal proficiency summary",
            "Aggregated Grade × School Year × Subject data. Prefer line charts, heatmaps, "
            "or the special WYTOPP stacked-bar mode. Do not recommend box plots because "
            "the rows are aggregated rather than individual student observations.",
        )

    student_markers = {"Student #", "Student ID", "Student"}
    grade_markers = {"Grade", "Grade Level", "Testing Grade"}
    has_student = bool(columns.intersection(student_markers))
    has_grade = bool(columns.intersection(grade_markers))
    numeric_count = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns)

    if has_student and has_grade and numeric_count >= 1:
        return (
            "student-level assessment data",
            "One row appears to represent one student. Bar charts, heatmaps, line charts, "
            "and box plots may be appropriate depending on the request.",
        )

    likert_terms = {"Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"}
    object_values = set()
    for col in df.select_dtypes(exclude=["number"]).columns[:10]:
        object_values.update(df[col].dropna().astype(str).unique()[:20])
    if len(likert_terms.intersection(object_values)) >= 3:
        return (
            "survey or Likert-response data",
            "Prefer bar charts or stacked bars for response categories and avoid treating "
            "ordinal responses as continuous unless the user explicitly requests it.",
        )

    if any(name in columns for name in ["School Year", "Year", "Date"]):
        return (
            "longitudinal summary data",
            "The file includes a time field. Prefer line charts for trends and bar charts "
            "for discrete comparisons.",
        )

    return (
        "general tabular data",
        "Choose a chart supported by the available categorical and numeric columns. "
        "Do not invent columns or request unsupported distributions.",
    )


def build_vizcreate_prompt(df: pd.DataFrame, user_prompt: str) -> str:
    """Construct the dataset-aware prompt sent to the visualization planner."""
    schema_text = summarize_dataframe_for_prompt(df)
    family_name, family_guidance = detect_dataset_family(df)

    system_instructions = """
You are VizCreate, a data visualization planning assistant.

Your job is to inspect a tabular dataset and convert the user's request into one valid chart specification.

Supported chart types:
- "bar": a single or grouped bar chart using one numeric y column
- "stacked_bar": a stacked bar chart
- "line": an ordered trend chart, usually over time
- "heatmap": a grid where row and col are categorical columns and y is numeric
- "box": a box-and-whisker plot; use only for student-level or observation-level data

Rules:
- Use only exact column names from the dataset.
- Never invent a column.
- Respect the detected dataset family and avoid charts that the data cannot support.
- For a heatmap, always populate row, col, and y explicitly.
- For time trends, normally use School Year or Year as x.
- When the dataset contains separate columns named Percent Basic and Below and
  Percent Proficient and Advanced, and the user asks to compare or stack those measures,
  use chart_type "stacked_bar" and special_mode "wytopp_stacked".
- In WYTOPP stacked mode, set x to Grade or School Year, based on the user's wording.
  Set y to null and group to null. The plotting engine will stack the two known percentage columns.
- Return JSON only, with no markdown or explanation outside the JSON.
"""

    json_schema = """
Return exactly one JSON object with these fields:
{
  "chart_type": "bar | stacked_bar | line | heatmap | box",
  "special_mode": "wytopp_stacked | null",
  "x": "exact x-axis column name, or null",
  "y": "exact primary numeric column name, or null",
  "group": "exact grouping/series column name, or null",
  "row": "heatmap row column, or null",
  "col": "heatmap column column, or null",
  "filters": {"ColumnName": "exact value or array of values"},
  "aggregation": "mean | sum | count | none",
  "sort_x": "none | ascending | descending",
  "facets": null,
  "notes": "brief interpretation"
}

Heatmap example:
{
  "chart_type": "heatmap",
  "special_mode": null,
  "x": null,
  "y": "Percent Proficient and Advanced",
  "group": null,
  "row": "Grade",
  "col": "School Year",
  "filters": {"Subject": "Math"},
  "aggregation": "mean",
  "sort_x": "ascending",
  "facets": null,
  "notes": "Math proficiency by grade and school year."
}

WYTOPP stacked-bar example:
{
  "chart_type": "stacked_bar",
  "special_mode": "wytopp_stacked",
  "x": "Grade",
  "y": null,
  "group": null,
  "row": null,
  "col": null,
  "filters": {"Subject": "Math"},
  "aggregation": "mean",
  "sort_x": "ascending",
  "facets": null,
  "notes": "Stack Basic and Below with Proficient and Advanced for each grade."
}
"""

    return f"""
{system_instructions}

DETECTED DATASET FAMILY
-----------------------
Family: {family_name}
Guidance: {family_guidance}

DATASET SCHEMA
--------------
{schema_text}

USER REQUEST
------------
{user_prompt}

OUTPUT FORMAT
-------------
{json_schema}
""".strip()


def get_chart_spec_from_llm(df: pd.DataFrame, user_prompt: str) -> dict:
    """
    Call the OpenAI API with our prompt and return a parsed chart spec dict.

    If anything goes wrong, show the error in Streamlit and return a simple fallback spec.
    """
    prompt_text = build_vizcreate_prompt(df, user_prompt)

    try:
        # Call the Responses API
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt_text,
        )

        # Extract text from the response
        try:
            blocks = response.output[0].content
            pieces = []
            for block in blocks:
                if hasattr(block, "text") and block.text is not None:
                    pieces.append(block.text)
            raw_text = "".join(pieces).strip()
        except Exception as e:
            raw_text = str(response)
            st.warning(f"Could not extract text cleanly from response: {e}")

        # Try to parse JSON
        try:
            spec = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                spec = json.loads(match.group(0))
            else:
                raise

        return spec

    except Exception as e:
        if "insufficient_quota" in str(e) or "quota" in str(e):
           st.warning("VizCreate AI is temporarily unavailable due to API usage limits. "
        "You can still create charts using the manual configuration below."
           )
        else:
            st.error(f"VizCreate agent error: {e}")


        # Fallback so the rest of the app keeps working
        return {
            "chart_type": "bar",
            "special_mode": None,
            "x": None,
            "y": None,
            "group": None,
            "row": None,
            "col": None,
            "filters": {},
            "aggregation": "mean",
            "sort_x": "none",
            "facets": None,
            "notes": "Fallback spec returned due to an error in the agent.",
        }

def apply_spec(
    df: pd.DataFrame,
    spec: dict,
    ui_filters: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Apply filters from the UI (quick filters) and from the spec.
    - ui_filters: dict like {"Grade": [3,4,5], "School Year": ["2020-21", "2021-22"]}
      These ALWAYS take precedence over spec filters for the same column.
    - spec["filters"]: used for remaining columns not controlled by the UI.
    """
    filtered = df.copy()

    # 1) Apply UI filters first
    if ui_filters:
        for col, wanted in ui_filters.items():
            if col not in filtered.columns:
                continue
            if isinstance(wanted, list):
                filtered = filtered[filtered[col].isin(wanted)]
            else:
                filtered = filtered[filtered[col] == wanted]

    # 2) Apply spec filters, but skip any column already filtered by the UI
    spec_filters = spec.get("filters", {}) if spec else {}
    for col, wanted in spec_filters.items():
        if ui_filters and col in ui_filters:
            # UI already decided this column; skip the spec filter
            continue
        if col not in filtered.columns:
            continue

        if isinstance(wanted, list):
            filtered = filtered[filtered[col].isin(wanted)]
        else:
            filtered = filtered[filtered[col] == wanted]

    return filtered



def _format_chart_value(value, column_name: Optional[str] = None) -> str:
    """Format a chart value for the Read This Chart section."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)

    column_text = str(column_name or "").lower()
    if "percent" in column_text:
        return f"{numeric_value:.1f}%"
    if abs(numeric_value - round(numeric_value)) < 1e-9:
        return f"{int(round(numeric_value)):,}"
    return f"{numeric_value:,.1f}"


def generate_read_this_chart(
    df: pd.DataFrame,
    spec: dict,
    ui_filters: Optional[dict] = None,
) -> tuple[str, list[str]]:
    """
    Build a concise, deterministic chart description and a few data-based
    observations using the same filters that generated the visualization.
    """
    if not spec:
        return "This visualization summarizes the selected data.", []

    data = apply_spec(df, spec, ui_filters=ui_filters)
    if data.empty:
        return "This visualization contains no data after the selected filters were applied.", []

    chart_type_raw = str(spec.get("chart_type", "")).strip().lower()
    chart_names = {
        "bar": "bar chart",
        "stacked_bar": "stacked bar chart",
        "line": "line chart",
        "heatmap": "heatmap",
        "box": "box-and-whisker plot",
    }
    chart_name = chart_names.get(chart_type_raw, "chart")

    special_mode = spec.get("special_mode")
    x_col = spec.get("x")
    y_col = spec.get("y")
    group_col = spec.get("group")
    row_col = spec.get("row")
    col_col = spec.get("col")
    aggregation = spec.get("aggregation", "mean")
    use_agg = "mean" if aggregation in ["none", None] else aggregation

    # Normalize the same common heatmap structure handled by the plotter.
    if chart_type_raw == "heatmap":
        if row_col is None and group_col is not None:
            row_col = group_col
        if col_col is None and x_col is not None:
            col_col = x_col

    # Match the plotter's multi-grade behavior for line and bar charts.
    grade_dim = next(
        (column for column in ["Grade", "Grade Level", "Testing Grade"] if column in data.columns),
        None,
    )
    if (
        ui_filters
        and grade_dim
        and x_col in ["School Year", "Year"]
        and chart_type_raw in ["bar", "line", "stacked_bar"]
    ):
        selected_grades = ui_filters.get(grade_dim)
        if isinstance(selected_grades, list) and len(selected_grades) > 1:
            if group_col is None or group_col == grade_dim:
                group_col = grade_dim

    filters = dict(spec.get("filters", {}) or {})
    if ui_filters:
        filters.update(ui_filters)

    filter_parts = []
    for column, value in filters.items():
        if isinstance(value, list):
            if len(value) == 0:
                continue
            displayed = ", ".join(str(item) for item in value)
        else:
            displayed = str(value)
        filter_parts.append(f"{column}: {displayed}")

    filter_phrase = ""
    if filter_parts:
        filter_phrase = " The displayed data are filtered to " + "; ".join(filter_parts) + "."

    notices: list[str] = []

    # -----------------------------------------------------
    # Special WYTOPP stacked chart
    # -----------------------------------------------------
    if chart_type_raw == "stacked_bar" and special_mode == "wytopp_stacked":
        basic_col = "Percent Basic and Below"
        prof_col = "Percent Proficient and Advanced"
        description = (
            f"This stacked bar chart compares {basic_col} with {prof_col}"
            + (f" across {x_col}." if x_col else ".")
            + filter_phrase
        )

        if x_col in data.columns and prof_col in data.columns:
            grouped = data.groupby(x_col)[[basic_col, prof_col]].mean().reset_index()
            grouped = grouped.dropna(subset=[prof_col])
            if not grouped.empty:
                highest = grouped.loc[grouped[prof_col].idxmax()]
                lowest = grouped.loc[grouped[prof_col].idxmin()]
                notices.append(
                    f"The highest proficient-and-advanced value is "
                    f"{_format_chart_value(highest[prof_col], prof_col)} for {highest[x_col]}."
                )
                if len(grouped) > 1:
                    notices.append(
                        f"The lowest proficient-and-advanced value is "
                        f"{_format_chart_value(lowest[prof_col], prof_col)} for {lowest[x_col]}."
                    )
        return description, notices[:3]

    # -----------------------------------------------------
    # Line, bar, and ordinary stacked bar charts
    # -----------------------------------------------------
    if chart_type_raw in ["line", "bar", "stacked_bar"] and x_col and y_col:
        if chart_type_raw == "line":
            description = (
                f"This line chart displays {use_agg} {y_col} across {x_col}."
            )
            if group_col:
                description += f" Each line represents {group_col}."
        elif chart_type_raw == "bar":
            description = f"This bar chart compares {use_agg} {y_col} across {x_col}."
            if group_col:
                description += f" Bars are grouped by {group_col}."
        else:
            description = f"This stacked bar chart displays {use_agg} {y_col} across {x_col}."
            if group_col:
                description += f" Each stack represents {group_col}."
        description += filter_phrase

        group_columns = [x_col] + ([group_col] if group_col else [])
        if use_agg == "sum":
            grouped = data.groupby(group_columns)[y_col].sum().reset_index()
        elif use_agg == "count":
            grouped = data.groupby(group_columns)[y_col].count().reset_index()
        else:
            grouped = data.groupby(group_columns)[y_col].mean().reset_index()
        grouped = grouped.dropna(subset=[y_col])

        if not grouped.empty:
            if chart_type_raw == "line":
                if group_col:
                    changes = []
                    latest_rows = []
                    for group_value, subset in grouped.groupby(group_col):
                        subset = subset.sort_values(x_col)
                        if subset.empty:
                            continue
                        latest_rows.append(subset.iloc[-1])
                        if len(subset) >= 2:
                            change = float(subset.iloc[-1][y_col]) - float(subset.iloc[0][y_col])
                            changes.append((group_value, change, subset.iloc[0], subset.iloc[-1]))

                    if changes:
                        greatest = max(changes, key=lambda item: item[1])
                        direction = "increased" if greatest[1] >= 0 else "decreased"
                        notices.append(
                            f"{greatest[0]} shows the largest change, having {direction} by "
                            f"{_format_chart_value(abs(greatest[1]), y_col)} from "
                            f"{greatest[2][x_col]} to {greatest[3][x_col]}."
                        )

                        declines = [item for item in changes if item[1] < 0]
                        if declines:
                            largest_decline = min(declines, key=lambda item: item[1])
                            if largest_decline[0] != greatest[0]:
                                notices.append(
                                    f"{largest_decline[0]} shows the largest decline: "
                                    f"{_format_chart_value(abs(largest_decline[1]), y_col)}."
                                )

                    if latest_rows:
                        latest_df = pd.DataFrame(latest_rows)
                        highest_latest = latest_df.loc[latest_df[y_col].idxmax()]
                        notices.append(
                            f"At the latest displayed {x_col}, {highest_latest[group_col]} has the "
                            f"highest value at {_format_chart_value(highest_latest[y_col], y_col)}."
                        )
                else:
                    grouped = grouped.sort_values(x_col)
                    if len(grouped) >= 2:
                        first = grouped.iloc[0]
                        last = grouped.iloc[-1]
                        change = float(last[y_col]) - float(first[y_col])
                        direction = "increased" if change >= 0 else "decreased"
                        notices.append(
                            f"The value {direction} by {_format_chart_value(abs(change), y_col)} "
                            f"from {first[x_col]} to {last[x_col]}."
                        )
            else:
                highest = grouped.loc[grouped[y_col].idxmax()]
                lowest = grouped.loc[grouped[y_col].idxmin()]
                high_label = str(highest[x_col])
                low_label = str(lowest[x_col])
                if group_col:
                    high_label += f" / {highest[group_col]}"
                    low_label += f" / {lowest[group_col]}"
                notices.append(
                    f"The highest displayed value is {_format_chart_value(highest[y_col], y_col)} "
                    f"for {high_label}."
                )
                if len(grouped) > 1:
                    notices.append(
                        f"The lowest displayed value is {_format_chart_value(lowest[y_col], y_col)} "
                        f"for {low_label}."
                    )

        return description, notices[:3]

    # -----------------------------------------------------
    # Heatmaps
    # -----------------------------------------------------
    if chart_type_raw == "heatmap" and row_col and col_col and y_col:
        description = (
            f"This heatmap shows {use_agg} {y_col} for combinations of "
            f"{row_col} and {col_col}. Darker and lighter cells indicate relative differences."
            + filter_phrase
        )

        if use_agg == "sum":
            grouped = data.groupby([row_col, col_col])[y_col].sum().reset_index()
        else:
            grouped = data.groupby([row_col, col_col])[y_col].mean().reset_index()
        grouped = grouped.dropna(subset=[y_col])

        if not grouped.empty:
            highest = grouped.loc[grouped[y_col].idxmax()]
            lowest = grouped.loc[grouped[y_col].idxmin()]
            notices.append(
                f"The highest cell is {highest[row_col]} × {highest[col_col]} at "
                f"{_format_chart_value(highest[y_col], y_col)}."
            )
            if len(grouped) > 1:
                notices.append(
                    f"The lowest cell is {lowest[row_col]} × {lowest[col_col]} at "
                    f"{_format_chart_value(lowest[y_col], y_col)}."
                )
        return description, notices[:3]

    # -----------------------------------------------------
    # Box plots
    # -----------------------------------------------------
    if chart_type_raw == "box" and x_col and y_col:
        description = (
            f"This box-and-whisker plot displays the distribution of {y_col} across {x_col}. "
            "The center line marks the median, the box shows the middle half of values, "
            "and the whiskers show the broader spread."
            + filter_phrase
        )
        grouped = data[[x_col, y_col]].dropna()
        if not grouped.empty:
            medians = grouped.groupby(x_col)[y_col].median()
            highest_group = medians.idxmax()
            lowest_group = medians.idxmin()
            notices.append(
                f"{highest_group} has the highest median at "
                f"{_format_chart_value(medians.loc[highest_group], y_col)}."
            )
            if len(medians) > 1:
                notices.append(
                    f"{lowest_group} has the lowest median at "
                    f"{_format_chart_value(medians.loc[lowest_group], y_col)}."
                )
        return description, notices[:3]

    return (
        f"This {chart_name} summarizes the selected data using the requested fields."
        + filter_phrase,
        notices,
    )


def make_chart_from_spec(
    df: pd.DataFrame,
    spec: dict,
    title_override: Optional[str] = None,
    x_label_override: Optional[str] = None,
    y_label_override: Optional[str] = None,
    show_value_labels: bool = False,
    n_col_name: Optional[str] = None,
    ui_filters: Optional[dict] = None,
    color_palette: Optional[str] = None,
):


    """
    Convert an LLM spec into a matplotlib figure using your existing logic style.
    Returns: (fig, debug_message)
    """
    if not spec or "chart_type" not in spec:
        return None, "Spec missing chart_type."

    # Map spec chart types to your internal names
    chart_map = {
        "bar": "Bar",
        "stacked_bar": "Stacked Bar",
        "line": "Line",
        "heatmap": "Heatmap",
        "box": "Box",
    }
    chart_type = chart_map.get(str(spec.get("chart_type", "")).strip().lower())
    if chart_type is None:
        return None, f"Unsupported chart_type in spec: {spec.get('chart_type')}"

    # Pull fields
    special_mode = spec.get("special_mode")
    x_col = spec.get("x")
    y_col = spec.get("y")
    group_col = spec.get("group")
    row_col = spec.get("row")
    col_col = spec.get("col")
    agg = spec.get("aggregation", "mean")
    sort_x = spec.get("sort_x", "none")

    # Recognize WYTOPP stacked-bar requests even if the model omits special_mode.
    has_wytopp_stack_columns = all(
        column in df.columns
        for column in [
            "Percent Basic and Below",
            "Percent Proficient and Advanced",
        ]
    )
    if chart_type == "Stacked Bar" and has_wytopp_stack_columns:
        request_notes = str(spec.get("notes", "")).lower()
        if special_mode == "wytopp_stacked" or y_col is None or (
            "basic" in request_notes and "proficient" in request_notes
        ):
            special_mode = "wytopp_stacked"
            if x_col is None:
                x_col = "Grade" if "Grade" in df.columns else "School Year"
            y_col = None
            group_col = None

    # Normalize common heatmap specs. Models sometimes put heatmap
    # dimensions in x/group instead of row/col.
    if chart_type == "Heatmap":
        if row_col is None and group_col is not None:
            row_col = group_col
        if col_col is None and x_col is not None:
            col_col = x_col

    # Choose palette
    palette_name = color_palette or "Default"
    palette = PALETTES.get(palette_name, None)

    def get_color(index: int):
        if not palette:
            return None
        if not isinstance(palette, list) or len(palette) == 0:
            return None
        return palette[index % len(palette)]

    # Apply UI + spec filters (UI wins when both mention same column)
    data = apply_spec(df, spec, ui_filters=ui_filters)

    # Let UI grade selection drive grouping when multiple grades are chosen
    grade_dim = next(
        (c for c in ["Grade", "Grade Level", "Testing Grade"] if c in data.columns),
        None,
    )
    year_dim = x_col if x_col in ["School Year", "Year"] else None

    if (
        ui_filters
        and grade_dim
        and year_dim
        and chart_type in ["Bar", "Line", "Stacked Bar"]
    ):
        wanted_grades = ui_filters.get(grade_dim)
        if isinstance(wanted_grades, list) and len(wanted_grades) > 1:
            # If the spec hasn't already chosen a grouping column,
            # or is grouping by Grade anyway, use Grade as the group.
            if group_col is None or group_col == grade_dim:
                group_col = grade_dim

    # Basic validation
    def col_ok(c):
        return (c is None) or (c in data.columns)

    columns_to_validate = [x_col, y_col, group_col, row_col, col_col]
    missing = [
        column for column in columns_to_validate
        if column is not None and column not in data.columns
    ]
    if missing:
        return None, f"Spec referenced missing columns: {missing}"

    # Ensure aggregation is sane
    agg = agg if agg in ["mean", "sum", "count", "none"] else "mean"

    # ---------- Build chart ----------
       # ---------- Build chart ----------
    fig = None

    # BAR / STACKED / LINE
    if chart_type in ["Bar", "Stacked Bar", "Line"]:
        if x_col is None:
            return None, f"{chart_type} requires an x-axis column."

        use_agg = "mean" if agg == "none" else agg
        fig, ax = plt.subplots(figsize=(10, 5))

        # -------------------------------------------------
        # Special WYTOPP stacked-bar mode
        # -------------------------------------------------
        if chart_type == "Stacked Bar" and special_mode == "wytopp_stacked":
            basic_col = "Percent Basic and Below"
            prof_col = "Percent Proficient and Advanced"

            if not all(c in data.columns for c in [basic_col, prof_col]):
                return None, (
                    "WYTOPP stacked mode requires Percent Basic and Below and "
                    "Percent Proficient and Advanced."
                )

            if use_agg == "sum":
                grouped = data.groupby(x_col)[[basic_col, prof_col]].sum().reset_index()
            else:
                grouped = data.groupby(x_col)[[basic_col, prof_col]].mean().reset_index()

            if sort_x in ["ascending", "descending"]:
                grouped = grouped.sort_values(
                    x_col,
                    ascending=(sort_x == "ascending"),
                )

            categories = grouped[x_col].astype(str).tolist()
            x_positions = np.arange(len(categories))
            basic_values = pd.to_numeric(grouped[basic_col], errors="coerce").fillna(0).to_numpy()
            prof_values = pd.to_numeric(grouped[prof_col], errors="coerce").fillna(0).to_numpy()

            ax.bar(
                x_positions,
                basic_values,
                label="Basic & Below",
                color=get_color(0),
            )
            ax.bar(
                x_positions,
                prof_values,
                bottom=basic_values,
                label="Proficient & Advanced",
                color=get_color(1),
            )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(categories, rotation=45, ha="right")
            ax.set_xlabel(x_col)
            ax.set_ylabel("Percent")
            ax.set_ylim(0, 105)
            ax.set_title(f"Performance levels by {x_col}")
            ax.legend()

        else:
            if y_col is None:
                return None, f"{chart_type} requires a numeric y-axis column."

            group_cols = [x_col] + ([group_col] if group_col else [])
            if use_agg == "mean":
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()
            elif use_agg == "sum":
                grouped = data.groupby(group_cols)[y_col].sum().reset_index()
            elif use_agg == "count":
                grouped = data.groupby(group_cols)[y_col].count().reset_index()
            else:
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()

            if sort_x in ["ascending", "descending"]:
                grouped = grouped.sort_values(
                    by=x_col,
                    ascending=(sort_x == "ascending"),
                )

            if chart_type == "Bar":
                if group_col:
                    categories = grouped[x_col].unique()
                    groups = grouped[group_col].unique()
                    x_positions = np.arange(len(categories))
                    width = 0.8 / max(len(groups), 1)

                    for i, group_value in enumerate(groups):
                        subset = grouped[grouped[group_col] == group_value]
                        heights = [
                            subset.loc[subset[x_col] == category, y_col].iloc[0]
                            if not subset.loc[subset[x_col] == category].empty
                            else 0
                            for category in categories
                        ]
                        ax.bar(
                            x_positions + i * width,
                            heights,
                            width,
                            label=str(group_value),
                            color=get_color(i),
                        )

                    ax.set_xticks(x_positions + width * (len(groups) - 1) / 2)
                    ax.set_xticklabels(categories, rotation=45, ha="right")
                    ax.legend()
                else:
                    ax.bar(
                        grouped[x_col].astype(str),
                        grouped[y_col],
                        color=get_color(0),
                    )
                    plt.xticks(rotation=45, ha="right")

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{use_agg} of {y_col}")
                ax.set_title(f"{y_col} by {x_col}")

            elif chart_type == "Stacked Bar":
                if group_col is None:
                    return None, "Stacked bar requires a categorical group column."

                categories = grouped[x_col].unique()
                stacks = grouped[group_col].unique()
                x_positions = np.arange(len(categories))
                bottom = np.zeros(len(categories))

                for i, stack_value in enumerate(stacks):
                    subset = grouped[grouped[group_col] == stack_value]
                    heights = np.asarray([
                        subset.loc[subset[x_col] == category, y_col].iloc[0]
                        if not subset.loc[subset[x_col] == category].empty
                        else 0
                        for category in categories
                    ], dtype=float)

                    ax.bar(
                        x_positions,
                        heights,
                        bottom=bottom,
                        label=str(stack_value),
                        color=get_color(i),
                    )
                    bottom += heights

                ax.set_xticks(x_positions)
                ax.set_xticklabels(categories, rotation=45, ha="right")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{use_agg} of {y_col}")
                ax.set_title(f"{y_col} by {x_col}, stacked by {group_col}")
                ax.legend()

            elif chart_type == "Line":
                if group_col:
                    for i, (group_value, subset) in enumerate(grouped.groupby(group_col)):
                        ax.plot(
                            subset[x_col],
                            subset[y_col],
                            marker="o",
                            label=str(group_value),
                            color=get_color(i),
                        )
                    ax.legend()
                else:
                    ax.plot(
                        grouped[x_col],
                        grouped[y_col],
                        marker="o",
                        color=get_color(0),
                    )

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{use_agg} of {y_col}")
                ax.set_title(f"{y_col} over {x_col}")
                plt.xticks(rotation=45, ha="right")

#Heatmap
    elif chart_type == "Heatmap":
        if row_col is None or col_col is None or y_col is None:
            return None, (
                "The heatmap specification is incomplete. "
                f"Received row={row_col}, col={col_col}, y={y_col}, "
                f"x={x_col}, and group={group_col}."
            )

        use_agg = "mean" if agg in ["none", None] else agg
        if use_agg == "mean":
            grouped = data.groupby([row_col, col_col])[y_col].mean().reset_index()
        else:
            grouped = data.groupby([row_col, col_col])[y_col].sum().reset_index()

        pivot = grouped.pivot(index=row_col, columns=col_col, values=y_col)

        fig, ax = plt.subplots(figsize=(10, 5))
        if palette_name == "Greyscale":
            im = ax.imshow(pivot.values, aspect="auto", cmap="Greys")
        else:
            im = ax.imshow(pivot.values, aspect="auto")

        ax.set_xticks(np.arange(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
        ax.set_yticks(np.arange(pivot.shape[0]))
        ax.set_yticklabels(pivot.index.astype(str))

        ax.set_xlabel(col_col)
        ax.set_ylabel(row_col)
        ax.set_title(f"{y_col} ({use_agg}) by {row_col} × {col_col}")
        cbar = fig.colorbar(im)
        cbar.set_label(y_col)


    # BOX
    elif chart_type == "Box":
        if x_col is None or y_col is None:
            return None, "Box plot requires 'x' and 'y' in spec."

        plot_df = data[[x_col, y_col]].dropna()
        groups = [plot_df[plot_df[x_col] == g][y_col].values for g in plot_df[x_col].unique()]
        labels = plot_df[x_col].unique().astype(str)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(groups, labels=labels)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Distribution of {y_col} by {x_col}")
        plt.xticks(rotation=45, ha="right")
    # --- Apply manual overrides, if provided ---
       # --- Apply manual overrides, if provided ---
    try:
        ax = plt.gca()
        if title_override:
            ax.set_title(title_override)
        if x_label_override:
            ax.set_xlabel(x_label_override)
        if y_label_override:
            ax.set_ylabel(y_label_override)
    except Exception:
        pass

    # --- Add value labels if requested ---
    try:
        if show_value_labels and fig is not None:
            ax = plt.gca()
            # Bar / stacked bar: label bars
            if chart_type in ["Bar", "Stacked Bar"]:
                for patch in ax.patches:
                    height = patch.get_height()
                    if np.isnan(height):
                        continue
                    x = patch.get_x() + patch.get_width() / 2
                    y = patch.get_y() + height
                    ax.text(
                        x,
                        y,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            # Line: label points
            elif chart_type == "Line":
                # For each line on the axes
                for line in ax.get_lines():
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    for x, y in zip(x_data, y_data):
                        if np.isnan(y):
                            continue
                        ax.text(
                            x,
                            y,
                            f"{y:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )
    except Exception:
        # If anything goes wrong with labels, fail silently
        pass

    # --- Add N counts if a column is provided ---
    try:
        if n_col_name and n_col_name in df.columns and fig is not None:
            ax = plt.gca()

            # Use the same filtered data that the chart is using
            data_for_n = data


            # For bar/stacked/line, aggregate N by x-axis
            if chart_type in ["Bar", "Stacked Bar", "Line"] and x_col is not None:
                n_by_x = (
                    data_for_n.groupby(x_col)[n_col_name]
                    .sum()
                    .to_dict()
                )

                # Adjust x tick labels: e.g., "3rd\nN=45"
                tick_labels = ax.get_xticklabels()
                new_labels = []
                for lbl in tick_labels:
                    txt = lbl.get_text()
                    # Look up N for this category if possible
                    n_val = n_by_x.get(txt, None)
                    if n_val is not None:
                        new_labels.append(f"{txt}\nN={int(n_val)}")
                    else:
                        new_labels.append(txt)

                ax.set_xticklabels(new_labels, rotation=45, ha="right")
    except Exception:
        # Again, don't let N annotation crash the chart
        pass

    return fig, "OK"



# -------------------------
# Streamlit page settings & CSS
# -------------------------
st.set_page_config(page_title="VizCreate", layout="wide")

# Light blue-green background and big buttons
st.markdown(
    """
    <style>
    body, .main {
        background-color: #e7f6f7 !important;  /* soft blue-green */
    }

    /* Make all st.button elements larger */
    .stButton > button {
        padding-top: 24px;
        padding-bottom: 24px;
        font-size: 20px !important;
        border-radius: 14px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.2em;'>VizCreate</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#555;'>Describe the visualization you want, "
    "then refine it with the controls below.</p>",
    unsafe_allow_html=True,
)

# -------------------------
# 1. Chart-type gallery (always visible)
# -------------------------
st.subheader("1. Choose a chart style (optional)")

if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar"
if "prompt" not in st.session_state:
    st.session_state.prompt = ""


def choose_chart(chart_type: str, example_prompt: str):
    st.session_state.chart_type = chart_type
    st.session_state.prompt = example_prompt


col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊\nBar Chart", use_container_width=True, key="tile_bar"):
        choose_chart(
            "Bar",
            "Create a bar chart showing the average of a numeric score by a category "
            "(e.g., average Math Scale Score by Grade).",
        )

with col2:
    if st.button("🧱\nStacked Bar", use_container_width=True, key="tile_stacked_bar"):
        choose_chart(
            "Stacked Bar",
            "Create a stacked bar chart showing Basic & Below vs Proficient & Advanced by Grade.",
        )

with col3:
    if st.button("📈\nLine Chart", use_container_width=True, key="tile_line"):
        choose_chart(
            "Line",
            "Create a line chart showing a score over time (e.g., average Math Scale Score by School Year).",
        )

with col4:
    if st.button("🟩\nHeatmap", use_container_width=True, key="tile_heatmap"):
        choose_chart(
            "Heatmap",
            "Create a heatmap where rows are grades, columns are tests or subscores, and colors show the average score.",
        )

with col5:
    if st.button("📦\nBox Plot", use_container_width=True, key="tile_box"):
        choose_chart(
            "Box",
            "Create a box-and-whisker plot showing the distribution of a score by grade or subgroup.",
        )

# -------------------------
# 2. Prompt (always visible)
# -------------------------
st.subheader("2. Describe your visualization")

prompt = st.text_area(
    "Prompt",
    value=st.session_state.prompt,
    height=100,
    placeholder=(
        "Example: Create a stacked bar chart showing Percent Basic and Below vs "
        "Percent Proficient and Advanced by Grade."
    ),
)

st.markdown(
    "<p style='color:#777; font-size:0.9em;'>"
    "In a later version, VizCreate will fully control the chart from this prompt. "
    "For now, the agent can suggest a spec, and you can still refine via the controls below."
    "</p>",
    unsafe_allow_html=True,
)

# -------------------------
# 3. File upload
# -------------------------
st.subheader("3. Upload your data")

uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx", "xls"],
    help="Use the same style of file you use for Vizit / VizitOverTime.",
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file to configure and generate a chart.")
    st.stop()

# -------------------------
# 4. Once file is uploaded: read + preview
# -------------------------
if uploaded_file.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

# Clean column names a bit
df.columns = (
    df.columns.astype(str)
    .str.replace("\n", " ")
    .str.replace("  ", " ")
    .str.strip()
)
# Try to auto-detect the N/enrollment column
DEFAULT_N_COL = "Number of Students Tested" if "Number of Students Tested" in df.columns else None


st.subheader("4. Data preview")
st.dataframe(df.head())

family_name, family_guidance = detect_dataset_family(df)
st.caption(f"Detected dataset family: {family_name}")

# -------------------------
# 5. Let VizCreate interpret the prompt (agent beta)
# -------------------------
st.subheader("5. Let VizCreate interpret your prompt (beta)")

use_ai = st.button("Let VizCreate interpret my prompt", key="use_ai_button")

if "agent_spec" not in st.session_state:
    st.session_state.agent_spec = None

if use_ai:
    st.session_state.agent_spec = get_chart_spec_from_llm(df, prompt)
    st.success("Spec received from VizCreate agent.")
    st.write("Agent spec (debug):", st.session_state.agent_spec)


# -------------------------
# Type inference
# -------------------------
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in df.columns if c not in numeric_cols]

if not numeric_cols:
    st.error("No numeric columns detected. You’ll need at least one numeric column to plot.")
    st.stop()


spec = st.session_state.agent_spec

if spec:
    st.subheader("6. Visualization (from prompt)")

    # --- Advanced display options ---
    with st.expander("Advanced display options"):
        title_override = st.text_input(
            "Chart title (optional override)",
            value="",
            key="title_override",
        )
        x_label_override = st.text_input(
            "X-axis label (optional override)",
            value="",
            key="x_label_override",
        )
        y_label_override = st.text_input(
            "Y-axis label (optional override)",
            value="",
            key="y_label_override",
        )

        show_value_labels = st.checkbox(
            "Show value labels on chart (bars/lines)",
            value=True,
            key="show_value_labels",
        )

        n_col_name_input = st.text_input(
            "Name of N/enrollment column (optional, e.g., 'Number of Students Tested')",
            value=DEFAULT_N_COL or "",
            key="n_col_name_input",
        )
        color_scheme = st.selectbox(
            "Color scheme",
            [
                "Default",
                "Greyscale",
                "UW Brown & Gold",
                "Blue-Orange",
                "Colorblind-friendly",
                "Muted",
            ],
            index=0,
            key="color_scheme",
        )

    # Convert empty strings to None so the helper keeps defaults if not overridden
    title_arg = title_override.strip() or None
    x_label_arg = x_label_override.strip() or None
    y_label_arg = y_label_override.strip() or None
    n_col_arg = n_col_name_input.strip() or None
    color_scheme_arg = color_scheme


    # --- Quick filters (optional) ---
    st.markdown("**Quick filters (optional)**")

    # We'll build a dictionary of UI filters to pass into make_chart_from_spec
    ui_filters = {}

    # Grade filter (if available)
    grade_col = None
    for candidate in ["Grade", "Grade Level", "Testing Grade"]:
        if candidate in df.columns:
            grade_col = candidate
            break

    if grade_col is not None:
        grade_vals = sorted(df[grade_col].dropna().unique())
        selected_grades = st.multiselect(
            f"Filter by {grade_col}",
            options=grade_vals,
            default=grade_vals,
            key="grade_filter",
        )
        if selected_grades:
            ui_filters[grade_col] = selected_grades

    # School Year filter (if available)
    year_col = None
    for candidate in ["School Year", "Year"]:
        if candidate in df.columns:
            year_col = candidate
            break

    if year_col is not None:
        year_vals = list(df[year_col].dropna().unique())
        selected_years = st.multiselect(
            f"Filter by {year_col}",
            options=year_vals,
            default=year_vals,
            key="year_filter",
        )
        if selected_years:
            ui_filters[year_col] = selected_years

    # --- Generate the chart from the spec on EVERY rerun ---
    fig, msg = make_chart_from_spec(
        df,
        spec,
        title_override=title_arg,
        x_label_override=x_label_arg,
        y_label_override=y_label_arg,
        show_value_labels=show_value_labels,
        n_col_name=n_col_arg,
        ui_filters=ui_filters,
        color_palette=color_scheme_arg,
    )


    if fig is None:
        st.error(f"Could not generate chart from spec: {msg}")
    else:
        st.pyplot(fig)

        read_description, notice_items = generate_read_this_chart(
            df,
            spec,
            ui_filters=ui_filters,
        )

        st.markdown("### 📖 Read This Chart")
        st.write(read_description)

        st.markdown("#### What to Notice")
        if notice_items:
            for notice in notice_items:
                st.markdown(f"- {notice}")
        else:
            st.caption(
                "Use the chart and quick filters to compare categories, trends, "
                "and areas of relative strength or need."
            )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)

        st.download_button(
            label="Download PNG",
            data=buf,
            file_name="vizcreate_chart.png",
            mime="image/png",
            key="download_png_from_spec",
        )

# -------------------------
# 6. Configure your chart (manual controls) Expander added
# -------------------------
with st.expander("Advanced: manually configure the chart"):
    st.subheader("Configure your chart")

    chart_type = st.selectbox(
        "Chart type",
        ["Bar", "Stacked Bar", "Line", "Heatmap", "Box"],
        index=["Bar", "Stacked Bar", "Line", "Heatmap", "Box"].index(
            st.session_state.chart_type
        ),
    )

# Defaults
    x_col: Optional[str] = None
    y_col: Optional[str] = None
    group_col: Optional[str] = None
    row_col: Optional[str] = None
    col_col: Optional[str] = None
    agg_func: str = "mean"

    if chart_type in ["Bar", "Stacked Bar", "Line"]:
        st.markdown("**Axis & grouping**")

        x_col = st.selectbox(
            "X-axis (category or time)",
            options=categorical_cols + numeric_cols,
            index=0 if categorical_cols else 0,
        )

        y_col = st.selectbox(
            "Y-axis (numeric to summarize)",
            options=numeric_cols,
            index=0,
        )

        if chart_type == "Stacked Bar":
            group_col = st.selectbox(
                "Stack by (category)",
                options=[c for c in df.columns if c != x_col],
                index=0,
            )
        else:  # Bar / Line
            group_col = st.selectbox(
                "Optional grouping (color/series)",
                options=["(none)"] + [c for c in df.columns if c != x_col],
                index=0,
            )
            if group_col == "(none)":
                group_col = None

        agg_func = st.selectbox(
            "How to summarize Y",
            options=["mean", "sum", "count"],
            index=0,
        )

    elif chart_type == "Heatmap":
        st.markdown("**Heatmap layout**")
        row_col = st.selectbox("Rows (category)", options=categorical_cols, index=0)
        col_col = st.selectbox(
            "Columns (category)", options=[c for c in categorical_cols if c != row_col],
            index=0,
        )
        y_col = st.selectbox(
            "Color value (numeric)", options=numeric_cols, index=0
        )
        agg_func = st.selectbox(
            "How to summarize value",
            options=["mean", "sum"],
            index=0,
        )

    elif chart_type == "Box":
        st.markdown("**Box plot layout**")
        x_col = st.selectbox("Group by (category)", options=categorical_cols, index=0)
        y_col = st.selectbox("Value (numeric)", options=numeric_cols, index=0)

# -------------------------
# 7. Generate chart
# -------------------------
    generate = st.button("Generate Visualization", type="primary")

    fig = None

    if generate:
        if chart_type in ["Bar", "Stacked Bar", "Line"]:
            data = df.copy()

        # Aggregate
            group_cols = [x_col] + ([group_col] if group_col else [])
            if agg_func == "mean":
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()
            elif agg_func == "sum":
                grouped = data.groupby(group_cols)[y_col].sum().reset_index()
            elif agg_func == "count":
                grouped = data.groupby(group_cols)[y_col].count().reset_index()
            else:
                grouped = data.groupby(group_cols)[y_col].mean().reset_index()

            fig, ax = plt.subplots(figsize=(8, 5))

            if chart_type == "Bar":
                if group_col:
                    categories = grouped[x_col].unique()
                    groups = grouped[group_col].unique()
                    x = np.arange(len(categories))
                    width = 0.8 / len(groups)

                    for i, g in enumerate(groups):
                        sub = grouped[grouped[group_col] == g]
                        heights = [
                            sub[sub[x_col] == cat][y_col].values[0]
                            if not sub[sub[x_col] == cat].empty
                            else 0
                            for cat in categories
                        ]
                        ax.bar(x + i * width, heights, width, label=str(g))
                    ax.set_xticks(x + width * (len(groups) - 1) / 2)
                    ax.set_xticklabels(categories, rotation=45, ha="right")
                    ax.legend()
                else:
                    ax.bar(grouped[x_col].astype(str), grouped[y_col])
                    ax.set_xticklabels(grouped[x_col].astype(str), rotation=45, ha="right")

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"{chart_type} of {y_col} by {x_col}")

            elif chart_type == "Stacked Bar":
                categories = grouped[x_col].unique()
                stacks = grouped[group_col].unique()
                x = np.arange(len(categories))
                bottom = np.zeros(len(categories))

                fig, ax = plt.subplots(figsize=(8, 5))
                for s in stacks:
                    sub = grouped[grouped[group_col] == s]
                    heights = [
                        sub[sub[x_col] == cat][y_col].values[0]
                        if not sub[sub[x_col] == cat].empty
                        else 0
                        for cat in categories
                    ]
                    ax.bar(x, heights, bottom=bottom, label=str(s))
                    bottom += np.array(heights)

                ax.set_xticks(x)
                ax.set_xticklabels(categories, rotation=45, ha="right")
                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"Stacked Bar of {y_col} by {x_col} and {group_col}")
                ax.legend()

            elif chart_type == "Line":
                fig, ax = plt.subplots(figsize=(8, 5))
                if group_col:
                    for g, sub in grouped.groupby(group_col):
                        ax.plot(sub[x_col], sub[y_col], marker="o", label=str(g))
                    ax.legend()
                else:
                    ax.plot(grouped[x_col], grouped[y_col], marker="o")

                ax.set_xlabel(x_col)
                ax.set_ylabel(f"{agg_func} of {y_col}")
                ax.set_title(f"Line chart of {y_col} by {x_col}")
                plt.xticks(rotation=45, ha="right")

        elif chart_type == "Heatmap":
            data = df.copy()
            if agg_func == "mean":
                grouped = data.groupby([row_col, col_col])[y_col].mean().reset_index()
            else:
                grouped = data.groupby([row_col, col_col])[y_col].sum().reset_index()

            pivot = grouped.pivot(index=row_col, columns=col_col, values=y_col)
            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(pivot.values, aspect="auto")

            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels(pivot.index.astype(str))

            ax.set_xlabel(col_col)
            ax.set_ylabel(row_col)
            ax.set_title(f"Heatmap of {y_col} ({agg_func}) by {row_col} × {col_col}")
            cbar = fig.colorbar(im)
            cbar.set_label(y_col)

        elif chart_type == "Box":
            data = df[[x_col, y_col]].dropna()
            groups = [data[data[x_col] == g][y_col].values for g in data[x_col].unique()]
            labels = data[x_col].unique().astype(str)

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(groups, labels=labels)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Distribution of {y_col} by {x_col}")
            plt.xticks(rotation=45, ha="right")

    # Show figure + download
        if fig is not None:
            st.subheader("7. Visualization")
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)

            st.download_button(
                label="Download PNG",
                data=buf,
                file_name="vizcreate_chart.png",
                mime="image/png",
            )
