"""Matplotlib chart generation for VizCreate.

Phase 1, Step 5C keeps the public ``make_chart_from_spec`` interface
unchanged while separating each chart type into a focused builder.
"""

from collections.abc import Callable
import re
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from config.palettes import PALETTES, get_heatmap_cmap
from core.filters import apply_spec


ColorGetter = Callable[[int], Optional[str]]


def _normalize_chart_spec(
    df: pd.DataFrame,
    spec: dict,
    ui_filters: Optional[dict] = None,
) -> tuple[Optional[dict], Optional[str]]:
    """Return a repaired copy of an LLM chart spec before plotting."""
    if not spec or "chart_type" not in spec:
        return None, "Spec missing chart_type."

    normalized = dict(spec)

    chart_map = {
        "bar": "Bar",
        "stacked_bar": "Stacked Bar",
        "line": "Line",
        "heatmap": "Heatmap",
        "box": "Box",
        "scatter": "Scatter",
    }
    chart_type = chart_map.get(str(normalized.get("chart_type", "")).strip().lower())
    if chart_type is None:
        return None, f"Unsupported chart_type in spec: {normalized.get('chart_type')}"
    normalized["chart_type_internal"] = chart_type

    agg = normalized.get("aggregation", "mean")
    normalized["aggregation"] = agg if agg in ["mean", "sum", "count", "none"] else "mean"

    sort_x = normalized.get("sort_x", "none")
    normalized["sort_x"] = sort_x if sort_x in ["none", "ascending", "descending"] else "none"

    # Repair common heatmap specs that place dimensions in x/group.
    if chart_type == "Heatmap":
        if normalized.get("row") is None and normalized.get("group") is not None:
            normalized["row"] = normalized.get("group")
        if normalized.get("col") is None and normalized.get("x") is not None:
            normalized["col"] = normalized.get("x")

    # Recognize a wide-format multi-measure grouped bar chart.
    if chart_type == "Bar":
        item_columns = [
            column
            for column in normalized.get("item_columns", [])
            if isinstance(column, str) and column in df.columns
        ]
        if normalized.get("y") is None and normalized.get("x") and item_columns:
            normalized["special_mode"] = "multi_measure_bar"
            normalized["item_columns"] = item_columns

    # Recognize a wide-format multi-measure box plot. In this structure,
    # x identifies the comparison group (for example Grade Level) and
    # item_columns contains the numeric categories to compare.
    if chart_type == "Box":
        item_columns = [
            column
            for column in normalized.get("item_columns", [])
            if isinstance(column, str) and column in df.columns
        ]
        if normalized.get("y") is None and normalized.get("x") and item_columns:
            normalized["special_mode"] = "multi_measure_box"
            normalized["item_columns"] = item_columns

    # Recognize the special two-column WYTOPP stacked chart.
    has_wytopp_columns = all(
        column in df.columns
        for column in ["Percent Basic and Below", "Percent Proficient and Advanced"]
    )
    if chart_type == "Stacked Bar" and has_wytopp_columns:
        notes = str(normalized.get("notes", "")).lower()
        if (
            normalized.get("special_mode") == "wytopp_stacked"
            or normalized.get("y") is None
            or ("basic" in notes and "proficient" in notes)
        ):
            normalized["special_mode"] = "wytopp_stacked"
            if normalized.get("x") is None:
                normalized["x"] = "Grade" if "Grade" in df.columns else "School Year"
            normalized["y"] = None
            normalized["group"] = None

    # Aggregated WYTOPP rows do not support a true box-plot distribution.
    wytopp_required = {
        "School Year",
        "Grade",
        "Subject",
        "Percent Basic and Below",
        "Percent Proficient and Advanced",
    }
    if chart_type == "Box" and wytopp_required.issubset(set(df.columns.astype(str))):
        normalized.update(
            {
                "chart_type": "bar",
                "chart_type_internal": "Bar",
                "special_mode": None,
                "x": "Grade",
                "y": "Percent Proficient and Advanced",
                "group": None,
                "row": None,
                "col": None,
                "aggregation": "mean",
                "sort_x": "ascending",
            }
        )

    # Wide-format Likert construct comparison is an approved special mode.
    if normalized.get("special_mode") == "likert_construct_summary":
        item_columns = normalized.get("item_columns", [])
        if not isinstance(item_columns, list) or not item_columns:
            return None, "Likert construct summary requires a non-empty item_columns list."

        missing_items = [
            column for column in item_columns
            if column not in df.columns
        ]
        if missing_items:
            return None, f"Likert item_columns referenced missing columns: {missing_items}"

        non_numeric_items = [
            column for column in item_columns
            if not pd.api.types.is_numeric_dtype(df[column])
        ]
        if non_numeric_items:
            return None, (
                "Likert construct summary currently requires numeric-coded item columns: "
                f"{non_numeric_items}"
            )

        normalized.update(
            {
                "chart_type": "bar",
                "chart_type_internal": "Bar",
                "x": None,
                "y": None,
                "group": None,
                "row": None,
                "col": None,
                "aggregation": "mean",
            }
        )

    # When several grades are selected for a time chart, show one series per grade.
    grade_dim = next(
        (column for column in ["Grade", "Grade Level", "Testing Grade"] if column in df.columns),
        None,
    )
    if (
        ui_filters
        and grade_dim
        and normalized.get("x") in ["School Year", "Year"]
        and normalized["chart_type_internal"] in ["Bar", "Line", "Stacked Bar"]
    ):
        selected = ui_filters.get(grade_dim)
        if isinstance(selected, list) and len(selected) > 1:
            if normalized.get("group") is None or normalized.get("group") == grade_dim:
                normalized["group"] = grade_dim

    return normalized, None


def _make_color_getter(palette_name: str) -> ColorGetter:
    """Return a cycling color getter for categorical charts."""
    palette = PALETTES.get(palette_name)

    def get_color(index: int) -> Optional[str]:
        if not palette:
            return None
        return palette[index % len(palette)]

    return get_color


def _aggregate_chart_data(
    data: pd.DataFrame,
    group_columns: list[str],
    y_col: str,
    aggregation: str,
) -> pd.DataFrame:
    """Aggregate one numeric measure for bar, stacked-bar, and line charts."""
    grouped = data.groupby(group_columns)[y_col]
    if aggregation == "sum":
        return grouped.sum().reset_index()
    if aggregation == "count":
        return grouped.count().reset_index()
    return grouped.mean().reset_index()


def _sort_chart_data(
    grouped: pd.DataFrame,
    x_col: str,
    sort_x: str,
) -> pd.DataFrame:
    """Apply optional x-axis sorting without failing on mixed types."""
    if sort_x not in ["ascending", "descending"]:
        return grouped
    try:
        return grouped.sort_values(
            by=x_col,
            ascending=(sort_x == "ascending"),
        )
    except Exception:
        return grouped


def _make_wytopp_stacked_chart(
    data: pd.DataFrame,
    x_col: str,
    aggregation: str,
    sort_x: str,
    get_color: ColorGetter,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Build the special two-measure WYTOPP proficiency stacked bar chart."""
    basic_col = "Percent Basic and Below"
    prof_col = "Percent Proficient and Advanced"

    if not all(column in data.columns for column in [basic_col, prof_col]):
        return None, None, (
            "WYTOPP stacked mode requires Percent Basic and Below and "
            "Percent Proficient and Advanced."
        )

    grouped_source = data.groupby(x_col)[[basic_col, prof_col]]
    if aggregation == "sum":
        grouped = grouped_source.sum().reset_index()
    else:
        grouped = grouped_source.mean().reset_index()
    grouped = _sort_chart_data(grouped, x_col, sort_x)

    categories = grouped[x_col].astype(str).tolist()
    x_positions = np.arange(len(categories))
    basic_values = pd.to_numeric(grouped[basic_col], errors="coerce").fillna(0).to_numpy()
    prof_values = pd.to_numeric(grouped[prof_col], errors="coerce").fillna(0).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
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
    return fig, ax, None


def _make_bar_chart(
    grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str],
    aggregation: str,
    get_color: ColorGetter,
) -> tuple[Figure, Axes]:
    """Build a single-series or grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if group_col:
        categories = grouped[x_col].unique()
        groups = grouped[group_col].unique()
        x_positions = np.arange(len(categories))
        width = 0.8 / max(len(groups), 1)

        for index, group_value in enumerate(groups):
            subset = grouped[grouped[group_col] == group_value]
            heights = [
                subset.loc[subset[x_col] == category, y_col].iloc[0]
                if not subset.loc[subset[x_col] == category].empty
                else 0
                for category in categories
            ]
            ax.bar(
                x_positions + index * width,
                heights,
                width,
                label=str(group_value),
                color=get_color(index),
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
        ax.tick_params(axis="x", labelrotation=45)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{aggregation} of {y_col}")
    ax.set_title(f"{y_col} by {x_col}")
    return fig, ax


def _make_stacked_bar_chart(
    grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str],
    aggregation: str,
    get_color: ColorGetter,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Build an ordinary long-format stacked bar chart."""
    if group_col is None:
        return None, None, "Stacked bar requires a categorical group column."

    categories = grouped[x_col].unique()
    stacks = grouped[group_col].unique()
    x_positions = np.arange(len(categories))
    bottom = np.zeros(len(categories))

    fig, ax = plt.subplots(figsize=(10, 5))
    for index, stack_value in enumerate(stacks):
        subset = grouped[grouped[group_col] == stack_value]
        heights = np.asarray(
            [
                subset.loc[subset[x_col] == category, y_col].iloc[0]
                if not subset.loc[subset[x_col] == category].empty
                else 0
                for category in categories
            ],
            dtype=float,
        )
        ax.bar(
            x_positions,
            heights,
            bottom=bottom,
            label=str(stack_value),
            color=get_color(index),
        )
        bottom += heights

    ax.set_xticks(x_positions)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{aggregation} of {y_col}")
    ax.set_title(f"{y_col} by {x_col}, stacked by {group_col}")
    ax.legend()
    return fig, ax, None


def _make_line_chart(
    grouped: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: Optional[str],
    aggregation: str,
    get_color: ColorGetter,
) -> tuple[Figure, Axes]:
    """Build a single-series or grouped line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if group_col:
        for index, (group_value, subset) in enumerate(grouped.groupby(group_col)):
            ax.plot(
                subset[x_col],
                subset[y_col],
                marker="o",
                label=str(group_value),
                color=get_color(index),
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
    ax.set_ylabel(f"{aggregation} of {y_col}")
    ax.set_title(f"{y_col} over {x_col}")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return fig, ax



def _make_scatter_plot(
    data: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
    group_col: Optional[str],
    get_color: ColorGetter,
    special_mode: Optional[str] = None,
    label_col: Optional[str] = None,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Build a scatterplot for the relationship between two numeric variables."""
    if x_col is None or y_col is None:
        return None, None, "Scatterplot requires 'x' and 'y' numeric columns."
    if x_col == y_col:
        return None, None, "Scatterplot x and y must be different columns."
    if not pd.api.types.is_numeric_dtype(data[x_col]):
        return None, None, f"Scatterplot x column must be numeric: {x_col}"
    if not pd.api.types.is_numeric_dtype(data[y_col]):
        return None, None, f"Scatterplot y column must be numeric: {y_col}"

    columns = [x_col, y_col]
    if group_col:
        columns.append(group_col)
    if label_col and label_col in data.columns:
        columns.append(label_col)
    columns = list(dict.fromkeys(columns))
    plot_df = data[columns].dropna(subset=[x_col, y_col])

    if plot_df.empty:
        return None, None, "No complete numeric pairs are available for the scatterplot."

    fig, ax = plt.subplots(figsize=(8.5, 6))

    if group_col:
        for index, (group_value, subset) in enumerate(plot_df.groupby(group_col)):
            ax.scatter(
                subset[x_col],
                subset[y_col],
                label=str(group_value),
                alpha=0.78,
                color=get_color(index),
            )
        ax.legend(title=group_col)
    else:
        ax.scatter(
            plot_df[x_col],
            plot_df[y_col],
            alpha=0.78,
            color=get_color(0),
        )

    if special_mode == "student_support_map":
        x_median = float(plot_df[x_col].median())
        y_median = float(plot_df[y_col].median())
        ax.axvline(x_median, linestyle=":", linewidth=1.2, color="gray")
        ax.axhline(y_median, linestyle=":", linewidth=1.2, color="gray")

        if label_col and label_col in plot_df.columns:
            for _, row in plot_df.iterrows():
                ax.annotate(
                    str(row[label_col]),
                    (row[x_col], row[y_col]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.78,
                )

        ax.text(
            0.98,
            0.98,
            "Relatively higher on both",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.text(
            0.02,
            0.02,
            "Relatively lower on both",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )

    # Add a simple least-squares trend line when variation exists.
    if (
        plot_df[x_col].nunique() >= 2
        and plot_df[y_col].nunique() >= 2
    ):
        x_values = plot_df[x_col].astype(float).to_numpy()
        y_values = plot_df[y_col].astype(float).to_numpy()
        slope, intercept = np.polyfit(x_values, y_values, 1)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            linestyle="--",
            linewidth=1.5,
            color=get_color(1),
            label="Linear trend",
        )
        if not group_col:
            ax.legend()

    correlation = plot_df[[x_col, y_col]].corr().iloc[0, 1]
    if pd.notna(correlation):
        ax.text(
            0.02,
            0.98,
            f"Pearson r = {correlation:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} in Relation to {x_col}")
    fig.tight_layout()
    return fig, ax, None

def _make_heatmap(
    data: pd.DataFrame,
    row_col: Optional[str],
    col_col: Optional[str],
    y_col: Optional[str],
    aggregation: str,
    palette_name: str,
    x_col: Optional[str],
    group_col: Optional[str],
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Build an aggregated heatmap using a continuous palette-specific colormap."""
    if row_col is None or col_col is None or y_col is None:
        return None, None, (
            "The heatmap specification is incomplete. "
            f"Received row={row_col}, col={col_col}, y={y_col}, "
            f"x={x_col}, and group={group_col}."
        )

    if y_col in {row_col, col_col}:
        return None, None, (
            "A heatmap requires two categorical dimensions and a separate numeric value. "
            f"Received row={row_col}, col={col_col}, y={y_col}."
        )
    if row_col == col_col:
        return None, None, "Heatmap row and column dimensions must be different."

    grouped_source = data.groupby([row_col, col_col])[y_col]
    if aggregation == "sum":
        grouped = grouped_source.sum().reset_index()
    else:
        grouped = grouped_source.mean().reset_index()

    pivot = grouped.pivot(index=row_col, columns=col_col, values=y_col)
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap=get_heatmap_cmap(palette_name),
    )

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.astype(str))
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    ax.set_title(f"{y_col} ({aggregation}) by {row_col} × {col_col}")
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label(y_col)
    return fig, ax, None


def _make_box_plot(
    data: pd.DataFrame,
    x_col: Optional[str],
    y_col: Optional[str],
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Build a box-and-whisker plot for observation-level data."""
    if x_col is None or y_col is None:
        return None, None, "Box plot requires 'x' and 'y' in spec."

    plot_df = data[[x_col, y_col]].dropna()
    categories = plot_df[x_col].unique()
    groups = [plot_df[plot_df[x_col] == category][y_col].values for category in categories]
    labels = categories.astype(str)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(groups, tick_labels=labels)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Distribution of {y_col} by {x_col}")
    ax.tick_params(axis="x", labelrotation=45)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    return fig, ax, None



def _make_multi_measure_bar_chart(
    data: pd.DataFrame,
    group_col: Optional[str],
    item_columns: list[str],
    get_color: ColorGetter,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Compare mean scores across wide-format measures and categorical groups."""
    if group_col is None:
        return None, None, "Multi-measure bar chart requires a comparison-group column."

    valid_items = [column for column in item_columns if column in data.columns]
    if not valid_items:
        return None, None, "No valid numeric measure columns were supplied."

    working = data[[group_col] + valid_items].copy()
    for column in valid_items:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    long_df = working.melt(
        id_vars=[group_col],
        value_vars=valid_items,
        var_name="Measure",
        value_name="Score",
    ).dropna(subset=[group_col, "Score"])

    if long_df.empty:
        return None, None, "The selected measure columns contain no numeric scores."

    summary = (
        long_df.groupby(["Measure", group_col], dropna=False)["Score"]
        .mean()
        .reset_index()
    )

    groups = list(pd.unique(summary[group_col]))
    measures = valid_items
    centers = np.arange(len(measures), dtype=float)
    width = min(0.8 / max(len(groups), 1), 0.32)

    fig_width = max(10.0, min(20.0, 1.55 * len(measures)))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    for group_index, group_value in enumerate(groups):
        values = []
        for measure in measures:
            match = summary.loc[
                (summary["Measure"] == measure)
                & (summary[group_col] == group_value),
                "Score",
            ]
            values.append(float(match.iloc[0]) if not match.empty else np.nan)

        offset = (group_index - (len(groups) - 1) / 2) * width
        kwargs = {}
        color = get_color(group_index)
        if color is not None:
            kwargs["color"] = color

        ax.bar(
            centers + offset,
            values,
            width=width * 0.9,
            label=str(group_value),
            **kwargs,
        )

    labels = [_humanize_survey_item(column) for column in measures]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Category or subscore")
    ax.set_ylabel("Mean score")
    ax.set_title(f"Average Category Scores by {group_col}")
    ax.legend(title=group_col)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig, ax, None


def _make_multi_measure_box_plot(
    data: pd.DataFrame,
    group_col: Optional[str],
    item_columns: list[str],
    get_color: ColorGetter,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Compare several wide-format numeric measures across categorical groups."""
    if group_col is None:
        return None, None, "Multi-measure box plot requires a comparison-group column."
    valid_items = [
        column for column in item_columns
        if column in data.columns
    ]
    if not valid_items:
        return None, None, "No valid numeric item columns were supplied for the box plot."

    working = data[[group_col] + valid_items].copy()
    for column in valid_items:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    long_df = working.melt(
        id_vars=[group_col],
        value_vars=valid_items,
        var_name="Measure",
        value_name="Score",
    ).dropna(subset=[group_col, "Score"])

    if long_df.empty:
        return None, None, "The selected math-category columns contain no numeric scores."

    groups = list(pd.unique(long_df[group_col]))
    measures = valid_items
    n_groups = len(groups)
    n_measures = len(measures)
    width = min(0.75 / max(n_groups, 1), 0.28)
    centers = np.arange(n_measures, dtype=float)

    fig_width = max(10.0, min(18.0, 1.55 * n_measures))
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    legend_handles = []
    for group_index, group_value in enumerate(groups):
        offset = (group_index - (n_groups - 1) / 2) * width
        positions = centers + offset
        datasets = []
        for measure in measures:
            values = long_df.loc[
                (long_df[group_col] == group_value)
                & (long_df["Measure"] == measure),
                "Score",
            ].dropna().to_numpy()
            datasets.append(values)

        color = get_color(group_index)
        box = ax.boxplot(
            datasets,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            manage_ticks=False,
        )
        for patch in box["boxes"]:
            if color is not None:
                patch.set_facecolor(color)
            patch.set_alpha(0.65)
        for median in box["medians"]:
            median.set_linewidth(1.8)

        if box["boxes"]:
            legend_handles.append((box["boxes"][0], str(group_value)))

    labels = [_humanize_survey_item(column) for column in measures]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_xlabel("Math category")
    ax.set_ylabel("Score")
    ax.set_title(f"Math Category Score Distributions by {group_col}")
    if legend_handles:
        ax.legend(
            [handle for handle, _ in legend_handles],
            [label for _, label in legend_handles],
            title=group_col,
        )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig, ax, None


def _humanize_survey_item(column_name: str) -> str:
    """Convert survey item column names into readable construct labels."""
    text = str(column_name).strip()
    text = re.sub(
        r"^(q|item|question)\d+[_\-\s]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else str(column_name)


def _make_likert_construct_summary(
    data: pd.DataFrame,
    item_columns: list[str],
    sort_x: str,
    get_color: ColorGetter,
) -> tuple[Optional[Figure], Optional[Axes], Optional[str]]:
    """Rank wide-format numeric Likert items using descriptive mean ratings."""
    if not item_columns:
        return None, None, "No Likert survey item columns were supplied."

    numeric_items = data[item_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    means = numeric_items.mean(axis=0, skipna=True).dropna()

    if means.empty:
        return None, None, "The selected Likert item columns contain no numeric responses."

    summary = pd.DataFrame(
        {
            "Survey Construct": [
                _humanize_survey_item(column)
                for column in means.index
            ],
            "Mean Rating": means.values,
        }
    )

    if sort_x == "ascending":
        summary = summary.sort_values("Mean Rating", ascending=True)
    else:
        summary = summary.sort_values("Mean Rating", ascending=False)

    fig_height = max(4.5, 0.65 * len(summary) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_positions = np.arange(len(summary))
    colors = [
        get_color(index)
        for index in range(len(summary))
    ]
    if all(color is None for color in colors):
        colors = None

    bars = ax.barh(
        y_positions,
        summary["Mean Rating"],
        color=colors,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(summary["Survey Construct"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Rating")
    ax.set_ylabel("Survey Construct")
    ax.set_title("Average Rating by Survey Construct")

    observed = numeric_items.stack().dropna()
    if not observed.empty:
        observed_min = float(observed.min())
        observed_max = float(observed.max())
        lower = 0 if observed_min <= 1 else max(0, observed_min - 1)
        upper = observed_max + 0.5
        ax.set_xlim(lower, upper)

    for bar, value in zip(bars, summary["Mean Rating"]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}",
            va="center",
            ha="left",
            fontsize=9,
        )

    fig.tight_layout()
    return fig, ax, None

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
    """Convert a normalized chart specification into a Matplotlib figure."""
    normalized_spec, normalization_error = _normalize_chart_spec(
        df,
        spec,
        ui_filters=ui_filters,
    )
    if normalization_error:
        return None, normalization_error

    chart_type = normalized_spec["chart_type_internal"]
    special_mode = normalized_spec.get("special_mode")
    x_col = normalized_spec.get("x")
    y_col = normalized_spec.get("y")
    group_col = normalized_spec.get("group")
    row_col = normalized_spec.get("row")
    col_col = normalized_spec.get("col")
    aggregation = normalized_spec.get("aggregation", "mean")
    sort_x = normalized_spec.get("sort_x", "none")
    use_agg = "mean" if aggregation == "none" else aggregation

    palette_name = color_palette or "Default"
    get_color = _make_color_getter(palette_name)
    data = apply_spec(df, normalized_spec, ui_filters=ui_filters)

    if special_mode not in {
        "likert_construct_summary",
        "multi_measure_box",
        "multi_measure_bar",
    }:
        columns_to_validate = [x_col, y_col, group_col, row_col, col_col]
        missing = [
            column
            for column in columns_to_validate
            if column is not None and column not in data.columns
        ]
        if missing:
            return None, f"Spec referenced missing columns: {missing}"

    fig: Optional[Figure] = None
    ax: Optional[Axes] = None
    builder_error: Optional[str] = None

    if special_mode == "likert_construct_summary":
        fig, ax, builder_error = _make_likert_construct_summary(
            data=data,
            item_columns=normalized_spec.get("item_columns", []),
            sort_x=sort_x,
            get_color=get_color,
        )

    elif chart_type in ["Bar", "Stacked Bar", "Line"]:
        if x_col is None:
            return None, f"{chart_type} requires an x-axis column."

        if chart_type == "Bar" and special_mode == "multi_measure_bar":
            fig, ax, builder_error = _make_multi_measure_bar_chart(
                data=data,
                group_col=x_col,
                item_columns=normalized_spec.get("item_columns", []),
                get_color=get_color,
            )
        elif chart_type == "Stacked Bar" and special_mode == "wytopp_stacked":
            fig, ax, builder_error = _make_wytopp_stacked_chart(
                data=data,
                x_col=x_col,
                aggregation=use_agg,
                sort_x=sort_x,
                get_color=get_color,
            )
        else:
            if y_col is None:
                return None, f"{chart_type} requires a numeric y-axis column."

            group_columns = [x_col] + ([group_col] if group_col else [])
            grouped = _aggregate_chart_data(
                data=data,
                group_columns=group_columns,
                y_col=y_col,
                aggregation=use_agg,
            )
            grouped = _sort_chart_data(grouped, x_col, sort_x)

            if chart_type == "Bar":
                fig, ax = _make_bar_chart(
                    grouped=grouped,
                    x_col=x_col,
                    y_col=y_col,
                    group_col=group_col,
                    aggregation=use_agg,
                    get_color=get_color,
                )
            elif chart_type == "Stacked Bar":
                fig, ax, builder_error = _make_stacked_bar_chart(
                    grouped=grouped,
                    x_col=x_col,
                    y_col=y_col,
                    group_col=group_col,
                    aggregation=use_agg,
                    get_color=get_color,
                )
            else:
                fig, ax = _make_line_chart(
                    grouped=grouped,
                    x_col=x_col,
                    y_col=y_col,
                    group_col=group_col,
                    aggregation=use_agg,
                    get_color=get_color,
                )

    elif chart_type == "Scatter":
        fig, ax, builder_error = _make_scatter_plot(
            data=data,
            x_col=x_col,
            y_col=y_col,
            group_col=group_col,
            get_color=get_color,
            special_mode=special_mode,
            label_col=normalized_spec.get("label"),
        )

    elif chart_type == "Heatmap":
        fig, ax, builder_error = _make_heatmap(
            data=data,
            row_col=row_col,
            col_col=col_col,
            y_col=y_col,
            aggregation=use_agg,
            palette_name=palette_name,
            x_col=x_col,
            group_col=group_col,
        )

    elif chart_type == "Box":
        if special_mode == "multi_measure_box":
            fig, ax, builder_error = _make_multi_measure_box_plot(
                data=data,
                group_col=x_col,
                item_columns=normalized_spec.get("item_columns", []),
                get_color=get_color,
            )
        else:
            fig, ax, builder_error = _make_box_plot(
                data=data,
                x_col=x_col,
                y_col=y_col,
            )

    if builder_error:
        return None, builder_error
    if fig is None or ax is None:
        return None, "VizCreate could not build the requested chart."

    # Post-processing remains centralized for Step 5D.
    try:
        if title_override:
            ax.set_title(title_override)
        if x_label_override:
            ax.set_xlabel(x_label_override)
        if y_label_override:
            ax.set_ylabel(y_label_override)
    except Exception:
        pass

    try:
        if show_value_labels:
            if chart_type in ["Bar", "Stacked Bar"]:
                for patch in ax.patches:
                    height = patch.get_height()
                    if np.isnan(height):
                        continue
                    x_position = patch.get_x() + patch.get_width() / 2
                    y_position = patch.get_y() + height
                    ax.text(
                        x_position,
                        y_position,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            elif chart_type == "Line":
                for line in ax.get_lines():
                    for x_value, y_value in zip(line.get_xdata(), line.get_ydata()):
                        if np.isnan(y_value):
                            continue
                        ax.text(
                            x_value,
                            y_value,
                            f"{y_value:.1f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                        )
    except Exception:
        pass

    try:
        if n_col_name and n_col_name in df.columns:
            if chart_type in ["Bar", "Stacked Bar", "Line"] and x_col is not None:
                n_by_x = data.groupby(x_col)[n_col_name].sum().to_dict()
                new_labels = []
                for label in ax.get_xticklabels():
                    text = label.get_text()
                    n_value = n_by_x.get(text)
                    if n_value is not None:
                        new_labels.append(f"{text}\nN={int(n_value)}")
                    else:
                        new_labels.append(text)
                ax.set_xticklabels(new_labels, rotation=45, ha="right")
    except Exception:
        pass

    return fig, "OK"
