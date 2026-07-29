"""Color palette definitions used by VizCreate charts."""

from typing import Optional


PALETTES: dict[str, Optional[list[str]]] = {
    "Default": None,  # Use Matplotlib's default color cycle.
    "Greyscale": ["#000000", "#555555", "#888888", "#BBBBBB", "#DDDDDD"],
    "UW Brown & Gold": ["#3B2314", "#FFC72C", "#6B4C3B", "#FFB81C"],
    "Blue-Orange": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
    "Colorblind-friendly": ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#56B4E9"],
    "Muted": ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"],
}


HEATMAP_CMAPS: dict[str, str] = {
    "Default": "viridis",
    "Greyscale": "Greys",
    "UW Brown & Gold": "YlOrBr",
    "Blue-Orange": "coolwarm",
    "Colorblind-friendly": "cividis",
    "Muted": "PuBuGn",
}


def palette_names() -> list[str]:
    """Return palette names in the order used by the Streamlit selector."""
    return list(PALETTES.keys())


def get_heatmap_cmap(palette_name: str) -> str:
    """Return the continuous Matplotlib colormap associated with a palette."""
    return HEATMAP_CMAPS.get(palette_name, "viridis")
