"""Session-state utilities for starting and resetting VizCreate analyses."""

from __future__ import annotations

import hashlib
from typing import Iterable

import streamlit as st


# Keys that belong to one analysis and should not leak into the next dataset.
ANALYSIS_STATE_KEYS = {
    "agent_spec",
    "prompt",
    "prompt_input",
    "chart_type",
    "title_override",
    "x_label_override",
    "y_label_override",
    "show_value_labels",
    "n_col_name_input",
    "grade_filter",
    "year_filter",
}


def initialize_session_state() -> None:
    """Create the small set of session values VizCreate expects."""
    st.session_state.setdefault("prompt", "")
    st.session_state.setdefault("prompt_input", "")
    st.session_state.setdefault("chart_type", "Bar")
    st.session_state.setdefault("agent_spec", None)
    st.session_state.setdefault("uploaded_file_fingerprint", None)
    st.session_state.setdefault("uploader_version", 0)


def reset_analysis_state(*, preserve_palette: bool = True) -> None:
    """Clear prompt, chart, filters, and display overrides for one analysis."""
    keys_to_clear = set(ANALYSIS_STATE_KEYS)
    if not preserve_palette:
        keys_to_clear.add("color_scheme")

    for key in keys_to_clear:
        st.session_state.pop(key, None)

    # Restore predictable defaults after clearing widget-backed values.
    st.session_state["prompt"] = ""
    st.session_state["prompt_input"] = ""
    st.session_state["chart_type"] = "Bar"
    st.session_state["agent_spec"] = None


def start_new_analysis() -> None:
    """Clear the current analysis and reset the file uploader."""
    reset_analysis_state(preserve_palette=True)
    st.session_state["uploaded_file_fingerprint"] = None
    st.session_state["uploader_version"] = (
        int(st.session_state.get("uploader_version", 0)) + 1
    )


def uploaded_file_fingerprint(uploaded_file) -> str:
    """Return a stable fingerprint using filename, size, and file bytes."""
    file_bytes = uploaded_file.getvalue()
    digest = hashlib.sha256(file_bytes).hexdigest()
    return f"{uploaded_file.name}:{len(file_bytes)}:{digest}"


def dataset_changed(uploaded_file) -> bool:
    """Record the uploaded file and report whether it differs from the prior file."""
    current = uploaded_file_fingerprint(uploaded_file)
    previous = st.session_state.get("uploaded_file_fingerprint")

    if previous is None:
        st.session_state["uploaded_file_fingerprint"] = current
        return False

    if current != previous:
        st.session_state["uploaded_file_fingerprint"] = current
        return True

    return False
