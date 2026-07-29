"""Streamlit display helpers for Dataset Intelligence."""

from __future__ import annotations

import streamlit as st

from profiles.base_profile import DatasetProfileResult


def render_dataset_intelligence(
    profile: DatasetProfileResult,
    alternatives: list[DatasetProfileResult] | None = None,
) -> None:
    """Display profile, structure, roles, recommendations, and cautions."""
    alternatives = alternatives or []

    st.markdown("#### 🧠 Dataset Intelligence")

    top1, top2, top3 = st.columns(3)
    top1.metric("Dataset profile", profile.display_name)
    top2.metric("Confidence", f"{profile.confidence_percent}%")
    top3.metric("Structure", profile.structure)

    st.caption(profile.description)

    if profile.organizational_level:
        st.markdown(
            f"**Organizational level:** {profile.organizational_level}"
        )

    with st.expander("View detected structure and recommendations"):
        if profile.detected_roles:
            st.markdown("**Detected column roles**")
            for role, column in profile.detected_roles.items():
                readable_role = role.replace("_", " ").title()
                st.markdown(f"- **{readable_role}:** {column}")

        if profile.recommended_charts:
            st.markdown("**Recommended views**")
            st.write(" · ".join(profile.recommended_charts))

        if profile.discouraged_charts:
            st.markdown("**Use cautiously or avoid**")
            for item in profile.discouraged_charts:
                st.markdown(f"- {item}")

        if profile.cautions:
            st.markdown("**Analytical cautions**")
            for item in profile.cautions:
                st.markdown(f"- {item}")

        if profile.evidence:
            st.markdown("**Why VizCreate selected this profile**")
            for item in profile.evidence:
                st.markdown(f"- {item}")

        meaningful_alternatives = [
            result
            for result in alternatives
            if result.profile_id != profile.profile_id
            and result.confidence >= 0.30
        ][:2]

        if meaningful_alternatives:
            st.markdown("**Possible alternative matches**")
            for result in meaningful_alternatives:
                st.markdown(
                    f"- {result.display_name}: {result.confidence_percent}%"
                )

    if profile.suggested_questions:
        st.markdown("**Suggested questions for this dataset**")
        for question in profile.suggested_questions:
            st.markdown(f"- {question}")
