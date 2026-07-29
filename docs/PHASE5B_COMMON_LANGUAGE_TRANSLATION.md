# Phase 5B — Common Language Translation

Phase 5B strengthens the translation layer without turning VizCreate into a
report-writing application.

Each completed analysis now receives three grounded communication sections:

1. What the data shows
2. What this may mean
3. What this does not show

The translation is generated once and stored with the completed analysis so
Streamlit reruns do not repeatedly call the model. The prompt prohibits invented
causes, recommendations, interventions, causal claims, and unsupported context.

If the language-model call fails, VizCreate uses a deterministic translation
built from the Evidence Engine output.

The main `Plan and Create My Visualization` button is now full-width and uses a
high-contrast gold treatment so the principal action remains easy to locate as
the page grows.
