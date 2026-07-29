# Phase 1, Step 5A — Relocate the Chart Engine

This checkpoint moves the existing `make_chart_from_spec()` function from
`vizcreate_app.py` into `core/chart_engine.py`.

## What changed

- The complete chart-generation function was moved unchanged.
- `core/chart_engine.py` imports palettes and filtering dependencies directly.
- `vizcreate_app.py` now imports `make_chart_from_spec` from the chart engine.
- The manual chart controls remain in the Streamlit app for now.

## What did not change

- Chart behavior
- Supported chart types
- WYTOPP stacked mode
- Quick-filter behavior
- Color palettes
- Value labels and N counts
- Downloads and manual controls

## Test checklist

1. Line chart by grade over time
2. Heatmap by grade and school year
3. Grouped bar for the latest year
4. WYTOPP stacked bar
5. Open-ended principal prompt
6. Grade and year quick filters
7. Color palette, labels, N counts, and PNG download
