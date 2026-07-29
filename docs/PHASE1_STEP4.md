# Phase 1, Step 4 — Filtering module

Filtering logic has been moved from `vizcreate_app.py` into `core/filters.py`.

## What changed

- The existing `apply_spec()` behavior now lives in the filters module.
- UI quick filters still override LLM/spec filters for the same column.
- Existing missing-column behavior is preserved.
- The main app imports `apply_spec` from `core.filters`.

## Scaffolding added for later phases

The filters module also includes:

- `apply_filters()` — applies spec and UI filters directly.
- `resolve_special_filter_values()` — supports `__LATEST__` and `__EARLIEST__` tokens.
- `detect_filter_dimensions()` — identifies practical filter columns.
- `default_ui_filters()` — builds inclusive default selections.

The new scaffolding is not yet used by the Streamlit quick-filter interface, so this step should not change visible behavior.

## Test checklist

1. Upload the same demo dataset used in Step 3.
2. Run a known line-chart prompt.
3. Toggle one and multiple grade levels.
4. Toggle school years.
5. Confirm N values update with the displayed data.
6. Run a heatmap prompt.
7. Run the WYTOPP stacked-bar prompt.
