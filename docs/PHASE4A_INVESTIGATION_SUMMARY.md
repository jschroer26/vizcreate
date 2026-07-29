# Phase 4A — Investigation Summary

Phase 4A adds deterministic synthesis for either the current visualization or
the entire Investigation Trail.

## Controls

- Current visualization / Entire investigation
- Concise / Detailed

## Output

- narrative synthesis
- patterns across the investigation
- important differences or view-specific findings
- limitations
- recommended next step
- confidence label
- number of visualizations reviewed

## Convergence rule

A repeated finding increases confidence only when it occurs across distinct
prompts or visualization types. Repetition within essentially the same analysis
is not treated as independent confirmation.

## Reset

Start a New Investigation clears the inquiry history, current planner result,
visualization specification, and summary state while preserving the uploaded
dataset.


## 4A.1 loading fix

The summary scope and detail controls now use explicit defaults and defensive
fallbacks. This prevents `NoneType.lower()` errors on first render or in
Streamlit versions that initially return `None` from segmented controls.
