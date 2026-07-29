# Phase 3B.1 — Data Analyst Coach Bridge

## Purpose

Remove the dead end that occurred when deterministic Dive Deeper paths were
exhausted even though the Data Analyst Coach had already identified other
reasonable analyses and questions.

## Revised empty state

> You have explored the strongest guided paths available for this view.
>
> VizCreate’s Data Analyst Coach identified several additional ways to continue.
> These options are broader and may help you approach the evidence from a
> different perspective.

## Behavior

When no guided Dive Deeper cards remain, VizCreate now:

1. Opens a Data Analyst Coach continuation area.
2. Shows non-duplicate Alternative Analyses.
3. Shows non-duplicate Suggested Next Questions.
4. Converts each item into an executable Planner 2.0 prompt.
5. Records the selected Coach path in the Investigation Trail.
6. Filters suggestions already represented by the current prompt or trail.

This preserves a continuous educational inquiry experience and adds the Coach
selections to the history needed for Phase 4 investigation summaries.
