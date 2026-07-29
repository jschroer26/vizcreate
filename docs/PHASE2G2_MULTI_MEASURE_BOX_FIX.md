# Phase 2G.2 — Multi-Measure Box Plot Fix

A prompt such as “Compare 4th and 5th grade math scores for all Math
categories” produces a wide-format specification: Grade Level is the
comparison group, while several math-category columns are stored in
`item_columns`.

The chart engine now recognizes this structure and reshapes it internally
from wide to long form. It renders one cluster per math category and one box
per selected grade, with Grade Level shown in the legend.

This allows a valid multi-category distribution comparison without inventing
a single `y` field or discarding the requested categories.
