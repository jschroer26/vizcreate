# Phase 2C — Statistical Intent Recognition

VizCreate now recognizes the analytical purpose of a prompt separately
from the uploaded dataset profile.

## Initial intents

- Relationship
- Trend or change
- Distribution
- Ranking
- Composition
- Group comparison
- Overview or status

## Relationship behavior

When the user names two exact numeric columns and asks whether they are
related, correlated, associated, or predictive:

- VizCreate forces a scatterplot.
- The exact two numeric columns become x and y.
- Aggregation is set to none.
- A least-squares trend line is added when possible.
- Pearson's correlation is displayed.
- Read This Chart describes direction, strength, and the non-causal limitation.

## Defensive heatmap repair

Heatmaps now reject:
- A value column that is also used as a row or column dimension.
- Identical row and column dimensions.

This prevents the pandas duplicate-column reset_index error that exposed
the intent mismatch.
