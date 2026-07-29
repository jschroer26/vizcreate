# Phase 2G.1 — Visualization Synchronization Fix

This maintenance release corrects two integration defects:

1. `generate_dive_deeper_suggestions()` now publicly accepts and forwards
   `analytical_objective`.
2. Phase 2G's winning chart family now governs later intent-repair logic.
   A median-ranking box plot can no longer be replaced by an older
   strengths/needs scatterplot rule.

The recommended candidate, planner `final_spec`, session-state `agent_spec`,
and rendered visualization are explicitly synchronized after intent-aware
filter enrichment.
