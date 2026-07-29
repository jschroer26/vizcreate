# Phase 2G — Visualization Decision Engine

Phase 2G asks whether a candidate visualization directly answers the user's
analytical question before generic usefulness is considered.

The shared analytical-objective object records the requested statistic,
comparison mode, preferred chart families, acceptable alternatives, and
exploratory-only charts.

Candidate labels are now:

- Direct Answer
- Statistical Suitability
- Overall Recommendation

Candidates that do not directly answer the question remain available as
exploratory alternatives, but cannot outrank an eligible direct-answer
visualization when one is available.

The analytical objective is also passed to Dive Deeper so follow-up
investigations extend the same line of inquiry.
