# Phase 2G.3 Revised — Subject and Measure Hierarchy Detection

This release replaces the earlier Math-specific measure-scope rule with a
generic subject hierarchy engine.

VizCreate now detects:

- Math
- ELA/Reading
- Science
- future subject families that follow recognizable column naming patterns

Within each subject family it separates:

- overall measures
- category, domain, strand, standard-area, component, reasoning-skill, and
  subscore measures

Prompt interpretation distinguishes:

- overall score
- categories/subscores/domains
- all measures
- unspecified subject measure

For multi-category requests, VizCreate excludes the broad overall score,
preserves the requested comparison group and filters, and creates a grouped
`multi_measure_bar` specification.

The implementation uses:

1. subject aliases,
2. normalized prefixes and delimiters,
3. overall-score terminology,
4. numeric-column validation,
5. profile-role metadata where available, and
6. exact original dataset column names in the final chart specification.

A `measure_scope_metadata` block records the interpreted subject, requested
scope, selected columns, excluded overall columns, detected category columns,
and confidence.
