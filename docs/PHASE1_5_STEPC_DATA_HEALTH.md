# Phase 1.5C — Dataset Health

This checkpoint adds deterministic health checks before visualization.

## Analysis Session card

The top-right card now includes:

- Dataset family
- File metadata
- Dataset health
- Current analysis
- Active filters

## Detailed health panel

Below the data preview, VizCreate reports:

- Overall readiness
- Passed checks
- Warnings
- Critical issues
- Expandable plain-language details

The health engine checks general structure, missing values, duplicate
rows and headings, numeric usability, and selected profile-specific
conditions. WYTOPP files receive additional checks for required fields,
year coverage, percentage ranges, proficiency totals, and tested counts.

No LLM call is used for health checks.
