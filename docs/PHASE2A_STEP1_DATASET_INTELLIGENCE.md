# Phase 2A Step 1 — Dataset Intelligence Profiles

This checkpoint introduces ranked profile detection without changing
the existing chart-generation behavior.

## Profiles included

- WYTOPP Current-Year Assessment
- WYTOPP Longitudinal Assessment
- CBM / Progress-Monitoring Data
- Likert / Survey Response Data
- General Tabular Data

## Interface additions

After upload, VizCreate displays:

- Best-matching dataset profile
- Confidence score
- Detected data structure
- Organizational level when available
- Detected column roles
- Recommended and discouraged charts
- Analytical cautions
- Suggested questions
- Possible alternative profile matches

The existing prompt builder still uses its current behavior. Profile-specific
LLM guidance will be connected in the next checkpoint after detection is tested.
