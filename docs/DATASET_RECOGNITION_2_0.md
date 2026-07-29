# Dataset Recognition 2.0

This checkpoint improves the ontology and ranking of uploaded datasets
without yet changing the LLM prompt-planning behavior.

## New and improved profiles

### Student-Level Assessment Data
Recognizes one-row-per-student files with grade, score, and subgroup fields.
It explicitly distinguishes cross-sectional assessment from progress monitoring.

### CBM / Progress-Monitoring Data
Now requires meaningful repeated-measure or aggregated screening-window evidence.
A unique student ID plus one score is no longer enough for a strong CBM match.

### Likert / Survey Response Data
Now recognizes:
- Text-coded Likert responses
- Numeric-coded 1–4, 1–5, 1–6, 1–7, and similar bounded scales
- Wide and long survey formats
- Q1/Q2/Item naming patterns
- Respondent and grouping fields
- Likely response scales and observed values
- Human-readable construct names inferred from column headings

## Conditional suggested questions

CBM goal questions appear only when goal fields exist.
Student assessment questions focus on distributions and comparisons.
Survey questions reflect detected constructs and grouping fields.
