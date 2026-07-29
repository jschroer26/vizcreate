# Phase 5C - Analysis Report Generator

Phase 5C adds a compact export area for the current VizCreate analysis:

- PNG
- PDF Report
- Word Report

The PDF and Word reports mirror the simple viewer layout rather than exposing planner or developer metadata.

## Report layout

1. VizCreate Analysis Report
2. Question
3. Visualization
4. Analyst Summary
5. Key Findings
6. Supporting Evidence
7. Communication Preview
8. VizCreate / University of Wyoming College of Education footer

## Architecture

`core/report_generator.py` provides:

- `ReportModel`
- `build_analyst_summary()`
- `build_pdf_report()`
- `build_word_report()`

Both document exporters consume the same `ReportModel`, keeping PDF and Word output synchronized.

## Design boundaries

- No planner scores
- No candidate rankings
- No debug JSON
- No metadata appendix
- No causal claims added by the report generator
- Current visualization only

## Dependency note

The PDF exporter uses Matplotlib's built-in PDF backend, so Phase 5C does **not** require ReportLab. The Word exporter uses `python-docx`, which is included in `requirements.txt`.

## Entire Investigation Export

The Export area now includes a simple scope choice:

- **Current View** exports the analysis currently displayed.
- **Entire Investigation** exports all completed analyses stored in the current investigation, in chronological order.

PDF and Word honor the selected scope. PNG intentionally remains a download of the current chart only. The investigation export uses the same concise sections as the viewer for each analysis; no item-removal or reordering interface is added.
