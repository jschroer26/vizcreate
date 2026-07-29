# Phase 1.5, Step 1 — Clean Analysis Sessions

This checkpoint adds a session manager without changing chart-generation logic.

## New behavior

- A **New Analysis** button clears the current file, prompt, agent spec, quick filters,
  titles, labels, and value-label settings.
- Uploading a different file automatically clears residual analysis state.
- The selected color palette is preserved across analyses.
- The file uploader uses a versioned key so it can be reset without refreshing the browser.

## Test sequence

1. Upload the WYTOPP multi-year file and generate a line chart.
2. Change title, grade filter, year filter, and palette.
3. Upload a different dataset. Confirm prompt, chart, filters, title, and labels reset.
4. Confirm the selected palette remains.
5. Click **New Analysis**. Confirm the uploader and analysis are cleared.
6. Upload the original file again and verify a clean start.
