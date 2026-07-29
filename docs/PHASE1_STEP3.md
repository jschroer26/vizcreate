# Phase 1, Step 3 — Data loading module

This step moves CSV/Excel reading and column-header cleanup out of
`vizcreate_app.py` and into `core/data_loader.py`.

## New module

- `load_dataframe(uploaded_file)` reads `.csv`, `.xlsx`, and `.xls` files.
- `clean_column_names(df)` collapses newlines, tabs, and repeated spaces in headers.
- Read failures and unsupported file types are converted into concise `ValueError`
  messages that the Streamlit interface displays to the user.

## Behavior preserved

The dataframe contents are unchanged. The main app still previews the uploaded
file, detects the enrollment column, identifies the dataset family, and runs the
same chart and prompt workflows.

## Test

```powershell
streamlit run vizcreate_app.py
```

Confirm that both CSV and Excel uploads work and that the four established demo
prompts still generate the expected charts.
