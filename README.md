# VizCreate

VizCreate is an educational data analysis assistant built with Streamlit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run vizcreate_app.py
```

## Repository structure

- `vizcreate_app.py` — Streamlit application entry point
- `core/` — analysis, planning, charting, evidence, communication, and reporting logic
- `profiles/` — dataset profile definitions
- `config/` — application configuration and palettes
- `tests/` — regression and integration tests
- `docs/` — development-phase documentation

## Export options

Phase 5C supports PNG export plus PDF and Word reports for either the current view or the entire investigation.

## Secrets

Do not commit API keys. Configure `OPENAI_API_KEY` through Streamlit secrets or your local environment.
