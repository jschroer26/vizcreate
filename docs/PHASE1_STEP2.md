# Phase 1, Step 2 — Extract color palettes

This step moves the chart palette configuration out of `vizcreate_app.py` and into `config/palettes.py`.

## Changes

- Added `PALETTES` to `config/palettes.py`.
- Added `palette_names()` so the Streamlit selector is populated from the same source.
- Replaced the palette dictionary in `vizcreate_app.py` with:

```python
from config.palettes import PALETTES, palette_names
```

- No plotting behavior or palette values were changed.

## Test

From the project folder:

```powershell
streamlit run vizcreate_app.py
```

Upload the normal demo file, generate a chart, and switch among at least three color schemes.
