# Phase 2H — Intelligent Dataset Recognition

## Purpose

VizCreate now accepts authentic Wyoming assessment exports without requiring users to rename spreadsheet columns.

## Supported official export families

1. **WYTOPP longitudinal proficiency summaries**
   - Recognizes `Specific Grade` as `Grade`.
   - Recognizes `No. of Students Tested` as `Number of Students Tested`.
   - Recognizes ampersand and `and` variants in proficiency headings.
   - Detects banded/suppressed student-count ranges such as `290 - 299` and does not treat them as exact counts.

2. **ACT longitudinal summaries**
   - Recognizes ACT through `Test Type`, school year, ACT subject domains, average score, number tested, and standard-deviation columns.
   - Defaults to a composite-score trend when available.
   - Recommends trend and subject-comparison charts rather than WYTOPP proficiency stacking.

## Architecture

`core/schema_normalizer.py` runs immediately after file loading and before profile detection. It:

- collapses line breaks, tabs, and repeated spaces in headers;
- applies conservative semantic aliases;
- classifies common value patterns;
- records renaming, value types, and warnings in `DataFrame.attrs["schema_normalization"]`.

## Files added or changed

- **New:** `core/schema_normalizer.py`
- **New:** `profiles/act_longitudinal.py`
- **Modified:** `core/data_loader.py`
- **Modified:** `core/profile_detector.py`
- **Modified:** `profiles/__init__.py`
- **New:** `tests/test_phase2h_intelligent_dataset_recognition.py`

## Design principle

Normalization is conservative. Unknown columns remain available under cleaned versions of their original names. VizCreate does not silently alter data values.
