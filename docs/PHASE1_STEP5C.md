# Phase 1, Step 5C

The chart engine has been decomposed into focused private builders while preserving the public interface:

```python
from core.chart_engine import make_chart_from_spec
```

New chart builders:

- `_make_bar_chart()`
- `_make_stacked_bar_chart()`
- `_make_line_chart()`
- `_make_wytopp_stacked_chart()`
- `_make_heatmap()`
- `_make_box_plot()`

Shared aggregation, sorting, palette, normalization, label, and N-count behavior remain centralized. Heatmaps now use palette-specific continuous colormaps.
