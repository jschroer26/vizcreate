# Phase 2B — Profile-Governed Planning

This checkpoint expands VizCreate from dataset recognition to executable,
profile-aware interpretation.

## Added capabilities

- The detected dataset profile is now passed into the LLM prompt.
- Detected roles and construct names are explicitly identified as metadata,
  not dataframe columns.
- Wide-format numeric Likert surveys support a new special mode:
  `likert_construct_summary`.
- The special mode carries exact `item_columns` and reshapes the wide survey
  internally before plotting.
- Invented conceptual fields such as `survey_construct` and `rating` are
  automatically repaired for recognized wide-format Likert questions.
- The chart engine ranks constructs using descriptive mean ratings and includes
  an ordinal-data caution in Read This Chart.
- Student-level and CBM prompt guidance now blocks unsupported progress claims.

## Test prompt

`Which survey construct received the strongest ratings?`

Expected behavior:
- Bar chart
- Exact survey item columns
- Human-readable construct labels
- Descending mean-rating order
- No missing-column error
