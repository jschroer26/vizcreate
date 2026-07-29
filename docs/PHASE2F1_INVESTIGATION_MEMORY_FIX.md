# Phase 2F.1 — Durable Investigation Memory

## Problem corrected

The Investigation Summary previously saved evidence only when a user selected a
Dive Deeper or Data Analyst Coach recommendation. A new question typed directly
into the main prompt produced a visualization but did not become a durable
investigation record.

Example:

1. What areas were strongest for Elementary Ed?
2. What areas were strongest for Secondary Ed?

The second visualization replaced the first in the active interface, and the
Entire Investigation summary could not reliably synthesize both analyses.

## New behavior

Every successfully rendered and interpreted analysis is now automatically
recorded after the Evidence Engine runs.

The record includes:

- exact user prompt
- chart type and analytical specification
- applied filter scope
- chart insight
- evidence overview
- structured evidence findings

## Duplicate protection

Streamlit reruns the application when a toggle, expander, or other control
changes. A deterministic fingerprint based on prompt, chart specification, and
filters prevents those reruns from creating duplicate investigation records.

A genuinely new prompt or a changed analytical scope creates a new record.

## Summary behavior

Entire Investigation now reads only completed-analysis records. Navigation
choices such as clicking a Dive Deeper recommendation are retained internally
for path management but are not mistaken for completed evidence.

## Investigation Trail

The trail now displays analyses actually completed, including the applied scope
and evidence overview.
