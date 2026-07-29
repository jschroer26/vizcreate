# Phase 2F.2 — Longitudinal Scope Execution

## Problem

Phase 2F correctly recognized prompts such as “fourth grade,” but Step 6 created
Quick Filter controls with every grade selected. Because manual UI filters take
precedence over planner filters, the all-grade selection overrode the inferred
fourth-grade constraint.

## Correction

Grade and year controls now initialize from the prompt-derived chart filters.

Examples:

- “Show fourth-grade mathematics” initializes the grade control with only Grade 4.
- “Show the most recent year” initializes the year control with the latest available year.
- “Show the last three years” initializes the year control with the three most recent dataset years.

When a new prompt changes the analytical scope, the corresponding widget state
is refreshed. Ordinary Streamlit reruns do not erase a user’s subsequent manual
changes.

## Relative year intelligence

Phase 2F now recognizes:

- most recent year
- latest year
- current year
- last N years
- previous N years
- past N years

The requested period is resolved deterministically from actual values in the
dataset. No year is invented.

## Scope precedence

The prompt-derived scope establishes the initial Step 6 controls. A user can
still change those controls manually, and that deliberate selection then
overrides the prompt scope.
