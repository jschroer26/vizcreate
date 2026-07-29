# Phase 2F — Intent-Aware Dataset Filtering

## Purpose

Ensure that narrow educational questions are analyzed on the exact subset of
data requested by the user.

Example:

> As a teacher, look only at third-grade math.

VizCreate now resolves that request against actual dataset columns and values,
adds the constraints to the chart specification, and filters the data before:

- visualization rendering
- Insights
- VizCreate Insights
- Dive Deeper
- Data Analyst Coach continuation
- Investigation Summary

## Deterministic scope resolution

Phase 2F recognizes:

- grade expressions such as Grade 3, third grade, and 3rd graders
- subject aliases such as math, mathematics, ELA, reading, and science
- exact low-cardinality dataset values such as a named school, subgroup, or year

The system never invents a filter value. It matches the prompt only to values
that actually occur in the uploaded dataset.

## Safeguards

- Existing valid Planner filters are preserved.
- Inferred constraints are added only when an actual dataset value can be found.
- A combined scope that produces zero rows is rejected rather than silently
  generating an empty chart.
- The interface displays the applied scope and the number of included rows.
- The planner prompt now explicitly treats narrow scope language as executable
  constraints rather than optional chart dimensions.

## Downstream benefit

Because all downstream systems receive the correctly filtered dataset, Phase 2F
also improves the accuracy of evidence statements and the Phase 4 Investigation
Summary.
