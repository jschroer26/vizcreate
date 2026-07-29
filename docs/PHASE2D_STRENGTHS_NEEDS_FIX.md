# Phase 2D Fix — Strengths, Needs, Unit of Analysis, and Decision Basis

## Problem addressed

A teacher asked which students were doing great and which students needed more
help. The coach recognized the teacher role, but the planner generated a
grade-level box plot. The chart was statistically reasonable as an overview,
but it did not preserve the requested student-level unit.

## New logic

### Strengths and Needs Identification

A new statistical intent detects language such as:

- doing well or doing great;
- needs help or support;
- strengths and weaknesses;
- areas of concern;
- intervention or enrichment;
- struggling or at risk;
- where should we focus?

### Unit of Analysis

VizCreate now distinguishes among:

- student;
- skill or subscore;
- grade;
- school;
- subject;
- subgroup;
- survey item;
- survey construct;
- time period.

An explicitly named unit governs the first analysis. Therefore, “which
students” cannot be replaced with an aggregated grade-level first chart.

### Decision Basis

VizCreate labels the evidence standard as:

- goal-referenced;
- criterion-referenced;
- trend-referenced;
- peer-relative;
- distribution-relative;
- descriptive only.

When no benchmark or cut score exists, the app uses comparative language and
warns that the result is a screening view rather than a stand-alone
intervention decision.

## Profile-aware behavior

- Student assessment: student support map or ranked student chart.
- CBM: student progress, goal discrepancy, or growth.
- WYTOPP: priority grades and subjects using proficiency and trend evidence.
- Likert survey: construct/item ranking plus disagreement and polarization.
- Generic data: cautious comparative analysis until direction and criteria are known.

## Student support map

When a student-level assessment dataset contains a student identifier and at
least two numeric assessment measures, the deterministic repair layer creates:

- a scatterplot using the two measures;
- one point per student;
- grade grouping when available;
- student labels;
- median reference lines;
- comparative rather than diagnostic language.

This repair is applied even when the LLM initially proposes an aggregated box
plot.
