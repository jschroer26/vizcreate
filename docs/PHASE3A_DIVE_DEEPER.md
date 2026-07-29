# Phase 3A — Dive Deeper Investigation Engine

## Purpose

Dive Deeper turns a completed visualization into the beginning of a guided
educational data investigation.

Rather than recommending another chart, VizCreate recommends the next
investigation. When the user selects one, the recommended question becomes the
next prompt and returns to Planner 2.0 for interpretation and visualization
selection.

## Inputs

The engine evaluates:

- dataset profile;
- current statistical intent;
- user role;
- requested unit of analysis;
- decision basis;
- selected chart and special visualization mode;
- available fields;
- investigations already completed in the current session.

## Outputs

Each recommendation contains:

- investigation title;
- educational rationale;
- executable natural-language prompt;
- educational-value score;
- likely form of visual evidence;
- explanation of why it is useful in the current context.

The value score is used for ranking but is translated into user-friendly
labels rather than displayed numerically.

## Profile-aware libraries

### Student assessment and CBM

Possible paths include:

- investigate skills and subscores;
- compare student profiles;
- examine growth;
- compare grades;
- compare subgroups;
- find unusual profiles.

### WYTOPP

Possible paths include:

- find persistent strengths and needs;
- compare subjects;
- compare grades;
- compare schools;
- examine subgroup patterns.

### Likert surveys

Possible paths include:

- rank survey strengths and needs;
- check polarization;
- inspect neutral and missing responses;
- compare respondent groups.

## Role safeguards

The engine filters recommendations by role. For example, individual-student
investigations are not offered to board members, while school, subject,
longitudinal, and subgroup investigations receive higher priority.

## Investigation state

A lightweight session-state object records:

- selected Dive Deeper steps;
- visualizations seen;
- units examined;
- intents examined.

This prevents immediate loops and supports an investigation trail without
creating persistent user memory.

## User experience

After Read This Chart, users see a compact Dive Deeper section. Selecting an
investigation:

1. records the step in the current trail;
2. places the recommended question into the prompt box;
3. returns the question to Planner 2.0;
4. creates the next visualization.

Users may open the Investigation Trail or reset it at any time.
