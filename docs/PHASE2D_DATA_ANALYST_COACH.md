# Phase 2D — Data Analyst Coach: Role-Aware Planning

## Was role distinction already present?

Only in a limited way. Earlier versions recognized phrases such as
"if you were the principal" as broad-prompt triggers, but VizCreate did not
maintain a formal user role or systematically adapt recommendations to that role.

## Added in this checkpoint

VizCreate now supports these perspectives:

- Auto-detect from prompt
- Classroom Teacher
- Instructional Coach
- School Leader / Principal
- District Leader / Superintendent
- Assessment / Data Coordinator
- School Board / Governance
- Researcher / Faculty

The role is either explicitly selected or inferred from the prompt.

## Data Analyst Coach plan

Before the chart is generated, VizCreate now displays:

- the recommended first analysis;
- why it comes first for the selected role;
- what the analysis can answer;
- what it cannot establish;
- alternative analyses ranked by priority;
- suggested next questions;
- a caution when the dataset is too small or lacks a required time field.

The role, dataset profile, statistical intent, and coach plan are also passed
to the LLM so that the chart specification and notes use the same reasoning.

## Important design rule

Role awareness changes the decision framing, not the statistical facts.
A teacher, principal, superintendent, board member, and researcher may receive
different recommendations and explanations, but VizCreate must not alter the
meaning or limitations of the evidence.
