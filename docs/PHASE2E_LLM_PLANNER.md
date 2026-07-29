# Phase 2E — LLM Planner 2.0

## Purpose

The LLM no longer jumps directly from a prompt to one chart specification.
It first interprets the educational decision, generates multiple executable
analysis candidates, ranks them, and recommends one.

## User experience

The user still clicks one button:

**Plan and create my visualization**

The internal planning stages remain hidden unless the user opens the
“How VizCreate understood and planned this analysis” expander.

## Planner output

The planner returns:

- the educational decision;
- a plain-language interpretation of the request;
- genuine ambiguities;
- necessary assumptions;
- two or three candidate visual analyses;
- a fit score and limitation for each;
- a recommended candidate;
- recommendation confidence;
- suggested next questions.

## Hybrid ranking

The LLM evaluates semantic and educational fit. A deterministic ranking layer
then checks:

- exact-column validity;
- supported chart types;
- numeric requirements;
- preservation of the requested unit;
- statistical-intent fit;
- profile fit;
- role fit;
- decision-basis language.

The deterministic layer may override the LLM’s preferred candidate when a
different candidate is more executable or better preserves the user’s request.

## Reliability

If Planner 2.0 fails, VizCreate automatically falls back to the existing
single-spec LLM planner and deterministic repair system. The user can still
receive a visualization without managing the failure.
