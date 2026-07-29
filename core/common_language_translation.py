"""Phase 5B: grounded Common Language Translation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
import json
import re


@dataclass
class TranslationResult:
    what_the_data_shows: str
    what_this_may_mean: str
    what_this_does_not_show: str
    combined_text: str
    source: str = "deterministic"


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def deterministic_translation(
    *,
    question: str,
    scope_text: str,
    insights: str,
    evidence_overview: str,
    evidence_findings: list[str],
) -> TranslationResult:
    findings = [_clean(item) for item in evidence_findings if _clean(item)]
    overview = _clean(evidence_overview)
    insights = _clean(insights)
    question = _clean(question)
    scope_text = _clean(scope_text)

    lead = findings[0] if findings else overview or insights
    if not lead:
        lead = "The current analysis did not identify a sufficiently distinct pattern."

    shows = lead
    if scope_text:
        shows += f" This finding applies to the selected scope: {scope_text}."

    if len(findings) > 1:
        may_mean = (
            f"A second pattern worth considering is that {findings[1][0].lower() + findings[1][1:]}. "
            "Together, these results identify areas that may deserve closer discussion or follow-up."
        )
    else:
        may_mean = (
            "This result identifies a relative pattern in the selected data and may help focus "
            "professional discussion or a more detailed follow-up analysis."
        )

    does_not = (
        "The analysis does not explain why the pattern occurred, establish causation, "
        "or determine by itself what action should be taken."
    )

    prefix = f"For the question “{question},” " if question else ""
    combined = (
        f"{prefix}{shows} {may_mean} {does_not}"
    )
    return TranslationResult(shows, may_mean, does_not, combined, "deterministic")


def build_translation_prompt(
    *,
    question: str,
    scope_text: str,
    insights: str,
    evidence_overview: str,
    evidence_findings: list[str],
) -> str:
    evidence_lines = "\n".join(f"- {_clean(item)}" for item in evidence_findings if _clean(item))
    return f"""
You are the Common Language Translation layer in VizCreate.

Translate the supplied analytical evidence into clear professional language
that a teacher, administrator, researcher, board member, or community partner
could understand. Preserve the exact meaning of the evidence.

Hard rules:
- Use only the supplied question, scope, insights, and evidence.
- Do not invent causes, recommendations, interventions, or contextual facts.
- Do not imply causation.
- Do not exaggerate strength or certainty.
- Keep each section to 1-3 sentences.
- Avoid technical statistical jargon when ordinary language is accurate.
- Do not use promotional language.
- Return JSON only.

Question:
{_clean(question)}

Applied scope:
{_clean(scope_text) or "No additional scope was recorded."}

Chart insight:
{_clean(insights)}

Evidence overview:
{_clean(evidence_overview)}

Evidence findings:
{evidence_lines or "- No distinct structured finding was recorded."}

Return exactly:
{{
  "what_the_data_shows": "...",
  "what_this_may_mean": "...",
  "what_this_does_not_show": "..."
}}
""".strip()


def parse_translation_response(raw_text: str) -> TranslationResult:
    text = _clean(raw_text)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in translation response.")
    data = json.loads(match.group(0))
    shows = _clean(data.get("what_the_data_shows"))
    may_mean = _clean(data.get("what_this_may_mean"))
    does_not = _clean(data.get("what_this_does_not_show"))
    if not all([shows, may_mean, does_not]):
        raise ValueError("Translation response omitted a required section.")
    return TranslationResult(
        what_the_data_shows=shows,
        what_this_may_mean=may_mean,
        what_this_does_not_show=does_not,
        combined_text=f"{shows} {may_mean} {does_not}",
        source="llm_grounded",
    )


def generate_translation(
    client: Any,
    *,
    question: str,
    scope_text: str,
    insights: str,
    evidence_overview: str,
    evidence_findings: list[str],
    model: str = "gpt-4.1-mini",
) -> TranslationResult:
    fallback = deterministic_translation(
        question=question,
        scope_text=scope_text,
        insights=insights,
        evidence_overview=evidence_overview,
        evidence_findings=evidence_findings,
    )
    try:
        response = client.responses.create(
            model=model,
            input=build_translation_prompt(
                question=question,
                scope_text=scope_text,
                insights=insights,
                evidence_overview=evidence_overview,
                evidence_findings=evidence_findings,
            ),
        )
        raw = getattr(response, "output_text", None)
        if not raw:
            pieces = []
            for output in getattr(response, "output", []) or []:
                for block in getattr(output, "content", []) or []:
                    block_text = getattr(block, "text", None)
                    if block_text:
                        pieces.append(block_text)
            raw = "".join(pieces)
        return parse_translation_response(raw or "")
    except Exception:
        return fallback
