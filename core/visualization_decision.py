"""Phase 2G: determine whether a visualization directly answers the question."""

from __future__ import annotations
from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class AnalyticalObjective:
    objective_id: str
    display_name: str
    statistic: str | None = None
    comparison_mode: str | None = None
    preferred_charts: list[str] = field(default_factory=list)
    acceptable_charts: list[str] = field(default_factory=list)
    exploratory_charts: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


def detect_analytical_objective(prompt: str, intent_id: str = "") -> AnalyticalObjective:
    text = str(prompt or "").lower()
    statistic = None
    if re.search(r"\bmedian\b", text): statistic="median"
    elif re.search(r"\bmean\b|\baverage\b", text): statistic="mean"
    elif re.search(r"\bpercent\b|\bpercentage\b|\bproportion\b|\bshare\b", text): statistic="percentage"
    elif re.search(r"\bvariance\b|\bvariability\b|\bspread\b|\bconsistent\b", text): statistic="variability"

    ranking = bool(re.search(r"\bhighest\b|\blowest\b|\bstrongest\b|\bweakest\b|\brank\b|\bmost\b|\bleast\b", text))
    relationship = bool(re.search(r"\brelationship\b|\bcorrelat|\bassociated\b|\bversus\b|\bvs\.?\b", text))
    trend = bool(re.search(r"\btrend\b|\bover time\b|\bgrowth\b|\bchange\b|\bimprov", text))
    outliers = bool(re.search(r"\boutlier|\bunusual\b|\banomal", text))
    distribution = bool(re.search(r"\bdistribution\b|\bspread\b|\bvariability\b", text))

    if statistic=="median" and ranking:
        return AnalyticalObjective("rank_median","Rank groups by median","median","ranking",
            ["box","bar"],[],["scatter"],["The question explicitly requests a median-based ranking."])
    if statistic=="median":
        return AnalyticalObjective("compare_median","Compare group medians","median","comparison",
            ["box","bar"],[],["scatter"],["The question explicitly requests medians."])
    if relationship or intent_id=="relationship":
        return AnalyticalObjective("relationship","Examine a relationship",None,"relationship",
            ["scatter"],["line"],["bar","box"],["The question asks whether two measures are related."])
    if trend or intent_id=="trend":
        return AnalyticalObjective("trend","Examine change over time",None,"trend",
            ["line"],["bar"],["scatter"],["The question asks about ordered change or growth."])
    if outliers:
        return AnalyticalObjective("outliers","Identify unusual observations",None,"outliers",
            ["box","scatter"],[],["bar"],["The question asks about unusual observations."])
    if distribution or intent_id=="distribution":
        return AnalyticalObjective("distribution","Compare distributions",statistic,"distribution",
            ["box"],["scatter"],["bar"],["The question asks about spread or distribution."])
    if ranking or intent_id=="ranking":
        return AnalyticalObjective("ranking","Rank groups",statistic,"ranking",
            ["bar"],["box"],["scatter"],["The question asks for an ordered comparison."])
    if intent_id=="composition":
        return AnalyticalObjective("composition","Compare composition",statistic,"composition",
            ["stacked_bar"],["bar"],["heatmap"],["The question asks about parts of a whole."])
    return AnalyticalObjective("comparison","Compare groups or measures",statistic,"comparison",
        ["bar","box"],["heatmap","line"],["scatter"],["The question asks for a comparative summary."])


def direct_answer_score(objective: AnalyticalObjective, chart_type: str, spec: dict[str,Any] | None=None) -> tuple[float,bool,str]:
    chart=str(chart_type or "").lower()
    spec=spec or {}
    aggregation=str(spec.get("aggregation") or "").lower()
    if chart in objective.preferred_charts:
        score=100.0 if chart==objective.preferred_charts[0] else 95.0
        eligible=True
        reason="Directly answers the stated analytical objective."
    elif chart in objective.acceptable_charts:
        score=72.0
        eligible=True
        reason="Can answer the question, but less directly than the preferred visualization."
    elif chart in objective.exploratory_charts:
        score=25.0
        eligible=False
        reason="Useful for exploration, but does not directly answer the stated question."
    else:
        score=10.0
        eligible=False
        reason="Does not directly display the requested comparison or statistic."

    if objective.statistic=="median":
        if chart=="box":
            score=100.0; eligible=True; reason="Displays group medians directly and preserves score distributions."
        elif chart=="bar" and aggregation=="median":
            score=98.0; eligible=True; reason="Directly ranks the requested group medians."
        elif chart=="bar" and aggregation!="median":
            score=35.0; eligible=False; reason="A bar chart only answers the question if it aggregates by median."
        elif chart=="scatter":
            score=15.0; eligible=False; reason="Shows individual scores but does not directly compare group medians."
    return score,eligible,reason
