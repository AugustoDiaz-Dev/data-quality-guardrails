from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

import pandas as pd
from langgraph.graph import END, StateGraph

try:
    from backend.utils.stats_utils import (
        detect_drift,
        detect_schema_violations,
        infer_column_type,
        profile_dataframe,
        recommend_fixes,
        summarize_report,
    )
except ImportError:
    from utils.stats_utils import (
        detect_drift,
        detect_schema_violations,
        infer_column_type,
        profile_dataframe,
        recommend_fixes,
        summarize_report,
    )


class PipelineState(TypedDict, total=False):
    dataset: pd.DataFrame
    baseline: Optional[pd.DataFrame]
    report: Dict[str, Any]


def profile_node(state: PipelineState) -> PipelineState:
    profile = profile_dataframe(state["dataset"])
    report = state.get("report", {})
    report["profile"] = profile
    return {"report": report}


def schema_node(state: PipelineState) -> PipelineState:
    df = state["dataset"]
    inferred = {col: infer_column_type(df[col])[0] for col in df.columns}
    violations = detect_schema_violations(df, inferred)
    report = state.get("report", {})
    report["schema"] = {"inferred": inferred, "violations": violations}
    return {"report": report}


def drift_node(state: PipelineState) -> PipelineState:
    report = state.get("report", {})
    baseline = state.get("baseline")
    if baseline is not None:
        report["drift"] = detect_drift(state["dataset"], baseline)
    else:
        report["drift"] = None
    return {"report": report}


def fix_node(state: PipelineState) -> PipelineState:
    report = state.get("report", {})
    recommendations = recommend_fixes(report.get("profile", {}), report.get("schema", {}))
    report["recommendations"] = recommendations
    report["summary"] = summarize_report(report)
    return {"report": report}


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("profile", profile_node)
    graph.add_node("schema", schema_node)
    graph.add_node("drift", drift_node)
    graph.add_node("fix", fix_node)

    graph.set_entry_point("profile")
    graph.add_edge("profile", "schema")
    graph.add_edge("schema", "drift")
    graph.add_edge("drift", "fix")
    graph.add_edge("fix", END)
    return graph.compile()
