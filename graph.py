"""LangGraph wiring — supervisor routes between detector → classifier → rewriter."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

# Importing tracing has the side-effect of configuring LangSmith.
from . import tracing  # noqa: F401
from .agents import classifier_node, detector_node, rewriter_node
from .schemas import AnalysisResult, GraphState


def _route_after_detection(state: GraphState) -> Literal["classifier", "__end__"]:
    """If no greenwashing found, short-circuit. Otherwise continue."""
    if not state.get("spans"):
        return "__end__"
    return "classifier"


def build_graph():
    """Build and compile the LangGraph state machine."""
    g = StateGraph(GraphState)

    g.add_node("detector", detector_node)
    g.add_node("classifier", classifier_node)
    g.add_node("rewriter", rewriter_node)

    g.add_edge(START, "detector")
    g.add_conditional_edges(
        "detector",
        _route_after_detection,
        {"classifier": "classifier", "__end__": END},
    )
    g.add_edge("classifier", "rewriter")
    g.add_edge("rewriter", END)

    return g.compile()


# Compile once at import time so the LangSmith run tree is consistent.
graph = build_graph()


@traceable(name="analyse_document", run_type="chain", tags=["pipeline"])
def analyse(document: str) -> AnalysisResult:
    """High-level entrypoint. Runs the graph and returns a typed result."""
    initial: GraphState = {
        "document": document,
        "spans": [],
        "decisions": [],
        "replacements": [],
        "final_text": document,
        "errors": [],
        "retry_count": 0,
    }
    final_state = graph.invoke(initial)
    return AnalysisResult(
        original_text=document,
        final_text=final_state.get("final_text", document),
        spans=final_state.get("spans", []),
        decisions=final_state.get("decisions", []),
        replacements=final_state.get("replacements", []),
        errors=final_state.get("errors", []),
    )
