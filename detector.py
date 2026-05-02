"""Detector agent — finds candidate greenwashing phrases in the document."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from ..failure import safe_invoke
from ..models import get_llm
from ..schemas import DetectorOutput, GraphState

DETECTOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert in corporate sustainability claims. Identify "
            "phrases in the user's document that may be GREENWASHING — vague, "
            "unverifiable, or misleading environmental claims (e.g. 'eco-friendly', "
            "'all-natural', '100% sustainable', 'carbon neutral by intention').\n\n"
            "Return STRICT JSON matching this schema (no prose, no markdown):\n"
            "{{\n"
            '  "spans": [\n'
            '    {{"phrase": "<exact substring from document>", '
            '"reason": "<why suspicious>", '
            '"confidence": <float 0-1>}}\n'
            "  ]\n"
            "}}\n\n"
            "Rules:\n"
            "- `phrase` MUST appear verbatim in the document.\n"
            "- Return an empty list if nothing is suspicious.\n"
            "- Maximum 10 spans.",
        ),
        ("human", "DOCUMENT:\n{document}"),
    ]
)


@traceable(name="detector_agent", run_type="chain", tags=["agent", "detector"])
def detector_node(state: GraphState) -> GraphState:
    """Run the detector and update graph state."""
    llm = get_llm(role="agent", temperature=0.0)
    chain = DETECTOR_PROMPT | llm

    parsed, errors = safe_invoke(
        chain=chain,
        inputs={"document": state["document"]},
        schema=DetectorOutput,
        agent_name="detector",
    )

    spans = parsed.spans if parsed else []
    # Keep only spans whose phrase actually appears in the document.
    spans = [s for s in spans if s.phrase in state["document"]]

    return {
        "spans": spans,
        "errors": [*state.get("errors", []), *errors],
    }
