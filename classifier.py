"""Classifier agent — for each flagged span, decides keep / replace / delete."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from ..failure import safe_invoke
from ..models import get_llm
from ..schemas import Action, ClassifierOutput, Decision, GraphState

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are auditing greenwashing claims. For each FLAGGED PHRASE, decide "
            "ONE action:\n"
            "  - 'keep'    : the phrase is actually substantiated by the surrounding context.\n"
            "  - 'replace' : the phrase is misleading but the surrounding statement can be "
            "salvaged with a more honest wording.\n"
            "  - 'delete'  : the phrase has no substance; remove it.\n\n"
            "Return STRICT JSON, no markdown:\n"
            "{{\n"
            '  "decisions": [\n'
            '    {{"phrase": "<copy from input>", '
            '"action": "keep|replace|delete", '
            '"justification": "<one sentence>"}}\n'
            "  ]\n"
            "}}\n",
        ),
        (
            "human",
            "DOCUMENT:\n{document}\n\nFLAGGED PHRASES:\n{phrases_json}",
        ),
    ]
)


@traceable(name="classifier_agent", run_type="chain", tags=["agent", "classifier"])
def classifier_node(state: GraphState) -> GraphState:
    """Classify every flagged span. No-op if there are no spans."""
    spans = state.get("spans", [])
    if not spans:
        return {"decisions": []}

    llm = get_llm(role="agent", temperature=0.0)
    chain = CLASSIFIER_PROMPT | llm

    phrases_json = json.dumps(
        [{"phrase": s.phrase, "reason": s.reason} for s in spans],
        indent=2,
    )

    parsed, errors = safe_invoke(
        chain=chain,
        inputs={"document": state["document"], "phrases_json": phrases_json},
        schema=ClassifierOutput,
        agent_name="classifier",
    )

    if parsed is None:
        # Conservative fallback: keep everything if classification failed.
        decisions = [
            Decision(phrase=s.phrase, action=Action.KEEP, justification="classifier_failed")
            for s in spans
        ]
    else:
        decisions = parsed.decisions

    return {
        "decisions": decisions,
        "errors": [*state.get("errors", []), *errors],
    }
