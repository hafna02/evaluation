"""Rewriter agent — proposes honest replacements for spans marked 'replace'."""

from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from ..failure import safe_invoke
from ..models import get_llm
from ..schemas import Action, GraphState, Replacement, RewriterOutput

REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite each greenwashing phrase to be HONEST and SPECIFIC. "
            "If the phrase makes a quantitative claim, qualify it (e.g. 'eco-friendly' "
            "→ 'made with 30% recycled material' if context supports it, otherwise "
            "'designed to reduce material waste'). Never invent statistics — keep it "
            "qualitative when no numbers are given.\n\n"
            "Return STRICT JSON, no markdown:\n"
            "{{\n"
            '  "replacements": [\n'
            '    {{"phrase": "<copy>", '
            '"replacement": "<rewritten phrase>", '
            '"rationale": "<one sentence>"}}\n'
            "  ]\n"
            "}}\n",
        ),
        (
            "human",
            "DOCUMENT:\n{document}\n\nPHRASES TO REPLACE:\n{phrases_json}",
        ),
    ]
)


def _apply_edits(
    document: str,
    decisions,
    replacements: list[Replacement],
) -> str:
    """Apply keep / replace / delete decisions to produce final text."""
    repl_map = {r.phrase: r.replacement for r in replacements}
    text = document
    for d in decisions:
        if d.action == Action.KEEP:
            continue
        if d.action == Action.DELETE:
            text = text.replace(d.phrase, "")
        elif d.action == Action.REPLACE:
            new = repl_map.get(d.phrase, d.phrase)
            text = text.replace(d.phrase, new)
    # Tidy up double spaces left by deletes.
    return " ".join(text.split())


@traceable(name="rewriter_agent", run_type="chain", tags=["agent", "rewriter"])
def rewriter_node(state: GraphState) -> GraphState:
    """Generate replacements and assemble the final document."""
    decisions = state.get("decisions", [])
    to_replace = [d for d in decisions if d.action == Action.REPLACE]

    replacements: list[Replacement] = []
    errors: list[str] = []

    if to_replace:
        llm = get_llm(role="agent", temperature=0.2)
        chain = REWRITER_PROMPT | llm

        phrases_json = json.dumps(
            [{"phrase": d.phrase, "justification": d.justification} for d in to_replace],
            indent=2,
        )

        parsed, errs = safe_invoke(
            chain=chain,
            inputs={"document": state["document"], "phrases_json": phrases_json},
            schema=RewriterOutput,
            agent_name="rewriter",
        )
        errors.extend(errs)

        if parsed is not None:
            replacements = parsed.replacements
        else:
            # Fallback: keep originals if rewriting fails.
            replacements = [
                Replacement(phrase=d.phrase, replacement=d.phrase, rationale="rewriter_failed")
                for d in to_replace
            ]

    final_text = _apply_edits(state["document"], decisions, replacements)

    return {
        "replacements": replacements,
        "final_text": final_text,
        "errors": [*state.get("errors", []), *errors],
    }
