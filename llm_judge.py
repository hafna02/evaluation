"""LLM-as-judge evaluators.

We use a separate `role="judge"` model so the judge is decoupled from the agent
under test (configurable to a stronger model via OLLAMA_JUDGE_MODEL).
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable

from greenwashing.failure import _extract_json
from greenwashing.models import get_llm

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict auditor of corporate sustainability claims. Given a "
            "REWRITTEN document, decide whether it still contains greenwashing — "
            "vague, unverifiable, or unsubstantiated environmental claims.\n\n"
            "Score on a 1-5 Likert scale where:\n"
            "  5 = no greenwashing, all claims are specific or qualified\n"
            "  4 = mostly clean, minor vagueness\n"
            "  3 = some greenwashing remains\n"
            "  2 = significant greenwashing remains\n"
            "  1 = the rewrite did not improve things\n\n"
            "Return STRICT JSON, no markdown:\n"
            '{{"score": <1-5>, "reasoning": "<one paragraph>"}}',
        ),
        ("human", "ORIGINAL:\n{original}\n\nREWRITTEN:\n{rewritten}"),
    ]
)

FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are checking whether a rewritten document FAITHFULLY preserves the "
            "non-greenwashing content of the original. The rewrite should remove or "
            "qualify vague green claims, but it MUST NOT invent new facts, numbers, "
            "or certifications that weren't in the original.\n\n"
            "Return STRICT JSON, no markdown:\n"
            '{{"faithful": true|false, "reasoning": "<one paragraph>", '
            '"hallucinated_claims": ["<phrase>", ...]}}',
        ),
        ("human", "ORIGINAL:\n{original}\n\nREWRITTEN:\n{rewritten}"),
    ]
)


def _judge(prompt, inputs: dict, fallback: dict) -> dict:
    """Run a judge prompt and parse JSON, returning fallback on failure."""
    llm = get_llm(role="judge", temperature=0.0)
    chain = prompt | llm
    try:
        resp = chain.invoke(inputs)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        return json.loads(_extract_json(raw))
    except Exception:  # noqa: BLE001
        return fallback


@traceable(name="judge_greenwashing_residue", run_type="chain", tags=["evaluator", "judge"])
def judge_greenwashing_residue(run, example) -> dict[str, Any]:
    """LLM judge: how clean is the rewrite?"""
    out = run.outputs or {}
    rewritten = out.get("final_text") or ""
    original = (run.inputs or {}).get("document") or example.inputs.get("document", "")

    verdict = _judge(
        JUDGE_PROMPT,
        {"original": original, "rewritten": rewritten},
        fallback={"score": 0, "reasoning": "judge_failed"},
    )

    likert = int(verdict.get("score", 0))
    return {
        "key": "judge_greenwashing_residue",
        "score": max(0.0, min(1.0, (likert - 1) / 4)),  # normalise 1-5 → 0-1
        "comment": verdict.get("reasoning", ""),
    }


@traceable(name="judge_faithfulness", run_type="chain", tags=["evaluator", "judge"])
def judge_faithfulness(run, example) -> dict[str, Any]:
    """LLM judge: did the rewrite invent new claims?"""
    out = run.outputs or {}
    rewritten = out.get("final_text") or ""
    original = (run.inputs or {}).get("document") or example.inputs.get("document", "")

    verdict = _judge(
        FAITHFULNESS_PROMPT,
        {"original": original, "rewritten": rewritten},
        fallback={"faithful": False, "reasoning": "judge_failed", "hallucinated_claims": []},
    )

    return {
        "key": "judge_faithfulness",
        "score": 1.0 if verdict.get("faithful") else 0.0,
        "comment": verdict.get("reasoning", ""),
    }


JUDGE_EVALUATORS = [judge_greenwashing_residue, judge_faithfulness]
