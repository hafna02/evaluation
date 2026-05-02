"""Custom rule-based evaluators.

These run locally (no LLM call) and verify deterministic invariants of the
pipeline output. Each evaluator returns a dict the LangSmith SDK expects:
    {"key": "<metric_name>", "score": <0-1>, "comment": "<optional>"}
"""

from __future__ import annotations

from typing import Any


def detection_recall(run, example) -> dict[str, Any]:
    """Fraction of expected greenwashing phrases the detector caught."""
    expected: list[str] = example.outputs.get("expected_phrases", []) or []
    if not expected:
        # No phrases to find — full score iff detector also found none.
        detected = run.outputs.get("spans", []) if run.outputs else []
        score = 1.0 if not detected else 0.0
        return {"key": "detection_recall", "score": score}

    detected_phrases = {s["phrase"].lower() for s in (run.outputs or {}).get("spans", [])}
    hits = sum(1 for p in expected if p.lower() in detected_phrases)
    return {
        "key": "detection_recall",
        "score": hits / len(expected),
        "comment": f"{hits}/{len(expected)} expected phrases detected",
    }


def detection_precision(run, example) -> dict[str, Any]:
    """Fraction of detected phrases that were actually expected."""
    expected = {p.lower() for p in (example.outputs.get("expected_phrases") or [])}
    detected = [(s["phrase"]).lower() for s in (run.outputs or {}).get("spans", [])]

    if not detected:
        return {"key": "detection_precision", "score": 1.0 if not expected else 0.0}

    hits = sum(1 for p in detected if p in expected)
    return {
        "key": "detection_precision",
        "score": hits / len(detected),
        "comment": f"{hits}/{len(detected)} detected phrases were expected",
    }


def action_accuracy(run, example) -> dict[str, Any]:
    """For phrases present in BOTH expected and detected, did we pick the right action?"""
    expected_actions: dict[str, str] = example.outputs.get("expected_actions") or {}
    if not expected_actions:
        return {"key": "action_accuracy", "score": 1.0, "comment": "no expected actions"}

    decisions = {
        d["phrase"].lower(): d["action"].lower()
        for d in (run.outputs or {}).get("decisions", [])
    }

    overlap = [p for p in expected_actions if p.lower() in decisions]
    if not overlap:
        return {"key": "action_accuracy", "score": 0.0, "comment": "no overlap"}

    correct = sum(
        1 for p in overlap if decisions[p.lower()] == expected_actions[p].lower()
    )
    return {
        "key": "action_accuracy",
        "score": correct / len(overlap),
        "comment": f"{correct}/{len(overlap)} actions correct",
    }


def output_validity(run, example) -> dict[str, Any]:
    """Did the pipeline produce a non-empty `final_text` and zero hard errors?"""
    out = run.outputs or {}
    has_text = bool(out.get("final_text"))
    no_errors = not out.get("errors")
    score = 1.0 if has_text and no_errors else 0.5 if has_text else 0.0
    return {
        "key": "output_validity",
        "score": score,
        "comment": f"final_text={has_text}, no_errors={no_errors}",
    }


def removed_flagged_terms(run, example) -> dict[str, Any]:
    """For decisions of type 'replace' or 'delete', verify the phrase is gone from final text."""
    out = run.outputs or {}
    final = (out.get("final_text") or "").lower()
    decisions = out.get("decisions", [])
    targets = [d["phrase"] for d in decisions if d["action"] in ("replace", "delete")]

    if not targets:
        return {"key": "removed_flagged_terms", "score": 1.0, "comment": "nothing to remove"}

    removed = sum(1 for t in targets if t.lower() not in final)
    return {
        "key": "removed_flagged_terms",
        "score": removed / len(targets),
        "comment": f"{removed}/{len(targets)} target phrases removed",
    }


CUSTOM_EVALUATORS = [
    detection_recall,
    detection_precision,
    action_accuracy,
    output_validity,
    removed_flagged_terms,
]
