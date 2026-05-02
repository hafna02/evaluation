"""Run the full evaluation suite against a LangSmith dataset.

Combines:
  - Custom rule-based evaluators (precision/recall/action accuracy/etc.)
  - LangChain built-in evaluators (string distance for the rewrite)
  - LLM-as-judge evaluators (residue + faithfulness)

Usage:
    python -m evals.run_eval --seed
    python -m evals.run_eval                  # uses default dataset name
    python -m evals.run_eval --dataset my-ds
"""

from __future__ import annotations

import argparse
import logging

from langsmith.evaluation import LangChainStringEvaluator, evaluate

# Side-effect import: configures LangSmith env vars.
from greenwashing import tracing  # noqa: F401
from greenwashing.graph import analyse

from .custom_evaluators import CUSTOM_EVALUATORS
from .dataset import seed_dataset
from .llm_judge import JUDGE_EVALUATORS

log = logging.getLogger(__name__)


def _target(inputs: dict) -> dict:
    """Adapter: dataset input → graph invocation → flat dict for evaluators."""
    result = analyse(inputs["document"])
    return {
        "spans": [s.model_dump() for s in result.spans],
        "decisions": [d.model_dump() for d in result.decisions],
        "replacements": [r.model_dump() for r in result.replacements],
        "final_text": result.final_text,
        "errors": result.errors,
    }


def _builtin_evaluators():
    """Use LangSmith's built-in string-distance evaluator on the rewrite."""
    return [
        LangChainStringEvaluator(
            "embedding_distance",
            prepare_data=lambda run, example: {
                "prediction": (run.outputs or {}).get("final_text", ""),
                "reference": example.inputs.get("document", ""),
            },
        ),
    ]


def main(dataset_name: str = "greenwashing-eval", seed: bool = False) -> None:
    if seed:
        log.info("Seeding dataset %s …", dataset_name)
        seed_dataset(dataset_name)

    all_evaluators = [*CUSTOM_EVALUATORS, *JUDGE_EVALUATORS]

    log.info("Running evaluation on dataset %s …", dataset_name)
    results = evaluate(
        _target,
        data=dataset_name,
        evaluators=all_evaluators,
        summary_evaluators=[],
        experiment_prefix="greenwashing",
        max_concurrency=2,
    )
    print(f"\nDone. View results: {results}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="greenwashing-eval")
    parser.add_argument("--seed", action="store_true", help="Upload seed dataset first.")
    args = parser.parse_args()
    main(dataset_name=args.dataset, seed=args.seed)
