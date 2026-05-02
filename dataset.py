"""Upload the seed dataset to LangSmith."""

from __future__ import annotations

import json
from pathlib import Path

from langsmith import Client

SEED_PATH = Path(__file__).resolve().parent.parent / "data" / "seed_examples.json"


def seed_dataset(name: str, description: str | None = None) -> str:
    """Create (or reuse) a LangSmith dataset and upload the seed examples."""
    client = Client()
    examples = json.loads(SEED_PATH.read_text())

    if client.has_dataset(dataset_name=name):
        ds = client.read_dataset(dataset_name=name)
    else:
        ds = client.create_dataset(
            dataset_name=name,
            description=description or "Synthetic greenwashing examples for evaluation.",
        )

        client.create_examples(
            dataset_id=ds.id,
            inputs=[{"document": ex["document"]} for ex in examples],
            outputs=[
                {
                    "expected_phrases": ex["expected_phrases"],
                    "expected_actions": ex["expected_actions"],
                }
                for ex in examples
            ],
        )

    return str(ds.id)


if __name__ == "__main__":
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "greenwashing-eval"
    print(f"Seeded dataset {name}: {seed_dataset(name)}")
