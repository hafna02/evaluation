"""Failure-handling primitives shared by every agent.

Two patterns:
  1. `safe_invoke` — wraps a chain call with retries, JSON repair, and a
     guaranteed Pydantic-typed return. Errors are captured (not raised) so the
     graph keeps flowing.
  2. `record_error` — appends an error string to the graph state's `errors`
     field. Surfaces in LangSmith as run metadata.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from langsmith import traceable
from pydantic import BaseModel, ValidationError

from .config import settings

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of an LLM response."""
    text = text.strip()
    # Strip common code-fence wrappers.
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in LLM output.")
    return text[start : end + 1]


@traceable(name="safe_invoke", run_type="chain")
def safe_invoke(
    chain,  # noqa: ANN001 — duck-typed Runnable
    inputs: dict,
    schema: type[T],
    agent_name: str,
) -> tuple[T | None, list[str]]:
    """Invoke a chain, parse JSON, validate against `schema`, retry on failure.

    Returns `(parsed_or_none, errors)`. Errors are returned, never raised — the
    graph decides what to do with them.
    """
    errors: list[str] = []
    last_raw: str | None = None

    for attempt in range(settings.max_retries + 1):
        try:
            response = chain.invoke(inputs)
            raw = response.content if hasattr(response, "content") else str(response)
            last_raw = raw
            parsed = schema.model_validate_json(_extract_json(raw))
            if attempt > 0:
                log.info("[%s] succeeded on retry %d", agent_name, attempt)
            return parsed, errors
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            err = f"[{agent_name}] attempt {attempt + 1} parse/validation error: {exc}"
            log.warning(err)
            errors.append(err)
        except Exception as exc:  # noqa: BLE001
            err = f"[{agent_name}] attempt {attempt + 1} runtime error: {exc}"
            log.error(err)
            errors.append(err)

    log.error("[%s] all retries exhausted. Last raw output: %r", agent_name, last_raw)
    return None, errors
