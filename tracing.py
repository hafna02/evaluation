"""LangSmith tracing bootstrap.

Importing this module sets the env vars LangSmith reads, so any LangChain /
LangGraph object created afterwards is auto-traced.
"""

from __future__ import annotations

import logging
import os

from .config import settings

log = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Configure LangSmith env vars from settings. Idempotent."""
    if not settings.langsmith_tracing:
        log.info("LangSmith tracing disabled via settings.")
        return

    if not settings.langsmith_api_key:
        log.warning(
            "LANGSMITH_API_KEY not set — tracing will be skipped. "
            "Add it to .env to enable run capture."
        )
        os.environ["LANGSMITH_TRACING"] = "false"
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    log.info("LangSmith tracing enabled (project=%s).", settings.langsmith_project)


# Auto-run on import — keeps user code clean.
setup_tracing()
