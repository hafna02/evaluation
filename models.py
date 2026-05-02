"""LLM provider abstraction. Ollama is the default; HuggingFace is a fallback."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from .config import settings

log = logging.getLogger(__name__)


def _build_ollama(model: str, temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=settings.ollama_base_url,
        temperature=temperature,
        # Force JSON-friendly output where the agent expects structured data.
        format="json",
    )


def _build_huggingface(model: str, temperature: float) -> BaseChatModel:
    """Lazy HF import — only required if the user installs the `hf` extra."""
    try:
        from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "HuggingFace fallback requested but `langchain-huggingface` and "
            "`transformers` are not installed. Run: pip install '.[hf]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    pipe = pipeline(
        "text-generation",
        model=hf_model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=max(temperature, 0.01),  # HF rejects 0.0
        do_sample=temperature > 0.0,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return ChatHuggingFace(llm=llm)


@lru_cache(maxsize=8)
def get_llm(role: str = "agent", temperature: float = 0.0) -> BaseChatModel:
    """Return a chat model. `role` selects between agent/judge config.

    Tries the configured provider first; on failure, tries the other one.
    Cached so repeated calls reuse the same client.
    """
    is_judge = role == "judge"
    primary = settings.llm_provider
    fallback = "huggingface" if primary == "ollama" else "ollama"

    builders: dict[str, Any] = {
        "ollama": lambda: _build_ollama(
            settings.ollama_judge_model if is_judge else settings.ollama_model,
            temperature,
        ),
        "huggingface": lambda: _build_huggingface(
            settings.hf_judge_model if is_judge else settings.hf_model,
            temperature,
        ),
    }

    try:
        log.info("Loading %s model (provider=%s)", role, primary)
        return builders[primary]()
    except Exception as exc:  # noqa: BLE001
        log.warning("Primary provider %s failed: %s. Falling back to %s.", primary, exc, fallback)
        return builders[fallback]()
