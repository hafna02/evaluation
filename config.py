"""Centralised settings loaded from environment / .env file."""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.

    Values are loaded from environment variables and `.env` (env takes priority).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LangSmith
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = True
    langsmith_project: str = "greenwashing-detector"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    # Model provider
    llm_provider: Literal["ollama", "huggingface"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    ollama_judge_model: str = "llama3.2:3b"

    # HF fallback
    hf_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    hf_judge_model: str = "meta-llama/Llama-3.2-3B-Instruct"

    # Runtime
    max_retries: int = 2
    log_level: str = "INFO"


settings = Settings()
