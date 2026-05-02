"""Structured schemas used by every agent. Strict typing = clean traces & easy evals."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


class Action(str, Enum):
    """What to do with a flagged greenwashing span."""

    KEEP = "keep"
    REPLACE = "replace"
    DELETE = "delete"


class GreenwashingSpan(BaseModel):
    """A single phrase flagged by the detector."""

    phrase: str = Field(description="Exact substring from the document.")
    reason: str = Field(description="Why it looks like greenwashing.")
    confidence: float = Field(ge=0.0, le=1.0, description="Detector confidence 0-1.")


class DetectorOutput(BaseModel):
    """Detector agent output."""

    spans: list[GreenwashingSpan] = Field(default_factory=list)


class Decision(BaseModel):
    """Classifier decision for a single span."""

    phrase: str
    action: Action
    justification: str


class ClassifierOutput(BaseModel):
    """Classifier agent output."""

    decisions: list[Decision] = Field(default_factory=list)


class Replacement(BaseModel):
    """Rewriter agent output for a single span."""

    phrase: str
    replacement: str
    rationale: str


class RewriterOutput(BaseModel):
    """Rewriter agent output."""

    replacements: list[Replacement] = Field(default_factory=list)


class AnalysisResult(BaseModel):
    """Final result returned to the user / saved to disk."""

    original_text: str
    final_text: str
    spans: list[GreenwashingSpan]
    decisions: list[Decision]
    replacements: list[Replacement]
    errors: list[str] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    """LangGraph state. `total=False` so partial updates are valid."""

    document: str
    spans: list[GreenwashingSpan]
    decisions: list[Decision]
    replacements: list[Replacement]
    final_text: str
    errors: list[str]
    retry_count: int
