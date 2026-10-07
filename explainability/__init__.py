"""Explainability (XAI) layer: Grad-CAM via OmniXAI for the DR ensemble."""

from .contracts import (
    ExplainabilityConfig,
    ExplainabilityError,
    ExplainabilityUnavailableError,
    RenderedExplanation,
    RenderedModelExplanation,
)
from .service import ExplanationService

__all__ = [
    "ExplainabilityConfig",
    "ExplainabilityError",
    "ExplainabilityUnavailableError",
    "ExplanationService",
    "RenderedExplanation",
    "RenderedModelExplanation",
]
