"""
Facade of the explainability layer: compute (ensemble explainer) +
presentation (renderer), returning API-agnostic rendered results.
"""

import logging
import time
from typing import Dict, Optional, Sequence

import numpy as np

from .contracts import (
    ExplainabilityConfig,
    ExplainabilityUnavailableError,
    RenderedExplanation,
    RenderedModelExplanation,
)
from .ensemble_explainer import EnsembleGradCAMExplainer
from .gradcam_explainer import OMNIXAI_AVAILABLE
from .heatmap_renderer import HeatmapRenderer

FRAMEWORK_NAME = "OmniXAI"
METHOD_NAME = "Grad-CAM"


class ExplanationService:
    """Entry point used by the API layer."""

    def __init__(
        self,
        explainer: EnsembleGradCAMExplainer,
        renderer: HeatmapRenderer,
        config: ExplainabilityConfig,
    ):
        self.config = config
        self._explainer = explainer
        self._renderer = renderer

    @classmethod
    def create(
        cls,
        ensemble_models: Sequence[Dict],
        config: ExplainabilityConfig,
        logger: Optional[logging.Logger] = None,
    ) -> "ExplanationService":
        """
        Raises:
            ExplainabilityUnavailableError: if OmniXAI is not installed
        """
        if not OMNIXAI_AVAILABLE:
            raise ExplainabilityUnavailableError(
                "OmniXAI is not installed (pip install omnixai)"
            )
        return cls(
            explainer=EnsembleGradCAMExplainer(ensemble_models, config, logger),
            renderer=HeatmapRenderer(config.heatmap_alpha, config.jpeg_quality),
            config=config,
        )

    def explain(
        self,
        variants: Dict[str, np.ndarray],
        target_class_index: int,
        include_per_model: bool = False,
    ) -> RenderedExplanation:
        """
        Explain a prediction.

        Args:
            variants: preprocessed image variants of ONE request
            target_class_index: class predicted by the ensemble
            include_per_model: also return one heatmap per ensemble member
        """
        started = time.perf_counter()
        explanation = self._explainer.explain(variants, target_class_index)
        renderer = self._renderer

        per_model = None
        if include_per_model:
            per_model = [
                RenderedModelExplanation(
                    model_architecture=m.model_architecture,
                    preprocessing_variant=m.preprocessing_variant,
                    target_layer=m.target_layer,
                    heatmap_overlay=renderer.encode_base64_jpeg(
                        renderer.overlay(explanation.base_image, m.heatmap)
                    ),
                    heatmap=renderer.encode_base64_jpeg(renderer.colorize(m.heatmap)),
                )
                for m in explanation.model_heatmaps
            ]

        return RenderedExplanation(
            method=METHOD_NAME,
            framework=FRAMEWORK_NAME,
            target_class_index=explanation.target_class_index,
            heatmap_overlay=renderer.encode_base64_jpeg(
                renderer.overlay(explanation.base_image, explanation.ensemble_heatmap)
            ),
            heatmap=renderer.encode_base64_jpeg(
                renderer.colorize(explanation.ensemble_heatmap)
            ),
            explanation_time_ms=round((time.perf_counter() - started) * 1000, 2),
            per_model=per_model,
        )
