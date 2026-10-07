"""
Ensemble-level Grad-CAM: runs the atomic explainer for every ensemble member
and fuses the heatmaps into one explanation of the ensemble prediction.
"""

import logging
import threading
from typing import Dict, List, Optional, Sequence

import numpy as np

from .contracts import (
    EnsembleExplanation,
    ExplainabilityConfig,
    ExplainabilityError,
    ModelHeatmap,
)
from .gradcam_explainer import GradCAMExplainer


class EnsembleGradCAMExplainer:
    """
    Explains an ensemble prediction.

    Each member is explained on the preprocessing variant it was trained on,
    towards the class predicted by the ensemble. Heatmaps are averaged
    (an ensemble-level analogue of soft voting).
    """

    def __init__(
        self,
        ensemble_models: Sequence[Dict],
        config: ExplainabilityConfig,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Args:
            ensemble_models: ``ModelEnsemble.models`` - entries of
                ``{"model": nn.Module, "config": dict}``
            config: explainability configuration
            logger: optional logger
        """
        self._ensemble_models = ensemble_models
        self._config = config
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._explainers: Dict[int, GradCAMExplainer] = {}
        self._init_lock = threading.Lock()

    def _target_layer_path(self, model_config: Dict) -> str:
        architecture = model_config["architecture"].lower()
        path = model_config.get("target_layer") or self._config.target_layers.get(
            architecture
        )
        if not path:
            raise ExplainabilityError(
                f"No Grad-CAM target layer configured for architecture '{architecture}'"
            )
        return path

    def _get_explainer(self, index: int) -> GradCAMExplainer:
        """Lazily build (and cache) the explainer of one member."""
        with self._init_lock:
            explainer = self._explainers.get(index)
            if explainer is None:
                entry = self._ensemble_models[index]
                explainer = GradCAMExplainer(
                    entry["model"], self._target_layer_path(entry["config"])
                )
                self._explainers[index] = explainer
            return explainer

    @staticmethod
    def _select_variant(model_config: Dict, variants: Dict[str, np.ndarray]) -> Optional[str]:
        """Same variant selection as ``ModelEnsemble.predict_batch``."""
        name = model_config.get("preprocessing_variant", "original")
        if name in variants:
            return name
        return "original" if "original" in variants else None

    def explain(
        self, variants: Dict[str, np.ndarray], target_class_index: int
    ) -> EnsembleExplanation:
        """
        Args:
            variants: preprocessed image variants of ONE request
            target_class_index: class predicted by the ensemble

        Raises:
            ExplainabilityError: if no ensemble member could be explained
        """
        model_heatmaps: List[ModelHeatmap] = []

        for index, entry in enumerate(self._ensemble_models):
            model_config = entry["config"]
            variant_name = self._select_variant(model_config, variants)
            if variant_name is None:
                self._logger.warning(
                    "No suitable image variant for model %s; skipping explanation",
                    model_config.get("model_path", "unknown"),
                )
                continue

            try:
                explainer = self._get_explainer(index)
                heatmap = explainer.explain(variants[variant_name], target_class_index)
            except Exception as e:
                self._logger.error(
                    "Grad-CAM failed for model %s: %s",
                    model_config.get("model_path", "unknown"),
                    e,
                )
                continue

            model_heatmaps.append(
                ModelHeatmap(
                    model_architecture=model_config["architecture"],
                    preprocessing_variant=variant_name,
                    target_layer=explainer.target_layer_path,
                    heatmap=heatmap,
                )
            )

        if not model_heatmaps:
            raise ExplainabilityError("No ensemble member could be explained")

        return EnsembleExplanation(
            target_class_index=target_class_index,
            base_image=variants["original"]
            if "original" in variants
            else next(iter(variants.values())),
            ensemble_heatmap=self._fuse(model_heatmaps),
            model_heatmaps=model_heatmaps,
        )

    @staticmethod
    def _fuse(model_heatmaps: List[ModelHeatmap]) -> np.ndarray:
        """Average member heatmaps and rescale to [0, 1]."""
        reference_shape = model_heatmaps[0].heatmap.shape
        fused = np.mean(
            [h.heatmap for h in model_heatmaps if h.heatmap.shape == reference_shape],
            axis=0,
        )
        peak = float(fused.max())
        return (fused / peak if peak > 0 else fused).astype(np.float32)
