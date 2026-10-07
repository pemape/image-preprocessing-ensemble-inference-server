"""
Atomic Grad-CAM explainer for a single torch model, built on OmniXAI.

Knows nothing about ensembles, HTTP or rendering - it maps
``(image, target class) -> heatmap``.
"""

import copy
import threading

import numpy as np
import torch
import torch.nn as nn

from .contracts import ExplainabilityError, ExplainabilityUnavailableError
from .target_layer import resolve_target_layer

try:
    from omnixai.data.image import Image as OmniXAIImage
    from omnixai.explainers.vision.specific.gradcam.pytorch.gradcam import GradCAM

    OMNIXAI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the environment
    OmniXAIImage = None
    GradCAM = None
    OMNIXAI_AVAILABLE = False


def _preprocess(images) -> torch.Tensor:
    """OmniXAI ``Image`` (N, H, W, C in 0..255) -> model input (N, C, H, W).

    Mirrors ``ModelEnsemble._prepare_input`` (``ToTensor`` on a [0, 1] image).
    """
    array = images.to_numpy().astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(0, 3, 1, 2).contiguous()


class GradCAMExplainer:
    """Grad-CAM for one model.

    OmniXAI registers forward/backward hooks on the target layer that keep
    every activation they see. To keep the serving model hook-free (and
    inference thread-safe) the explainer works on a private copy of the model;
    explanations are serialised with a lock because the hooks are stateful.
    """

    def __init__(self, model: nn.Module, target_layer_path: str):
        if not OMNIXAI_AVAILABLE:
            raise ExplainabilityUnavailableError(
                "OmniXAI is not installed (pip install omnixai)"
            )

        self.target_layer_path = target_layer_path
        self._model = copy.deepcopy(model).eval()
        self._lock = threading.Lock()
        self._gradcam = GradCAM(
            model=self._model,
            target_layer=resolve_target_layer(self._model, target_layer_path),
            preprocess_function=_preprocess,
            mode="classification",
        )

    def explain(self, image: np.ndarray, target_class_index: int) -> np.ndarray:
        """
        Compute the Grad-CAM heatmap for ``target_class_index``.

        Args:
            image: (H, W, 3) RGB image exactly as fed to the model (float in
                [0, 1] or uint8, see ``ModelEnsemble._prepare_input``)
            target_class_index: class the heatmap should explain

        Returns:
            (H, W) float32 heatmap in [0, 1]
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ExplainabilityError(
                f"Expected an (H, W, 3) image, got shape {image.shape}"
            )

        unit_image = image.astype(np.float32)
        if image.dtype == np.uint8:
            unit_image /= 255.0
        omnixai_image = OmniXAIImage(
            data=unit_image * 255.0, batched=False, channel_last=True
        )

        with self._lock, torch.enable_grad():
            explanations = self._gradcam.explain(
                omnixai_image, y=int(target_class_index)
            )

        scores = np.asarray(explanations.get_explanations(0)["scores"])
        return np.clip(scores.astype(np.float32), 0.0, 1.0)
