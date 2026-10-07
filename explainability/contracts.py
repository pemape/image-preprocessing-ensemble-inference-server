"""
Data contracts of the explainability layer.

Pure data holders only - no computation and no framework imports - so every
other module of the package (and the server) can depend on them safely.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

SUPPORTED_METHODS = ("gradcam",)

# Default Grad-CAM target layers (dotted path, integer tokens index into containers).
# Models handed to the explainer are the raw backbones (see ModelEnsemble._load_model).
DEFAULT_TARGET_LAYERS: Dict[str, str] = {
    "efficientnetb4": "features.-1",
    "xception": "bn4",
}


class ExplainabilityError(RuntimeError):
    """Raised when an explanation cannot be produced."""


class ExplainabilityUnavailableError(ExplainabilityError):
    """Raised when the explainability backend (OmniXAI) cannot be used."""


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Configuration of the explainability layer (``explainability`` YAML section)."""

    enabled: bool = False
    method: str = "gradcam"
    heatmap_alpha: float = 0.4
    jpeg_quality: int = 90
    target_layers: Mapping[str, str] = field(
        default_factory=lambda: dict(DEFAULT_TARGET_LAYERS)
    )

    @classmethod
    def from_dict(cls, raw: Optional[Mapping[str, Any]]) -> "ExplainabilityConfig":
        raw = raw or {}

        method = str(raw.get("method", "gradcam")).lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported explainability method '{method}'. "
                f"Supported: {list(SUPPORTED_METHODS)}"
            )

        target_layers = dict(DEFAULT_TARGET_LAYERS)
        target_layers.update(
            {
                str(arch).lower(): str(path)
                for arch, path in (raw.get("target_layers") or {}).items()
            }
        )

        return cls(
            enabled=bool(raw.get("enabled", False)),
            method=method,
            heatmap_alpha=min(max(float(raw.get("heatmap_alpha", 0.4)), 0.0), 1.0),
            jpeg_quality=min(max(int(raw.get("jpeg_quality", 90)), 1), 100),
            target_layers=target_layers,
        )


@dataclass
class ModelHeatmap:
    """Raw Grad-CAM heatmap of one ensemble member."""

    model_architecture: str
    preprocessing_variant: str
    target_layer: str
    heatmap: np.ndarray  # (H, W) float32 in [0, 1]


@dataclass
class EnsembleExplanation:
    """Raw (numerical) explanation of an ensemble prediction."""

    target_class_index: int
    base_image: np.ndarray  # (H, W, 3) image the heatmaps are drawn on
    ensemble_heatmap: np.ndarray  # (H, W) float32 in [0, 1]
    model_heatmaps: List[ModelHeatmap] = field(default_factory=list)


@dataclass
class RenderedModelExplanation:
    """Presentation-ready (base64 JPEG) explanation of one ensemble member."""

    model_architecture: str
    preprocessing_variant: str
    target_layer: str
    heatmap_overlay: str
    heatmap: str


@dataclass
class RenderedExplanation:
    """Presentation-ready explanation of an ensemble prediction."""

    method: str
    framework: str
    target_class_index: int
    heatmap_overlay: str
    heatmap: str
    explanation_time_ms: float
    per_model: Optional[List[RenderedModelExplanation]] = None
