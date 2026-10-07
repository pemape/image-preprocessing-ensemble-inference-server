"""Tests of the explainability layer and its API integration (no OmniXAI required)."""

import base64
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

torch = pytest.importorskip("torch")

from explainability.contracts import (  # noqa: E402
    ExplainabilityConfig,
    ExplainabilityError,
    ModelHeatmap,
    RenderedExplanation,
    RenderedModelExplanation,
)
from explainability.ensemble_explainer import EnsembleGradCAMExplainer  # noqa: E402
from explainability.heatmap_renderer import HeatmapRenderer  # noqa: E402
from explainability.target_layer import resolve_target_layer  # noqa: E402


# ---------------------------------------------------------------- config


def test_config_defaults_to_disabled():
    cfg = ExplainabilityConfig.from_dict(None)
    assert cfg.enabled is False
    assert cfg.method == "gradcam"
    assert "efficientnetb4" in cfg.target_layers


def test_config_overrides_and_clamps():
    cfg = ExplainabilityConfig.from_dict(
        {
            "enabled": True,
            "heatmap_alpha": 5,
            "jpeg_quality": 0,
            "target_layers": {"EfficientNetB4": "features.-2"},
        }
    )
    assert cfg.enabled is True
    assert cfg.heatmap_alpha == 1.0
    assert cfg.jpeg_quality == 1
    assert cfg.target_layers["efficientnetb4"] == "features.-2"


def test_config_rejects_unknown_method():
    with pytest.raises(ValueError):
        ExplainabilityConfig.from_dict({"method": "lime"})


# ---------------------------------------------------------------- target layer


class _TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 8, 3)
        )
        self.bn4 = torch.nn.BatchNorm2d(8)


def test_resolve_target_layer_by_index_and_attribute():
    net = _TinyNet()
    assert resolve_target_layer(net, "features.-1") is net.features[1]
    assert resolve_target_layer(net, "features.0") is net.features[0]
    assert resolve_target_layer(net, "bn4") is net.bn4


def test_resolve_target_layer_invalid_path():
    with pytest.raises(ExplainabilityError):
        resolve_target_layer(_TinyNet(), "features.9")
    with pytest.raises(ExplainabilityError):
        resolve_target_layer(_TinyNet(), "missing")


# ---------------------------------------------------------------- renderer


def test_renderer_overlay_and_encoding_roundtrip():
    renderer = HeatmapRenderer(alpha=0.5, jpeg_quality=80)
    image = np.full((40, 50, 3), 128, dtype=np.uint8)
    heatmap = np.linspace(0, 1, 10 * 10, dtype=np.float32).reshape(10, 10)

    colored = renderer.colorize(heatmap)
    assert colored.shape == (10, 10, 3) and colored.dtype == np.uint8

    overlay = renderer.overlay(image, heatmap)  # heatmap is resized to the image
    assert overlay.shape == image.shape

    decoded = base64.b64decode(renderer.encode_base64_jpeg(overlay))
    assert decoded[:2] == b"\xff\xd8"  # JPEG magic bytes


# ---------------------------------------------------------------- ensemble


class _FakeMemberExplainer:
    target_layer_path = "fake.layer"

    def __init__(self, heatmap, fail=False):
        self._heatmap = heatmap
        self._fail = fail

    def explain(self, image, target_class_index):
        if self._fail:
            raise RuntimeError("boom")
        return self._heatmap


def _ensemble(models, explainers):
    ensemble = EnsembleGradCAMExplainer(models, ExplainabilityConfig(enabled=True))
    ensemble._explainers = dict(enumerate(explainers))
    return ensemble


def _member(variant):
    return {
        "model": None,
        "config": {"architecture": "efficientnetb4", "preprocessing_variant": variant},
    }


def test_ensemble_fuses_and_normalises_heatmaps():
    a = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    b = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    variants = {
        "original": np.zeros((2, 2, 3), np.uint8),
        "rgb_clahe": np.zeros((2, 2, 3), np.uint8),
    }
    ensemble = _ensemble(
        [_member("original"), _member("rgb_clahe")],
        [_FakeMemberExplainer(a), _FakeMemberExplainer(b)],
    )

    result = ensemble.explain(variants, target_class_index=2)

    assert result.target_class_index == 2
    assert len(result.model_heatmaps) == 2
    assert result.ensemble_heatmap.max() == pytest.approx(1.0)
    assert result.ensemble_heatmap[0, 1] == pytest.approx(1.0)
    assert result.ensemble_heatmap[1, 1] == pytest.approx(1.0)


def test_ensemble_skips_failing_member_but_fails_if_all_fail():
    variants = {"original": np.zeros((2, 2, 3), np.uint8)}
    ok = _FakeMemberExplainer(np.ones((2, 2), np.float32))
    bad = _FakeMemberExplainer(None, fail=True)

    partial = _ensemble([_member("original"), _member("original")], [ok, bad])
    assert len(partial.explain(variants, 0).model_heatmaps) == 1

    broken = _ensemble([_member("original")], [bad])
    with pytest.raises(ExplainabilityError):
        broken.explain(variants, 0)


# ---------------------------------------------------------------- API integration

flask = pytest.importorskip("flask")
pytest.importorskip("cv2")


@pytest.fixture
def server():
    from fundus_inference_server import FundusInferenceServer

    instance = FundusInferenceServer.__new__(FundusInferenceServer)
    instance.logger = SimpleNamespace(error=lambda *a, **k: None)
    instance.explanation_service = None
    return instance


def _classification(server, class_index=2):
    results = {
        "No DR": 0.05,
        "Mild DR": 0.05,
        "Moderate DR": 0.8,
        "Severe DR": 0.05,
        "Proliferative DR": 0.05,
        "predicted_class": "Moderate DR",
        "confidence": 0.8,
    }
    return server._build_classification_result(results)


def test_explain_flags_are_opt_in(server):
    app = flask.Flask(__name__)
    with app.test_request_context("/process"):
        assert server._get_explanation_flags() == (False, False)
    with app.test_request_context("/process?explain=true"):
        assert server._get_explanation_flags() == (True, False)
    with app.test_request_context("/process?explain=true&explain_per_model=true"):
        assert server._get_explanation_flags() == (True, True)
    # per-model has no effect without explain
    with app.test_request_context("/process?explain_per_model=true"):
        assert server._get_explanation_flags() == (False, False)


def test_run_explanation_success_maps_to_schema(server):
    captured = {}

    class _Service:
        def explain(self, variants, target_class_index, include_per_model=False):
            captured["index"] = target_class_index
            return RenderedExplanation(
                method="Grad-CAM",
                framework="OmniXAI",
                target_class_index=target_class_index,
                heatmap_overlay="overlay",
                heatmap="heat",
                explanation_time_ms=12.3,
                per_model=[
                    RenderedModelExplanation(
                        model_architecture="efficientnetb4",
                        preprocessing_variant="original",
                        target_layer="features.-1",
                        heatmap_overlay="o",
                        heatmap="h",
                    )
                ]
                if include_per_model
                else None,
            )

    server.explanation_service = _Service()
    result = server._run_explanation({}, _classification(server), per_model=True)
    payload = result.to_dict()

    assert captured["index"] == 2
    assert payload["status"] == "SUCCESS"
    assert payload["target_class_id"] == "DR_2"
    assert payload["heatmap_overlay"] == "overlay"
    assert payload["per_model"][0]["target_layer"] == "features.-1"


def test_run_explanation_failure_degrades_gracefully(server):
    class _Service:
        def explain(self, *args, **kwargs):
            raise ExplainabilityError("no heatmap")

    server.explanation_service = _Service()
    payload = server._run_explanation({}, _classification(server), False).to_dict()

    assert payload["status"] == "FAILED"
    assert payload["error"] == "no heatmap"
    assert payload["heatmap"] is None
