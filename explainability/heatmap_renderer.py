"""Presentation of heatmaps: colorisation, overlay and base64 JPEG encoding."""

import base64

import cv2
import numpy as np


class HeatmapRenderer:
    """Turns numerical heatmaps into displayable, base64-encoded JPEG images."""

    def __init__(self, alpha: float = 0.4, jpeg_quality: int = 90):
        self.alpha = alpha
        self.jpeg_quality = jpeg_quality

    @staticmethod
    def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
        if image.dtype == np.uint8:
            return image
        image = image.astype(np.float32)
        if image.max() <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def colorize(self, heatmap: np.ndarray) -> np.ndarray:
        """(H, W) float [0, 1] -> (H, W, 3) uint8 RGB (JET colormap)."""
        heat_u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        return cv2.cvtColor(
            cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB
        )

    def overlay(self, base_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        """Blend the colorised heatmap over ``base_image`` (RGB)."""
        base = self._to_uint8_rgb(base_image)
        if heatmap.shape[:2] != base.shape[:2]:
            heatmap = cv2.resize(
                heatmap, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return cv2.addWeighted(
            base, 1.0 - self.alpha, self.colorize(heatmap), self.alpha, 0.0
        )

    def encode_base64_jpeg(self, rgb_image: np.ndarray) -> str:
        """(H, W, 3) uint8 RGB -> base64 JPEG string."""
        ok, buffer = cv2.imencode(
            ".jpg",
            cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode heatmap image")
        return base64.b64encode(buffer).decode("utf-8")
