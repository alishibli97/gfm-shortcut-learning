#!/usr/bin/env python3
"""Model-specific RGB preprocessing for frozen GFM feature extraction.

All loaders first convert the source image to 8-bit RGB with Pillow.  This is
intentional for xView2 RGB imagery and for the established BRIGHT protocol,
where the single-band post-event SAR image is repeated across three channels.

The values below follow the released model configurations.  Prithvi is the
only special case: its released RGB bands are HLS reflectance bands, so an
8-bit RGB value is linearly range-matched from [0, 255] to [0, 10000] before
applying the selected HLS RED/GREEN/BLUE statistics.  This cannot recover
physical reflectance from display RGB; the approximation is recorded in every
output metadata file and should be stated as a limitation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessingProfile:
    name: str
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    input_scale: float
    source_range: str
    range_before_standardization: str
    notes: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


PROFILES = {
    # Prithvi config order is BLUE, GREEN, RED.  The extractor requests
    # RED, GREEN, BLUE, so the corresponding statistics are reversed.
    "prithvi": PreprocessingProfile(
        name="prithvi_rgb8_to_hls_reflectance_then_standardize",
        mean=(1433.0, 1342.0, 1087.0),
        std=(2178.0, 2179.0, 2248.0),
        input_scale=10000.0 / 255.0,
        source_range="Pillow RGB uint8 [0,255]",
        range_before_standardization="linearly mapped to [0,10000]",
        notes=(
            "Range matching for non-HLS RGB; it does not reconstruct physical "
            "HLS reflectance. Statistics correspond to requested RED/GREEN/BLUE."
        ),
    ),
    "terramind": PreprocessingProfile(
        name="terramind_rgb_standardize",
        mean=(87.271, 80.931, 66.667),
        std=(58.767, 47.663, 42.631),
        input_scale=1.0,
        source_range="Pillow RGB uint8 [0,255]",
        range_before_standardization="[0,255]",
        notes="TerraMind released RGB modality statistics.",
    ),
    "dofa": PreprocessingProfile(
        name="dofa_rgb_imagenet_standardize",
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        input_scale=1.0,
        source_range="Pillow RGB uint8 [0,255]",
        range_before_standardization="[0,255]",
        notes="DOFA released RGB/NAIP preprocessing (ImageNet statistics).",
    ),
    "dinov3": PreprocessingProfile(
        name="dinov3_lvd_imagenet_standardize",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        input_scale=1.0 / 255.0,
        source_range="Pillow RGB uint8 [0,255]",
        range_before_standardization="[0,1]",
        notes="Official DINOv3 transform for LVD-1689M weights.",
    ),
}


def canonical_model_name(model_name: str) -> str:
    if model_name.startswith("dofa"):
        return "dofa"
    if model_name not in PROFILES:
        raise ValueError(
            f"No preprocessing profile for {model_name!r}; "
            f"available profiles: {sorted(PROFILES)}"
        )
    return model_name


class RGBPreprocessor:
    """Convert a Pillow RGB image array to a standardized CHW float tensor."""

    def __init__(self, model_name: str):
        self.profile = PROFILES[canonical_model_name(model_name)]
        self.mean = np.asarray(self.profile.mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(self.profile.std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, rgb_uint8: np.ndarray) -> torch.Tensor:
        if rgb_uint8.ndim != 3 or rgb_uint8.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB array, got {rgb_uint8.shape}")

        arr = rgb_uint8.astype(np.float32, copy=False)
        arr = arr * self.profile.input_scale
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1)).copy()
        tensor = torch.from_numpy(arr)

        if not torch.isfinite(tensor).all():
            raise RuntimeError("Preprocessing produced NaN or Inf")

        return tensor

    def metadata(self) -> Dict[str, object]:
        return self.profile.to_dict()
