"""Audio channel layout helpers."""

from __future__ import annotations

import numpy as np

from .constants import CHANNELS


def to_stereo(block: np.ndarray, target: int = CHANNELS) -> np.ndarray:
    if block.ndim == 1:
        block = block[:, None]
    ch = block.shape[1]
    if ch == target:
        return block
    if ch == 1 and target == 2:
        return np.repeat(block, 2, axis=1)
    if ch > target:
        return block[:, :target]
    pad = np.zeros((block.shape[0], target - ch), dtype=block.dtype)
    return np.concatenate([block, pad], axis=1)
