"""Audio preprocessing for Whisper."""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy.signal import resample_poly

from .constants import WHISPER_SR


def to_mono16k(block: np.ndarray, src_sr: int) -> np.ndarray:
    """Down-mix to mono and resample to 16 kHz float32."""
    if block.ndim == 2:
        mono = block.mean(axis=1)
    else:
        mono = block
    mono = mono.astype(np.float32, copy=False)
    if src_sr == WHISPER_SR:
        return mono
    g = gcd(src_sr, WHISPER_SR)
    up = WHISPER_SR // g
    down = src_sr // g
    return resample_poly(mono, up, down).astype(np.float32, copy=False)
