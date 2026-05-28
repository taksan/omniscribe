"""Level meters and dBFS helpers."""

from __future__ import annotations

import math

from .constants import METER_WIDTH


def dbfs(x: float) -> float:
    return -120.0 if x <= 1e-6 else 20.0 * math.log10(x)


def meter_bar(peak: float, width: int = METER_WIDTH) -> str:
    peak = max(0.0, min(1.0, peak))
    db = dbfs(peak)
    frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {db:5.1f}dB"


def verdict_for_noise(rms_db: float) -> str:
    if rms_db < -60:
        return "EXCELLENT  (essentially silent, mic might be muted!)"
    if rms_db < -55:
        return "GREAT      (very clean signal)"
    if rms_db < -45:
        return "OK         (normal room ambience)"
    if rms_db < -35:
        return "NOISY      (fan/HVAC/keyboard; consider noise suppression or moving the mic)"
    if rms_db < -25:
        return "VERY NOISY (something is wrong - constant hum, mic too close to a fan, gain too high)"
    return "TERRIBLE   (clipping or extreme noise - this will wreck transcription)"


def verdict_for_system(rms_db: float) -> str:
    if rms_db < -65:
        return "SILENT     (no system audio playing - that's expected if nothing's playing now)"
    if rms_db < -45:
        return "OK         (some audio detected)"
    return "LOUD       (consider lowering system volume - may clip)"
