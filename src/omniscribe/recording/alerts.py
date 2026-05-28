"""Recording alerts and notifications."""

from __future__ import annotations

import sys
import threading

import numpy as np
import sounddevice as sd


def play_alert_tone() -> None:
    def _run():
        try:
            sr = 44100
            dur = 0.35
            t = np.linspace(0, dur, int(dur * sr), endpoint=False, dtype=np.float32)
            tone = 0.25 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
            fade = int(0.02 * sr)
            tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            sd.play(tone, sr, blocking=True)
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_run, daemon=True, name="alert-tone").start()
