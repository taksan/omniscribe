"""Pre-flight microphone and system audio level check."""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from omniscribe.ui.tui import OmniScribeTUI

from .capture import PulseMonitorRecorder, StreamRecorder
from .constants import CHANNELS, METER_REFRESH_S, SAMPLE_RATE
from .devices import (
    device_channels,
    device_label,
    looks_like_pulse_source,
    pulse_source_channels,
    pulse_source_description,
)
from .meters import dbfs, meter_bar, verdict_for_noise, verdict_for_system
from .tui_callbacks import make_audio_level_callback


def check_inputs(
    mic_device,
    system_device,
    duration: float = 3.0,
    tui: OmniScribeTUI | None = None,
) -> None:
    use_parec_for_mic = isinstance(mic_device, str) and looks_like_pulse_source(mic_device)
    use_parec_for_system = (
        isinstance(system_device, str) and looks_like_pulse_source(system_device)
    )
    mic_ch = (
        pulse_source_channels(mic_device, fallback=1)
        if use_parec_for_mic
        else device_channels(mic_device)
    )
    sys_ch = (
        pulse_source_channels(system_device, fallback=CHANNELS)
        if use_parec_for_system
        else device_channels(system_device)
    )
    mic_name = (
        pulse_source_description(mic_device)
        if use_parec_for_mic
        else device_label(mic_device, "input")
    )
    sys_name = (
        pulse_source_description(system_device)
        if use_parec_for_system
        else device_label(system_device, "input")
    )

    print()
    print("=" * 72)
    print(f"Audio input check  ({duration:.1f}s)")
    print(f"  Mic    : {mic_name}  ({mic_ch}ch{', via parec' if use_parec_for_mic else ''})")
    print(f"  System : {sys_name}  ({sys_ch}ch{', via parec' if use_parec_for_system else ''})")
    print("=" * 72)
    print("Stay quiet for a moment so we can measure background noise...")
    print()

    stop = threading.Event()
    mic_callback = make_audio_level_callback(tui, "mic") if tui else None
    sys_callback = make_audio_level_callback(tui, "system") if tui else None

    mic = (
        PulseMonitorRecorder(mic_device, mic_ch, "mic", stop, tui_callback=mic_callback)
        if use_parec_for_mic
        else StreamRecorder(mic_device, mic_ch, "mic", stop, tui_callback=mic_callback)
    )
    sysrec = (
        PulseMonitorRecorder(system_device, sys_ch, "system", stop, tui_callback=sys_callback)
        if use_parec_for_system
        else StreamRecorder(system_device, sys_ch, "system", stop, tui_callback=sys_callback)
    )
    mic.start()
    sysrec.start()

    mic_blocks: list[np.ndarray] = []
    sys_blocks: list[np.ndarray] = []
    t_end = time.monotonic() + duration
    last_print = 0.0
    while time.monotonic() < t_end:
        try:
            mic_blocks.append(mic.queue.get(timeout=0.1))
        except queue.Empty:
            pass
        try:
            sys_blocks.append(sysrec.queue.get_nowait())
        except queue.Empty:
            pass
        now = time.monotonic()
        if now - last_print >= METER_REFRESH_S:
            last_print = now
            mp = max((float(np.abs(b).max()) for b in mic_blocks[-3:]), default=0.0)
            sp = max((float(np.abs(b).max()) for b in sys_blocks[-3:]), default=0.0)
            remaining = max(0.0, t_end - now)
            print(
                f"\r  measuring {remaining:4.1f}s  "
                f"MIC {meter_bar(mp)}  SYS {meter_bar(sp)}\033[K",
                end="",
                flush=True,
            )

    stop.set()
    mic.join(timeout=2)
    sysrec.join(timeout=2)
    print()

    def _report(label: str, blocks: list[np.ndarray], verdict_fn) -> None:
        if not blocks:
            print(f"\n  {label}: NO SAMPLES CAPTURED - check device and permissions")
            return
        data = np.concatenate(blocks, axis=0).astype(np.float32)
        if data.ndim > 1:
            data = np.mean(np.abs(data), axis=1)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data * data) + 1e-12))
        win = max(1, int(0.05 * SAMPLE_RATE))
        n = (len(data) // win) * win
        if n > 0:
            wrms = np.sqrt(np.mean(data[:n].reshape(-1, win) ** 2, axis=1) + 1e-12)
            noise_floor = float(np.percentile(wrms, 10))
        else:
            noise_floor = rms
        dc = float(np.mean(data))

        print(f"\n  {label}")
        print(f"    Peak        : {dbfs(peak):6.1f} dBFS  ({peak:.4f} linear)")
        print(f"    RMS         : {dbfs(rms):6.1f} dBFS")
        print(f"    Noise floor : {dbfs(noise_floor):6.1f} dBFS  (10th pct of 50ms windows)")
        print(f"    DC offset   : {dc:+.5f}")
        print(f"    Verdict     : {verdict_fn(dbfs(noise_floor))}")
        if peak >= 0.99:
            print("    !! CLIPPING detected (peak hit 0 dBFS) - lower input gain")
        if abs(dc) > 0.01:
            print("    !! Large DC offset - hardware issue or codec quirk")

    _report("Mic", mic_blocks, verdict_for_noise)
    _report("System", sys_blocks, verdict_for_system)

    print()
    print("Tips:")
    print("  - Mic noise floor below -50 dBFS is great for Whisper transcription.")
    print("  - If noisy: try a different mic, move away from fans, or lower input gain")
    print("    with:  pactl set-source-volume @DEFAULT_SOURCE@ 80%")
    print("  - Re-run with:  python recorder.py --check")
    print()
