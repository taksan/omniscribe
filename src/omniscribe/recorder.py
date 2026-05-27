#!/usr/bin/env python3
"""
Meeting recorder: captures system audio (what you hear) and microphone
simultaneously, then writes a mixed WAV file (and optionally separate tracks).

Works on Linux with PulseAudio or PipeWire (pulse compatibility layer) by
recording from the default sink's "monitor" source. Also works on systems
where the user manually selects a loopback/monitor input device.

Usage:
    python recorder.py --list                       # list devices
    python recorder.py                              # record to meeting.wav
    python recorder.py -o talk.wav                  # custom output
    python recorder.py --mic 3 --system 7           # explicit device indices
    python recorder.py --separate                   # also save mic.wav/system.wav
    python recorder.py --mic-gain 1.0 --sys-gain 1.0

Stop recording with Ctrl+C.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from .config import Config, load_config, save_default_config
from . import tui as tui_module


SAMPLE_RATE = 48000
CHANNELS = 2
BLOCKSIZE = 1024
DTYPE = "float32"

# Below this peak amplitude, a block is treated as "silent" (~ -46 dBFS).
SILENCE_THRESHOLD = 0.005
METER_WIDTH = 20
METER_REFRESH_S = 0.25


def list_devices() -> None:
    print("PortAudio devices:")
    print(sd.query_devices())
    print()
    print(f"Default input : {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    monitor = detect_monitor_source()
    mic_src = detect_default_mic_source()
    if monitor:
        print(f"Pulse default monitor (system audio): {monitor}")
    if mic_src:
        print(f"Pulse default source (microphone):    {mic_src}")
    if shutil.which("pactl"):
        print()
        print("All Pulse/PipeWire sources (use any of these with --mic or --system):")
        try:
            out = subprocess.check_output(
                ["pactl", "list", "sources", "short"], text=True
            )
            print(out.rstrip())
        except subprocess.CalledProcessError:
            pass


def detect_monitor_source() -> str | None:
    """Return the PulseAudio/PipeWire monitor source name for the default sink."""
    if not shutil.which("pactl"):
        return None
    try:
        sink = subprocess.check_output(
            ["pactl", "get-default-sink"], text=True
        ).strip()
        if not sink:
            return None
        return f"{sink}.monitor"
    except subprocess.CalledProcessError:
        return None


def detect_default_mic_source() -> str | None:
    """Return the PulseAudio/PipeWire default input source name."""
    if not shutil.which("pactl"):
        return None
    try:
        src = subprocess.check_output(
            ["pactl", "get-default-source"], text=True
        ).strip()
        return src or None
    except subprocess.CalledProcessError:
        return None


def _pulse_source_channels(name: str, fallback: int = 1) -> int:
    """Read the channel count of a Pulse source from `pactl list sources short`."""
    if not shutil.which("pactl"):
        return fallback
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"], text=True
        )
    except subprocess.CalledProcessError:
        return fallback
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) >= 5 and cols[1] == name:
            # Format column looks like 's16le 1ch 32000Hz' or 's24le 2ch 48000Hz'.
            for tok in cols[4].split():
                if tok.endswith("ch"):
                    try:
                        return int(tok[:-2])
                    except ValueError:
                        pass
    return fallback


def _looks_like_pulse_source(name: str) -> bool:
    """Heuristic: does this string name a Pulse/PipeWire source (vs. PortAudio name)?"""
    n = name.lower()
    if ".monitor" in n or n.endswith("monitor"):
        return True
    # PulseAudio source names produced by alsa_card / pipewire start with these prefixes.
    return n.startswith(("alsa_input.", "alsa_output.", "input."))


def resolve_system_device(explicit: int | str | None) -> int | str:
    """Pick a device index/name that captures system audio.

    Returns either:
      * an int  -> PortAudio device index (used via sounddevice)
      * a str   -> PulseAudio/PipeWire source name (recorded via `parec`)
    """
    if explicit is not None:
        # Pulse source string: use parec path if available.
        if isinstance(explicit, str) and _looks_like_pulse_source(explicit):
            if not shutil.which("parec"):
                raise RuntimeError(
                    "Got a PulseAudio source name but 'parec' is not installed. "
                    "Install pulseaudio-utils (Debian/Ubuntu) or pass a PortAudio index."
                )
            return explicit
        return explicit

    # Preferred path on Linux: capture the default sink's monitor with parec.
    monitor = detect_monitor_source()
    if monitor and shutil.which("parec"):
        return monitor

    # PortAudio fallback: find a device whose name suggests a monitor/loopback.
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        n = dev["name"].lower()
        if "monitor" in n or "loopback" in n or "stereo mix" in n:
            return idx

    raise RuntimeError(
        "Could not auto-detect a system-audio (monitor/loopback) input. "
        "Install pulseaudio-utils for the 'parec' tool, or run with --list "
        "and pass --system <index>."
    )


def resolve_mic_device(explicit: int | str | None) -> int | str:
    """Pick a device index/name for the microphone.

    Returns either:
      * an int  -> PortAudio device index (used via sounddevice)
      * a str   -> PulseAudio/PipeWire source name (recorded via `parec`)
    """
    if explicit is not None:
        return explicit

    # Preferred path on Linux: capture the default Pulse input source with parec.
    src = detect_default_mic_source()
    if src and shutil.which("parec"):
        return src

    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        return default_in
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            return idx
    raise RuntimeError("No input (microphone) device found.")


def device_channels(device: int | str, desired: int = CHANNELS) -> int:
    info = sd.query_devices(device, "input")
    return min(desired, info["max_input_channels"]) or 1


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
    # ch < target and not the 1->2 case: pad with zeros
    pad = np.zeros((block.shape[0], target - ch), dtype=block.dtype)
    return np.concatenate([block, pad], axis=1)


def _meter_bar(peak: float, width: int = METER_WIDTH) -> str:
    """Render an ASCII level meter for a peak amplitude in [0, 1]."""
    peak = max(0.0, min(1.0, peak))
    db = -120.0 if peak <= 1e-6 else 20.0 * math.log10(peak)
    # Map -60 .. 0 dBFS to 0 .. width.
    frac = max(0.0, min(1.0, (db + 60.0) / 60.0))
    filled = int(round(frac * width))
    # Mark the last cell red-ish (>-3dB) by capitalizing for visibility.
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {db:5.1f}dB"


def _device_label(device, kind: str = "input") -> str:
    """Human-readable name for a sounddevice device index/string."""
    try:
        info = sd.query_devices(device, kind)
        return f"{info['name']} (idx {device if isinstance(device, int) else '?'})"
    except Exception:  # noqa: BLE001
        return str(device)


def _pulse_source_description(name: str) -> str:
    """Look up the friendly description of a Pulse source via `pactl`."""
    if not shutil.which("pactl"):
        return name
    try:
        out = subprocess.check_output(["pactl", "list", "sources"], text=True)
    except subprocess.CalledProcessError:
        return name
    block = []
    desc = None
    for line in out.splitlines():
        if line.startswith("Source #"):
            block = []
            desc = None
        block.append(line)
        s = line.strip()
        if s.startswith("Description:"):
            desc = s.split(":", 1)[1].strip()
        if s == f"Name: {name}" and desc:
            return f"{desc}"
        if s == f"Name: {name}":
            # Description may come after Name; scan forward.
            for follow in out.splitlines()[out.splitlines().index(line) + 1 :]:
                fs = follow.strip()
                if fs.startswith("Description:"):
                    return fs.split(":", 1)[1].strip()
                if fs.startswith("Source #"):
                    break
    return name


def _dbfs(x: float) -> float:
    return -120.0 if x <= 1e-6 else 20.0 * math.log10(x)


def _verdict_for_noise(rms_db: float) -> str:
    """Qualitative verdict for a microphone's idle noise level (in dBFS)."""
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


def _verdict_for_system(rms_db: float) -> str:
    if rms_db < -65:
        return "SILENT     (no system audio playing - that's expected if nothing's playing now)"
    if rms_db < -45:
        return "OK         (some audio detected)"
    return "LOUD       (consider lowering system volume - may clip)"


def check_inputs(mic_device, system_device, duration: float = 3.0) -> None:
    """Briefly capture from both inputs and print noise/level statistics."""
    use_parec_for_mic = isinstance(mic_device, str) and _looks_like_pulse_source(mic_device)
    use_parec_for_system = (
        isinstance(system_device, str) and _looks_like_pulse_source(system_device)
    )
    mic_ch = (
        _pulse_source_channels(mic_device, fallback=1)
        if use_parec_for_mic
        else device_channels(mic_device)
    )
    sys_ch = (
        _pulse_source_channels(system_device, fallback=CHANNELS)
        if use_parec_for_system
        else device_channels(system_device)
    )
    mic_name = (
        _pulse_source_description(mic_device)
        if use_parec_for_mic
        else _device_label(mic_device, "input")
    )
    sys_name = (
        _pulse_source_description(system_device)
        if use_parec_for_system
        else _device_label(system_device, "input")
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
    mic = (
        PulseMonitorRecorder(mic_device, mic_ch, "mic", stop)
        if use_parec_for_mic
        else StreamRecorder(mic_device, mic_ch, "mic", stop)
    )
    sysrec = (
        PulseMonitorRecorder(system_device, sys_ch, "system", stop)
        if use_parec_for_system
        else StreamRecorder(system_device, sys_ch, "system", stop)
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
        # Live meter while measuring.
        now = time.monotonic()
        if now - last_print >= METER_REFRESH_S:
            last_print = now
            mp = max((float(np.abs(b).max()) for b in mic_blocks[-3:]), default=0.0)
            sp = max((float(np.abs(b).max()) for b in sys_blocks[-3:]), default=0.0)
            remaining = max(0.0, t_end - now)
            print(
                f"\r  measuring {remaining:4.1f}s  "
                f"MIC {_meter_bar(mp)}  SYS {_meter_bar(sp)}\033[K",
                end="",
                flush=True,
            )

    stop.set()
    mic.join(timeout=2)
    sysrec.join(timeout=2)
    print()  # close the in-place line

    def _report(label: str, blocks: list[np.ndarray], verdict_fn) -> None:
        if not blocks:
            print(f"\n  {label}: NO SAMPLES CAPTURED - check device and permissions")
            return
        data = np.concatenate(blocks, axis=0).astype(np.float32)
        if data.ndim > 1:
            data = np.mean(np.abs(data), axis=1)
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data * data) + 1e-12))
        # Per-window RMS (50 ms windows), 10th percentile -> noise-floor estimate.
        win = max(1, int(0.05 * SAMPLE_RATE))
        n = (len(data) // win) * win
        if n > 0:
            wrms = np.sqrt(
                np.mean(data[:n].reshape(-1, win) ** 2, axis=1) + 1e-12
            )
            noise_floor = float(np.percentile(wrms, 10))
        else:
            noise_floor = rms
        dc = float(np.mean(data))

        print(f"\n  {label}")
        print(f"    Peak        : {_dbfs(peak):6.1f} dBFS  ({peak:.4f} linear)")
        print(f"    RMS         : {_dbfs(rms):6.1f} dBFS")
        print(f"    Noise floor : {_dbfs(noise_floor):6.1f} dBFS  (10th pct of 50ms windows)")
        print(f"    DC offset   : {dc:+.5f}")
        print(f"    Verdict     : {verdict_fn(_dbfs(noise_floor))}")
        if peak >= 0.99:
            print(f"    !! CLIPPING detected (peak hit 0 dBFS) - lower input gain")
        if abs(dc) > 0.01:
            print(f"    !! Large DC offset - hardware issue or codec quirk")

    _report("Mic", mic_blocks, _verdict_for_noise)
    _report("System", sys_blocks, _verdict_for_system)

    print()
    print("Tips:")
    print("  - Mic noise floor below -50 dBFS is great for Whisper transcription.")
    print("  - If noisy: try a different mic, move away from fans, or lower input gain")
    print(f"    with:  pactl set-source-volume @DEFAULT_SOURCE@ 80%")
    print("  - Re-run with:  python recorder.py --check")
    print()


def _play_alert_tone() -> None:
    """Play a short beep on the default output (non-blocking)."""

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
            # Also ring the terminal bell as a fallback.
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            try:
                sys.stdout.write("\a")
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass

    threading.Thread(target=_run, daemon=True, name="alert-tone").start()


class PulseMonitorRecorder(threading.Thread):
    """Capture a PulseAudio/PipeWire source via the `parec` subprocess.

    This bypasses PortAudio, which on PipeWire-Pulse typically exposes only a
    single generic 'pulse' device and cannot address per-source monitors by
    name.
    """

    def __init__(
        self,
        source_name: str,
        channels: int,
        name: str,
        stop_event: threading.Event,
        samplerate: int = SAMPLE_RATE,
        blocksize: int = BLOCKSIZE,
    ):
        super().__init__(daemon=True, name=f"rec-{name}")
        self.source_name = source_name
        self.channels = channels
        self.label = name
        self.stop_event = stop_event
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.error: Exception | None = None
        self._proc: subprocess.Popen | None = None

    def run(self) -> None:
        cmd = [
            "parec",
            f"--device={self.source_name}",
            "--format=float32le",
            f"--rate={self.samplerate}",
            f"--channels={self.channels}",
            "--raw",
            "--latency-msec=50",
            "--client-name=local-transcriber",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
            )
            bytes_per_sample = 4  # float32
            bytes_per_block = self.blocksize * self.channels * bytes_per_sample
            assert self._proc.stdout is not None
            while not self.stop_event.is_set():
                data = self._proc.stdout.read(bytes_per_block)
                if not data:
                    break
                if len(data) % (self.channels * bytes_per_sample) != 0:
                    # Drop a partial trailing sample, shouldn't normally happen
                    data = data[: -(len(data) % (self.channels * bytes_per_sample))]
                arr = np.frombuffer(data, dtype=np.float32).reshape(-1, self.channels)
                try:
                    self.queue.put(arr.copy(), timeout=1.0)
                except queue.Full:
                    pass
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.stop_event.set()
        finally:
            if self._proc is not None:
                if self._proc.poll() is None:
                    self._proc.terminate()
                    try:
                        self._proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self._proc.kill()
                if self._proc.returncode not in (0, None, -signal.SIGTERM):
                    err = b""
                    if self._proc.stderr is not None:
                        try:
                            err = self._proc.stderr.read() or b""
                        except Exception:  # noqa: BLE001
                            pass
                    if err:
                        self.error = RuntimeError(
                            f"parec exited {self._proc.returncode}: {err.decode(errors='replace').strip()}"
                        )


class StreamRecorder(threading.Thread):
    """Reads from one input device and pushes blocks into a queue."""

    def __init__(self, device, channels: int, name: str, stop_event: threading.Event):
        super().__init__(daemon=True, name=f"rec-{name}")
        self.device = device
        self.channels = channels
        self.label = name
        self.stop_event = stop_event
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.error: Exception | None = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[{self.label}] {status}", file=sys.stderr)
        # Copy because PortAudio reuses the buffer.
        self.queue.put(indata.copy())

    def run(self) -> None:
        try:
            with sd.InputStream(
                device=self.device,
                channels=self.channels,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
                dtype=DTYPE,
                callback=self._callback,
            ):
                while not self.stop_event.is_set():
                    self.stop_event.wait(0.1)
        except Exception as e:  # noqa: BLE001
            self.error = e
            self.stop_event.set()


def record(
    output: Path,
    mic_device,
    system_device,
    mic_gain: float,
    sys_gain: float,
    separate: bool,
    transcribe: bool = False,
    whisper_model: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    language: str | None = None,
    chunk_seconds: float = 10.0,
    mic_label: str = "You",
    system_label: str = "Them",
    silence_alert_seconds: float = 30.0,
    show_meters: bool = True,
    beam_size: int = 5,
    condition_on_previous_text: bool = True,
    initial_prompt: str | None = None,
    vad_min_silence_ms: int = 500,
    tui: bool = False,
) -> None:
    use_parec_for_mic = isinstance(mic_device, str) and _looks_like_pulse_source(mic_device)
    use_parec_for_system = isinstance(system_device, str) and _looks_like_pulse_source(system_device)

    if use_parec_for_mic:
        mic_ch = _pulse_source_channels(mic_device, fallback=1)
    else:
        mic_ch = device_channels(mic_device)

    if use_parec_for_system:
        sys_ch = _pulse_source_channels(system_device, fallback=CHANNELS)
    else:
        sys_ch = device_channels(system_device)

    mic_name = (
        _pulse_source_description(mic_device)
        if use_parec_for_mic
        else _device_label(mic_device, "input")
    )
    sys_name = (
        _pulse_source_description(system_device)
        if use_parec_for_system
        else _device_label(system_device, "input")
    )

    # Initialize TUI if requested
    tui_instance = None
    if tui:
        tui_instance = tui_module.create_tui()
        tui_instance.set_devices(mic_name, sys_name, mic_gain, sys_gain)
        tui_instance.set_transcription_config(
            model=whisper_model,
            device=whisper_device,
            language=language,
            initial_prompt=initial_prompt,
            gpu_detected=False,  # Will be updated by transcriber
        )
        tui_instance.start()
    else:
        print()
        print("=" * 72)
        print("Recording from:")
        print(
            f"  Mic    : {mic_name}  ({mic_ch}ch"
            + (", via parec" if use_parec_for_mic else "")
            + ")"
        )
        print(
            f"  System : {sys_name}  ({sys_ch}ch"
            + (", via parec" if use_parec_for_system else "")
            + ")"
        )
        print()
        print(f"Output WAV : {output}")
        if separate:
            print(f"             + {output.with_name(output.stem + '.mic.wav')}")
            print(f"             + {output.with_name(output.stem + '.system.wav')}")
        if silence_alert_seconds > 0:
            print(f"Silence alert: after {int(silence_alert_seconds)}s of silence on a source")
        print("=" * 72)
        print("Recording... press Ctrl+C to stop (Ctrl+C twice to force quit).")
    print()

    stop = threading.Event()
    interrupt_count = {"n": 0}

    def _on_signal(signum, _frame):
        interrupt_count["n"] += 1
        if interrupt_count["n"] == 1:
            sig = "Ctrl+C" if signum == signal.SIGINT else f"signal {signum}"
            print(
                f"\n{sig} received. Stopping recording, draining audio buffers"
                + (", and finalizing transcript" if transcribe else "")
                + "... (press Ctrl+C again to force quit)"
            )
            stop.set()
        else:
            print("\nForce quit requested. Files may be truncated.", file=sys.stderr)
            # Restore default handler so a third Ctrl+C kills us instantly.
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if use_parec_for_mic:
        mic = PulseMonitorRecorder(mic_device, mic_ch, "mic", stop)
    else:
        mic = StreamRecorder(mic_device, mic_ch, "mic", stop)
    if use_parec_for_system:
        sysrec = PulseMonitorRecorder(system_device, sys_ch, "system", stop)
    else:
        sysrec = StreamRecorder(system_device, sys_ch, "system", stop)

    output.parent.mkdir(parents=True, exist_ok=True)
    mixed = sf.SoundFile(
        str(output), mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16"
    )
    mic_file = sys_file = None
    if separate:
        mic_file = sf.SoundFile(
            str(output.with_name(output.stem + ".mic.wav")),
            mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16",
        )
        sys_file = sf.SoundFile(
            str(output.with_name(output.stem + ".system.wav")),
            mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16",
        )

    transcriber = None
    if transcribe:
        from .transcriber import LiveTranscriber

        transcript_path = output.with_suffix(".txt")
        # Set up TUI callback if enabled
        on_output = None
        if tui_instance:
            on_output = tui_instance.add_transcript

        transcriber = LiveTranscriber(
            output_path=transcript_path,
            model_name=whisper_model,
            device=whisper_device,
            compute_type=whisper_compute_type,
            language=language,
            source_sample_rate=SAMPLE_RATE,
            chunk_seconds=chunk_seconds,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            vad_min_silence_ms=vad_min_silence_ms,
            on_output=on_output,
        )
        transcriber.start()
        if tui_instance:
            tui_instance.state.gpu_detected = getattr(transcriber, '_gpu_detected', False)
        else:
            print(f"Transcript: {transcript_path}")

    mic.start()
    sysrec.start()

    total_frames = 0
    last_render = 0.0
    mic_peak_acc = 0.0
    sys_peak_acc = 0.0
    now0 = time.monotonic()
    last_signal_time = {"mic": now0, "system": now0}
    alerted = {"mic": False, "system": False}

    try:
        while not stop.is_set() or not mic.queue.empty() or not sysrec.queue.empty():
            try:
                m = mic.queue.get(timeout=0.2)
            except queue.Empty:
                m = None
            try:
                s = sysrec.queue.get_nowait()
            except queue.Empty:
                s = None

            if m is None and s is None:
                if stop.is_set():
                    break
                continue

            # Align block lengths (both streams use the same blocksize, so this
            # only trims rare edge blocks at start/stop).
            if m is not None and s is not None:
                n = min(len(m), len(s))
                m = m[:n]
                s = s[:n]
            elif m is None:
                m = np.zeros((len(s), mic_ch), dtype=DTYPE)
            elif s is None:
                s = np.zeros((len(m), sys_ch), dtype=DTYPE)

            m2 = to_stereo(m) * mic_gain
            s2 = to_stereo(s) * sys_gain
            mix = np.clip(m2 + s2, -1.0, 1.0)

            mixed.write(mix)
            if mic_file is not None:
                mic_file.write(m2)
            if sys_file is not None:
                sys_file.write(s2)
            if transcriber is not None:
                transcriber.feed(mic_label, m2)
                transcriber.feed(system_label, s2)

            total_frames += len(mix)
            seconds = total_frames / SAMPLE_RATE

            # Track peak amplitudes per source for the meter / silence detector.
            mic_peak_block = float(np.abs(m2).max()) if m2.size else 0.0
            sys_peak_block = float(np.abs(s2).max()) if s2.size else 0.0
            mic_peak_acc = max(mic_peak_acc, mic_peak_block)
            sys_peak_acc = max(sys_peak_acc, sys_peak_block)

            # Update TUI audio levels
            if tui_instance:
                mic_db = 20 * np.log10(max(mic_peak_acc, 1e-10))
                sys_db = 20 * np.log10(max(sys_peak_acc, 1e-10))
                tui_instance.update_audio_levels(mic_db, sys_db)

            now = time.monotonic()
            if mic_peak_block > SILENCE_THRESHOLD:
                last_signal_time["mic"] = now
                if alerted["mic"]:
                    print(f"\n  >> mic audio resumed at {seconds:6.1f}s", flush=True)
                    alerted["mic"] = False
            if sys_peak_block > SILENCE_THRESHOLD:
                last_signal_time["system"] = now
                if alerted["system"]:
                    print(f"\n  >> system audio resumed at {seconds:6.1f}s", flush=True)
                    alerted["system"] = False

            if silence_alert_seconds > 0 and not stop.is_set():
                for src in ("mic", "system"):
                    if alerted[src]:
                        continue
                    if now - last_signal_time[src] >= silence_alert_seconds:
                        alerted[src] = True
                        label = "microphone" if src == "mic" else "system audio"
                        print(
                            f"\n  !! ALERT: no {label} detected for "
                            f"{int(silence_alert_seconds)}s",
                            flush=True,
                        )
                        _play_alert_tone()

            if (
                show_meters
                and not tui_instance  # TUI has its own meters
                and not stop.is_set()
                and (now - last_render) >= METER_REFRESH_S
            ):
                last_render = now
                mm, ss = divmod(int(seconds), 60)
                hh, mm = divmod(mm, 60)
                line = (
                    f"  {hh:02d}:{mm:02d}:{ss:02d}  "
                    f"MIC {_meter_bar(mic_peak_acc)}  "
                    f"SYS {_meter_bar(sys_peak_acc)}"
                )
                # \033[K clears to end-of-line so shorter lines don't leave artefacts.
                print(f"\r{line}\033[K", end="", flush=True)
                # Decay accumulators so the meter follows the signal envelope.
                mic_peak_acc *= 0.4
                sys_peak_acc *= 0.4
    finally:
        stop.set()
        if tui_instance:
            tui_instance.stop()
        else:
            print()  # break out of the in-place "HH:MM:SS recorded" line

        print("  [1/3] Stopping audio capture threads...", flush=True)
        mic.join(timeout=2)
        sysrec.join(timeout=2)

        print("  [2/3] Closing WAV file(s)...", flush=True)
        mixed.close()
        if mic_file is not None:
            mic_file.close()
        if sys_file is not None:
            sys_file.close()

        if transcriber is not None:
            print(
                "  [3/3] Finalizing transcript "
                "(transcribing remaining audio, this may take a few seconds)...",
                flush=True,
            )
            transcriber.stop()
        else:
            print("  [3/3] Done.", flush=True)

        if mic.error:
            print(f"Mic stream error: {mic.error}", file=sys.stderr)
        if sysrec.error:
            print(f"System stream error: {sysrec.error}", file=sys.stderr)

        seconds = total_frames / SAMPLE_RATE
        mm, ss = divmod(int(seconds), 60)
        hh, mm = divmod(mm, 60)
        print(f"\nSaved {output}  ({hh:02d}:{mm:02d}:{ss:02d}, {seconds:.1f}s)")
        if transcriber is not None:
            print(f"Saved transcript {output.with_suffix('.txt')}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                 help="path to config JSON file (default: ~/.config/local-transcriber/config.json or ./config.json)")
    p.add_argument("--init-config", action="store_true",
                 help="create a default config.json file and exit")
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--check", action="store_true",
                   help="run a short mic/system noise check and exit")
    p.add_argument("--check-seconds", type=float, default=3.0,
                   help="duration of the --check measurement window")
    default_name = f"meeting-{dt.datetime.now():%Y%m%d-%H%M%S}.wav"
    p.add_argument("-o", "--output", type=Path, default=Path("meetings") / default_name)
    p.add_argument("--mic", help="microphone device (index or name)")
    p.add_argument("--system", help="system audio device (monitor/loopback index or name)")
    p.add_argument("--mic-gain", type=float, default=1.0)
    p.add_argument("--sys-gain", type=float, default=1.0)
    p.add_argument("--separate", action="store_true", help="also save mic.wav and system.wav")
    p.add_argument("--transcribe", action="store_true", help="live transcribe with Whisper")
    p.add_argument("--whisper-model", default="small",
                   help="faster-whisper model name (tiny, base, small, medium, large-v3, ...)")
    p.add_argument("--whisper-device", default="cpu", help="cpu|cuda|auto")
    p.add_argument("--whisper-compute-type", default="int8",
                   help="int8|int8_float16|float16|float32|auto")
    p.add_argument("--language", default=None, help="language code, e.g. en, pt (auto-detect if omitted)")
    p.add_argument("--chunk-seconds", type=float, default=10.0,
                   help="audio chunk length fed to Whisper (longer = better context, more latency)")
    p.add_argument("--beam-size", type=int, default=5,
                   help="Whisper beam search width (1=greedy, 5=default, 10=slower/better)")
    p.add_argument("--no-context", action="store_true",
                   help="disable conditioning on previously transcribed text "
                        "(use if you see hallucination / repetition)")
    p.add_argument("--initial-prompt", default=None,
                   help="vocabulary hint string fed before each chunk; great for proper "
                        "nouns, names, jargon, e.g. 'Reuni\u00e3o sobre Kubernetes, Ansible, Tiago'")
    p.add_argument("--vad-min-silence-ms", type=int, default=500,
                   help="silence (ms) required to split speech in the VAD filter")
    p.add_argument("--mic-label", default="You")
    p.add_argument("--system-label", default="Them")
    p.add_argument("--silence-alert-seconds", type=float, default=30.0,
                   help="play an alert tone after this many seconds of silence "
                        "on a source (0 to disable)")
    p.add_argument("--no-meters", action="store_true",
                   help="disable the live ASCII level meters")
    p.add_argument("--tui", action="store_true",
                   help="use fancy terminal UI with live transcription display")
    args = p.parse_args()

    # Handle --init-config: create default config and exit
    if args.init_config:
        path = save_default_config(args.config)
        print(f"Created default config file: {path}")
        print("Edit this file to customize your default settings.")
        return 0

    # Load config from file and merge with CLI args
    config = load_config(args.config)
    config.merge_with_args(args)

    if args.list:
        list_devices()
        return 0

    def _coerce(v):
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return v

    if args.check:
        mic_dev = resolve_mic_device(_coerce(config.mic))
        sys_dev = resolve_system_device(_coerce(config.system))
        check_inputs(mic_dev, sys_dev, duration=config.check_seconds)
        return 0

    mic_dev = resolve_mic_device(_coerce(config.mic))
    sys_dev = resolve_system_device(_coerce(config.system))

    # Build output path
    output = Path(config.output) if config.output else args.output

    try:
        record(
            output=output,
            mic_device=mic_dev,
            system_device=sys_dev,
            mic_gain=config.mic_gain,
            sys_gain=config.sys_gain,
            separate=config.separate,
            transcribe=config.transcribe,
            whisper_model=config.whisper_model,
            whisper_device=config.whisper_device,
            whisper_compute_type=config.whisper_compute_type,
            language=config.language,
            chunk_seconds=config.chunk_seconds,
            mic_label=config.mic_label,
            system_label=config.system_label,
            silence_alert_seconds=config.silence_alert_seconds,
            show_meters=config.show_meters,
            beam_size=config.beam_size,
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt=config.initial_prompt,
            vad_min_silence_ms=config.vad_min_silence_ms,
            tui=args.tui,
        )
    except KeyboardInterrupt:
        # Force-quit path: finally block in record() already ran best-effort.
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
