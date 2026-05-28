"""Audio capture threads (PortAudio and PulseAudio/parec)."""

from __future__ import annotations

import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

from .constants import BLOCKSIZE, DTYPE, SAMPLE_RATE


class PulseMonitorRecorder(threading.Thread):
    """Capture a PulseAudio/PipeWire source via the `parec` subprocess."""

    def __init__(
        self,
        source_name: str,
        channels: int,
        name: str,
        stop_event: threading.Event,
        samplerate: int = SAMPLE_RATE,
        blocksize: int = BLOCKSIZE,
        tui_callback: Callable[[np.ndarray], None] | None = None,
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
        self.tui_callback = tui_callback

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
            bytes_per_sample = 4
            bytes_per_block = self.blocksize * self.channels * bytes_per_sample
            assert self._proc.stdout is not None
            while not self.stop_event.is_set():
                data = self._proc.stdout.read(bytes_per_block)
                if not data:
                    break
                if len(data) % (self.channels * bytes_per_sample) != 0:
                    data = data[: -(len(data) % (self.channels * bytes_per_sample))]
                arr = np.frombuffer(data, dtype=np.float32).reshape(-1, self.channels)
                if self.tui_callback is not None:
                    self.tui_callback(arr)
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
    """Reads from one PortAudio input device and pushes blocks into a queue."""

    def __init__(
        self,
        device,
        channels: int,
        name: str,
        stop_event: threading.Event,
        tui_callback: Callable[[np.ndarray], None] | None = None,
    ):
        super().__init__(daemon=True, name=f"rec-{name}")
        self.device = device
        self.channels = channels
        self.label = name
        self.stop_event = stop_event
        self.queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self.error: Exception | None = None
        self.tui_callback = tui_callback

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[{self.label}] {status}", file=sys.stderr)
        if self.tui_callback is not None:
            self.tui_callback(indata)
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
