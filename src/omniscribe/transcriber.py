"""Live Whisper transcription.

Each audio source (e.g. "You", "Them") has its own ring buffer. A background
worker thread waits until a buffer has at least ``chunk_seconds`` of audio,
then runs faster-whisper on it, prints the result, and appends it to a text
file. Sources are processed independently so partial results from one do not
block the other.
"""

from __future__ import annotations

import datetime as dt
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

def _preload_cuda_libs() -> list[str]:
    """Preload cuBLAS + cuDNN from the nvidia-*-cu12 pip wheels so that
    ctranslate2's later dlopen("libcublas.so.12") / dlopen("libcudnn*.so.9")
    resolves without requiring LD_LIBRARY_PATH to be set.
    Returns the list of libraries successfully loaded.
    """
    import ctypes
    import glob
    import site
    import sys
    import os

    loaded: list[str] = []
    search_roots: list[str] = []

    # Add standard site-packages locations
    for base in {sys.prefix, *site.getsitepackages(), site.getusersitepackages()}:
        if base:
            search_roots.append(base)

    # Also check inside the package paths for nvidia libraries
    for site_path in site.getsitepackages():
        nvidia_base = Path(site_path) / "nvidia"
        if nvidia_base.exists():
            search_roots.append(str(nvidia_base))

    # Check common installation paths
    common_paths = [
        Path.home() / ".local" / "lib" / "python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.executable).parent.parent / "lib" / "python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for p in common_paths:
        if p.exists():
            search_roots.append(str(p))

    patterns = [
        "**/nvidia/cublas/lib/libcublas.so.12",
        "**/nvidia/cublas/lib/libcublasLt.so.12",
        # Load graph + base first, then dependents.
        "**/nvidia/cudnn/lib/libcudnn_graph.so.9",
        "**/nvidia/cudnn/lib/libcudnn.so.9",
        "**/nvidia/cudnn/lib/libcudnn_ops.so.9",
        "**/nvidia/cudnn/lib/libcudnn_cnn.so.9",
        "**/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
        # Also try direct patterns
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cudnn/lib/libcudnn*.so.9",
    ]

    seen: set[str] = set()
    for root in search_roots:
        for pat in patterns:
            for path in glob.glob(f"{root}/{pat}", recursive=True):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(path)
                except OSError as e:
                    # Log but don't fail - some libs might fail but others work
                    pass

    return loaded


def _check_cuda_available() -> tuple[bool, str]:
    """Check if CUDA is available and return diagnostics."""
    import subprocess
    import shutil

    checks = []

    # Check nvidia-smi
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                checks.append(f"GPU detected: {result.stdout.strip()}")
            else:
                checks.append("nvidia-smi found but failed to query GPU")
        except Exception as e:
            checks.append(f"nvidia-smi error: {e}")
    else:
        checks.append("nvidia-smi not found (NVIDIA driver not installed?)")

    # Check for nvidia pip packages
    try:
        import pkg_resources
        nvidia_packages = [d for d in pkg_resources.working_set if d.project_name.startswith("nvidia-")]
        if nvidia_packages:
            checks.append(f"NVIDIA pip packages: {', '.join(p.project_name for p in nvidia_packages)}")
        else:
            checks.append("No nvidia-* pip packages found (try: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)")
    except Exception as e:
        checks.append(f"Could not check pip packages: {e}")

    cuda_available = any("GPU detected" in c for c in checks)
    return cuda_available, "\n".join(checks)


try:
    from scipy.signal import resample_poly
except ImportError as e:  # pragma: no cover
    raise SystemExit("scipy is required for transcription. pip install scipy") from e

try:
    from faster_whisper import WhisperModel
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "faster-whisper is required for transcription. pip install faster-whisper"
    ) from e


WHISPER_SR = 16000


def _to_mono16k(block: np.ndarray, src_sr: int) -> np.ndarray:
    """Down-mix to mono and resample to 16 kHz float32."""
    if block.ndim == 2:
        mono = block.mean(axis=1)
    else:
        mono = block
    mono = mono.astype(np.float32, copy=False)
    if src_sr == WHISPER_SR:
        return mono
    # resample_poly handles arbitrary integer up/down ratios efficiently
    from math import gcd

    g = gcd(src_sr, WHISPER_SR)
    up = WHISPER_SR // g
    down = src_sr // g
    return resample_poly(mono, up, down).astype(np.float32, copy=False)


class _SourceBuffer:
    __slots__ = ("label", "buffer", "lock")

    def __init__(self, label: str):
        self.label = label
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()

    def append(self, samples: np.ndarray) -> None:
        with self.lock:
            self.buffer = np.concatenate([self.buffer, samples])

    def take(self, max_samples: int) -> Optional[np.ndarray]:
        with self.lock:
            if self.buffer.size < max_samples:
                return None
            chunk = self.buffer[:max_samples]
            self.buffer = self.buffer[max_samples:]
            return chunk

    def drain(self) -> np.ndarray:
        with self.lock:
            chunk = self.buffer
            self.buffer = np.zeros(0, dtype=np.float32)
            return chunk


class LiveTranscriber:
    def __init__(
        self,
        output_path: Path,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        source_sample_rate: int = 48000,
        chunk_seconds: float = 10.0,
        min_chunk_seconds: float = 2.0,
        beam_size: int = 5,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        vad_min_silence_ms: int = 500,
        on_output: Optional[Callable[[str], None]] = None,
    ):
        self.output_path = output_path
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.src_sr = source_sample_rate
        self.chunk_samples = int(chunk_seconds * WHISPER_SR)
        self.min_chunk_samples = int(min_chunk_seconds * WHISPER_SR)
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.initial_prompt = initial_prompt
        self.vad_min_silence_ms = vad_min_silence_ms
        self._on_output = on_output

        self._buffers: dict[str, _SourceBuffer] = {}
        self._buffers_lock = threading.Lock()
        # Rolling per-source prior text used as context for the next chunk.
        self._prior_text: dict[str, str] = {}
        self._stop = threading.Event()
        self._model: Optional[WhisperModel] = None
        self._worker: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._start_wallclock: Optional[dt.datetime] = None
        self._file = None
        self._gpu_detected: bool = False

    # ------------------------------------------------------------------ public
    def start(self) -> None:
        print(
            f"Loading Whisper model '{self.model_name}' "
            f"(device={self.device}, compute_type={self.compute_type})..."
        )
        kwargs = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.compute_type != "auto":
            kwargs["compute_type"] = self.compute_type

        cuda_available = False
        if self.device in ("cuda", "auto"):
            cuda_available, cuda_info = _check_cuda_available()
            print(f"CUDA check:\n{cuda_info}")

            loaded = _preload_cuda_libs()
            if loaded:
                print(f"Preloaded {len(loaded)} CUDA libs from pip wheels.")
            elif cuda_available:
                print("WARNING: No CUDA libs preloaded. GPU may fail.")
                print("Try: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12")

        # Track if we're using GPU
        self._gpu_detected = False
        
        try:
            self._model = WhisperModel(self.model_name, **kwargs)
            # If we get here without exception, check if using CUDA
            if self.device == "cuda":
                self._gpu_detected = True
            elif self.device == "auto" and cuda_available:
                # Check if model actually loaded on GPU by inspecting the model
                self._gpu_detected = hasattr(self._model, 'model') and 'cuda' in str(type(self._model.model)).lower()
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if self.device in ("auto", "cuda") and ("libcublas" in error_msg or "CUDA" in error_msg):
                print(f"\n[GPU Error: {error_msg}]")
                print("\nTroubleshooting:")
                print("1. Install NVIDIA pip packages:")
                print("   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12")
                print("2. Or use CPU mode: --whisper-device cpu")
                print("\nFalling back to CPU/int8...")
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
            elif self.device in ("auto", "cuda"):
                print(f"GPU load failed ({e}); falling back to CPU/int8.")
                self._model = WhisperModel(
                    self.model_name, device="cpu", compute_type="int8"
                )
            else:
                raise
        print(f"Whisper model ready (GPU: {'Yes' if self._gpu_detected else 'No'}).")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        self._start_wallclock = dt.datetime.now()
        self._write_line(
            f"# transcript started {self._start_wallclock:%Y-%m-%d %H:%M:%S}"
        )

        self._worker = threading.Thread(
            target=self._run, name="whisper-worker", daemon=True
        )
        self._worker.start()

    def feed(self, label: str, block: np.ndarray) -> None:
        """Feed an audio block (any sample rate set at init) for the given source."""
        if self._stop.is_set():
            return
        samples = _to_mono16k(block, self.src_sr)
        if samples.size == 0:
            return
        with self._buffers_lock:
            buf = self._buffers.get(label)
            if buf is None:
                buf = _SourceBuffer(label)
                self._buffers[label] = buf
        buf.append(samples)

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=30)
        # Flush remaining audio
        with self._buffers_lock:
            buffers = list(self._buffers.values())
        for buf in buffers:
            tail = buf.drain()
            if tail.size >= self.min_chunk_samples:
                self._transcribe(buf.label, tail)
        if self._file is not None:
            self._file.close()
            self._file = None

    # ----------------------------------------------------------------- internal
    def _run(self) -> None:
        while not self._stop.is_set():
            did_work = False
            with self._buffers_lock:
                buffers = list(self._buffers.values())
            for buf in buffers:
                chunk = buf.take(self.chunk_samples)
                if chunk is not None:
                    self._transcribe(buf.label, chunk)
                    did_work = True
            if not did_work:
                self._stop.wait(0.3)

    def _transcribe(self, label: str, audio: np.ndarray) -> None:
        assert self._model is not None
        # Build the prompt fed to Whisper for this chunk: a static initial_prompt
        # (vocabulary hints) plus the recent prior transcript for this source.
        prior = self._prior_text.get(label, "")
        prompt_parts = [p for p in (self.initial_prompt, prior) if p]
        prompt = " ".join(prompt_parts).strip() or None
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms},
                beam_size=self.beam_size,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=prompt,
                temperature=[0.0, 0.2, 0.4],  # graceful fallbacks if beam collapses
                no_speech_threshold=0.6,
                compression_ratio_threshold=2.4,
            )
            text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if ("libcublas" in msg or "CUDA" in msg) and self.device != "cpu":
                print(f"\n[GPU runtime error: {msg}] falling back to CPU/int8")
                try:
                    self._model = WhisperModel(
                        self.model_name, device="cpu", compute_type="int8"
                    )
                    self.device = "cpu"
                    self.compute_type = "int8"
                    return self._transcribe(label, audio)
                except Exception as e2:  # noqa: BLE001
                    print(f"\n[CPU fallback failed: {e2}]")
                    return
            print(f"\n[transcribe error: {e}]")
            return
        if not text_parts:
            return
        text = " ".join(text_parts)
        # Keep the tail of recent text (~200 chars) as context for the next chunk
        # of this same source. Whisper truncates prompts to ~224 tokens.
        if self.condition_on_previous_text:
            combined = (self._prior_text.get(label, "") + " " + text).strip()
            self._prior_text[label] = combined[-400:]
        ts = dt.datetime.now()
        rel = ts - (self._start_wallclock or ts)
        secs = int(rel.total_seconds())
        hh, rem = divmod(secs, 3600)
        mm, ss = divmod(rem, 60)
        line = f"[{hh:02d}:{mm:02d}:{ss:02d}] {label}: {text}"
        self._write_line(line)
        print("\n" + line)

    def _write_line(self, line: str) -> None:
        with self._writer_lock:
            if self._file is not None:
                self._file.write(line + "\n")
            if self._on_output is not None:
                self._on_output(line)
                self._file.flush()
