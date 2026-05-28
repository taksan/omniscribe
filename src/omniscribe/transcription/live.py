"""Live Whisper transcription worker."""

from __future__ import annotations

import datetime as dt
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "faster-whisper is required for transcription. pip install faster-whisper"
    ) from e

from .audio import to_mono16k
from .buffer import SourceBuffer
from .constants import WHISPER_SR
from .cuda import check_cuda_available, preload_cuda_libs
from .hallucination_filter import HallucinationFilter
from .transcript_metadata import get_git_sha


class LiveTranscriber:
    def __init__(
        self,
        output_path: Path,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
        source_sample_rate: int = 48000,
        chunk_seconds: float = 6.0,
        min_chunk_seconds: float = 2.0,
        beam_size: int = 5,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        vad_min_silence_ms: int = 500,
        on_output: Optional[Callable[[str], None]] = None,
        enable_hallucination_filter: bool = True,
        hallucination_blocklist: Path | None = None,
        silence_threshold_db: float = -45.0,
        per_segment_output: bool = True,
        min_logprob: float = -1.0,
        max_no_speech_prob: float = 0.4,
        enable_repetition_filter: bool = True,
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
        self.per_segment_output = per_segment_output
        self.min_logprob = min_logprob
        self.max_no_speech_prob = max_no_speech_prob
        self.silence_threshold_db = silence_threshold_db
        self.enable_repetition_filter = enable_repetition_filter
        self._hallucination_filter: Optional[HallucinationFilter] = None
        if enable_hallucination_filter:
            self._hallucination_filter = HallucinationFilter(hallucination_blocklist)

        self._buffers: dict[str, SourceBuffer] = {}
        self._buffers_lock = threading.Lock()
        self._prior_text: dict[str, str] = {}
        self._stop = threading.Event()
        self._model: Optional[WhisperModel] = None
        self._worker: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._start_wallclock: Optional[dt.datetime] = None
        self._file = None
        self._gpu_detected = False
        self._recent_outputs: dict[str, list[str]] = {}
        self._outputs_lock = threading.Lock()

    @property
    def has_hallucination_filter(self) -> bool:
        return self._hallucination_filter is not None

    @property
    def gpu_detected(self) -> bool:
        return self._gpu_detected

    def start(self) -> None:
        print(
            f"Loading Whisper model '{self.model_name}' "
            f"(device={self.device}, compute_type={self.compute_type})..."
        )
        kwargs: dict[str, str] = {}
        if self.device != "auto":
            kwargs["device"] = self.device
        if self.compute_type != "auto":
            kwargs["compute_type"] = self.compute_type

        cuda_available = False
        if self.device in ("cuda", "auto"):
            cuda_available, cuda_info = check_cuda_available()
            print(f"CUDA check:\n{cuda_info}")

            loaded = preload_cuda_libs()
            if loaded:
                print(f"Preloaded {len(loaded)} CUDA libs from pip wheels.")
            elif cuda_available:
                print("WARNING: No CUDA libs preloaded. GPU may fail.")
                print("Try: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 nvidia-cuda-runtime-cu12")

        self._gpu_detected = False
        try:
            self._model = WhisperModel(self.model_name, **kwargs)
            if self.device == "cuda":
                self._gpu_detected = True
            elif self.device == "auto" and cuda_available:
                self._gpu_detected = hasattr(self._model, "model") and "cuda" in str(
                    type(self._model.model)
                ).lower()
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            if self.device in ("auto", "cuda") and (
                "libcublas" in error_msg or "CUDA" in error_msg
            ):
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
        git_sha = get_git_sha()
        self._write_line(
            f"# transcript started {self._start_wallclock:%Y-%m-%d %H:%M:%S} (version: {git_sha})"
        )

        self._worker = threading.Thread(
            target=self._run, name="whisper-worker", daemon=True
        )
        self._worker.start()

    def feed(self, label: str, block: np.ndarray) -> None:
        if self._stop.is_set():
            return
        samples = to_mono16k(block, self.src_sr)
        if samples.size == 0:
            return
        with self._buffers_lock:
            buf = self._buffers.get(label)
            if buf is None:
                buf = SourceBuffer(label)
                self._buffers[label] = buf
        buf.append(samples)

    def stop(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=30)
        with self._buffers_lock:
            buffers = list(self._buffers.values())
        for buf in buffers:
            tail = buf.drain()
            if tail.size >= self.min_chunk_samples:
                self._transcribe(buf.label, tail)
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._hallucination_filter:
            self._hallucination_filter.filter_transcript_file(self.output_path)

    # --- Private ---

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

        if self._is_silent(audio):
            return

        prior = self._prior_text.get(label, "")
        prompt_parts = [p for p in (self.initial_prompt, prior) if p]
        prompt = " ".join(prompt_parts).strip() or None
        chunk_start_ts = dt.datetime.now()

        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.language,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": self.vad_min_silence_ms},
                beam_size=self.beam_size,
                condition_on_previous_text=self.condition_on_previous_text,
                initial_prompt=prompt,
                temperature=[0.0, 0.2, 0.4],
                no_speech_threshold=0.7,
                compression_ratio_threshold=2.4,
            )
            segments_list = list(segments)
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

        valid_segments = [
            seg
            for seg in segments_list
            if self._accept_segment(label, seg.text.strip(), seg)
        ]
        if not valid_segments:
            return

        all_text = " ".join(seg.text.strip() for seg in valid_segments)
        if self.condition_on_previous_text:
            combined = (self._prior_text.get(label, "") + " " + all_text).strip()
            self._prior_text[label] = combined[-400:]

        if self.per_segment_output:
            self._write_segments(label, valid_segments, chunk_start_ts)
        else:
            self._write_chunk(label, valid_segments)

    def _accept_segment(self, label: str, text: str, seg) -> bool:
        if not text:
            return False
        if hasattr(seg, "avg_logprob") and seg.avg_logprob < self.min_logprob:
            return False
        if hasattr(seg, "no_speech_prob") and seg.no_speech_prob > self.max_no_speech_prob:
            return False
        if hasattr(seg, "compression_ratio") and seg.compression_ratio > 2.0:
            return False
        if self._hallucination_filter and self._hallucination_filter.is_hallucination(text):
            print(f"[Hallucination filtered: {text[:50]}...]")
            return False
        if self._is_repetition(label, text):
            print(f"[Repetition filtered: {text[:50]}...]")
            return False
        return True

    def _write_segments(self, label: str, segments, chunk_start_ts: dt.datetime) -> None:
        chunk_start_rel = chunk_start_ts - (self._start_wallclock or chunk_start_ts)
        chunk_start_secs = int(chunk_start_rel.total_seconds())
        for seg in segments:
            seg_timestamp = chunk_start_secs + int(seg.start)
            hh, rem = divmod(seg_timestamp, 3600)
            mm, ss = divmod(rem, 60)
            line = f"[{hh:02d}:{mm:02d}:{ss:02d}] {label}: {seg.text.strip()}"
            self._write_line(line)
            print("\n" + line)
            self._record_output(label, seg.text.strip())

    def _write_chunk(self, label: str, segments) -> None:
        text = " ".join(seg.text.strip() for seg in segments)
        ts = dt.datetime.now()
        rel = ts - (self._start_wallclock or ts)
        secs = int(rel.total_seconds())
        hh, rem = divmod(secs, 3600)
        mm, ss = divmod(rem, 60)
        line = f"[{hh:02d}:{mm:02d}:{ss:02d}] {label}: {text}"
        self._write_line(line)
        print("\n" + line)
        self._record_output(label, text)

    def _is_silent(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        rms = np.sqrt(np.mean(audio**2))
        db = 20 * np.log10(max(rms, 1e-10))
        return db < self.silence_threshold_db

    def _is_repetition(self, label: str, text: str) -> bool:
        if not self.enable_repetition_filter:
            return False
        with self._outputs_lock:
            recent = self._recent_outputs.get(label, [])
            return any(text in prev or prev in text for prev in recent)

    def _record_output(self, label: str, text: str) -> None:
        if not self.enable_repetition_filter:
            return
        with self._outputs_lock:
            if label not in self._recent_outputs:
                self._recent_outputs[label] = []
            self._recent_outputs[label].append(text)
            self._recent_outputs[label] = self._recent_outputs[label][-5:]

    def _write_line(self, line: str) -> None:
        with self._writer_lock:
            if self._file is not None:
                self._file.write(line + "\n")
                self._file.flush()
            if self._on_output is not None:
                self._on_output(line)
