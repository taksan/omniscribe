"""Per-source ring buffer for live transcription."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class SourceBuffer:
    __slots__ = ("label", "buffer", "lock", "samples_yielded")

    def __init__(self, label: str):
        self.label = label
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.samples_yielded = 0  # cumulative samples taken for transcription

    def append(self, samples: np.ndarray) -> None:
        with self.lock:
            self.buffer = np.concatenate([self.buffer, samples])

    def take(self, max_samples: int) -> Optional[tuple[np.ndarray, int]]:
        """Return (chunk, chunk_start_sample) or None if not enough data."""
        with self.lock:
            if self.buffer.size < max_samples:
                return None
            chunk = self.buffer[:max_samples]
            self.buffer = self.buffer[max_samples:]
            start = self.samples_yielded
            self.samples_yielded += max_samples
            return chunk, start

    def drain(self) -> tuple[np.ndarray, int]:
        """Return (remaining_samples, chunk_start_sample)."""
        with self.lock:
            chunk = self.buffer
            start = self.samples_yielded
            self.samples_yielded += len(chunk)
            self.buffer = np.zeros(0, dtype=np.float32)
            return chunk, start
