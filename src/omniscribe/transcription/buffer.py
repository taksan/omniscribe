"""Per-source ring buffer for live transcription."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class SourceBuffer:
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
