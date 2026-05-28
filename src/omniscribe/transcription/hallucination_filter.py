"""Detect and remove known Whisper hallucination patterns."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np

from .constants import DEFAULT_HALLUCINATION_PATTERNS


class HallucinationFilter:
    """Filter to detect and reject Whisper hallucination patterns."""

    # Average speaking rate: ~3 words/second (150-180 wpm in Portuguese)
    WORDS_PER_SECOND = 3.0
    # If expected duration > actual * this threshold, likely a hallucination
    DURATION_MISMATCH_THRESHOLD = 1.5

    def __init__(self, custom_blocklist_path: Path | None = None):
        self.patterns: list[re.Pattern[str]] = [
            re.compile(re.escape(pattern), re.IGNORECASE)
            for pattern in DEFAULT_HALLUCINATION_PATTERNS
        ]
        if custom_blocklist_path and custom_blocklist_path.exists():
            self._load_custom_patterns(custom_blocklist_path)

    def _load_custom_patterns(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        self.patterns.append(re.compile(re.escape(line), re.IGNORECASE))
        except OSError as e:
            print(f"Warning: Could not load custom blocklist from {path}: {e}")

    def is_hallucination(
        self,
        text: str,
        duration_seconds: float | None = None,
        audio: np.ndarray | None = None,
    ) -> bool:
        text_normalized = unicodedata.normalize("NFC", text.lower()).strip()
        # Normalize whitespace for pattern matching
        text_normalized_ws = " ".join(text_normalized.split())

        for pattern in self.patterns:
            if pattern.search(text_normalized_ws):
                return True

        # Detect repetitive words (common Whisper glitches)
        words = text_normalized.split()
        if self._has_repetition_glitch(words):
            return True

        # Check for duration mismatch (phrase takes longer to say than audio duration)
        if duration_seconds is not None and duration_seconds > 0:
            if self._has_duration_mismatch(text_normalized, duration_seconds):
                return True

        # Check for low-energy audio (silence/noise producing transcription)
        if audio is not None and len(text_normalized) > 0:
            if self._has_low_energy_mismatch(audio, text_normalized):
                return True

        return False

    def _has_duration_mismatch(self, text: str, duration_seconds: float) -> bool:
        # Estimate expected speaking time based on word count
        words = len(text.split())
        expected_seconds = words / self.WORDS_PER_SECOND

        # If text would take significantly longer than the audio duration, it's likely hallucinated
        # Allow 1.5x threshold for fast speakers or slurred speech
        if expected_seconds > 0 and duration_seconds > 0:
            ratio = expected_seconds / duration_seconds
            return ratio > self.DURATION_MISMATCH_THRESHOLD
        return False

    def _has_repetition_glitch(self, words: list[str]) -> bool:
        if len(words) < 4:
            return False

        # Detect 4+ consecutive identical words (clear hallucination pattern)
        # This handles: "the the the the", "um um um um", "e e e e", etc.
        for i in range(len(words) - 3):
            if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                return True

        return False

    def _has_low_energy_mismatch(self, audio: np.ndarray, text: str) -> bool:
        # Check if there's transcribed text from very low-energy audio
        # This catches hallucinations on silence/background noise
        if audio.size == 0:
            return False

        # Calculate audio energy (RMS in dB)
        rms = np.sqrt(np.mean(audio**2))
        db = 20 * np.log10(max(rms, 1e-10))

        # Threshold: if audio is extremely quiet (below -38 dB) and we have text,
        # it's very likely a hallucination. Real speech even when whispered is
        # typically above -35 dB. Use -38 to allow for measurement variance.
        # Only flag if we have multiple words (avoid false positives on short utterances).
        if db < -38.0:
            # Audio is too quiet to contain intelligible speech
            # Check if we have substantial text (more than just artifacts)
            words = text.split()
            if len(words) >= 2:
                return True  # Suspicious: text from very quiet audio

        return False

    def filter_transcript_file(self, path: Path) -> int:
        """Remove hallucinated lines from a transcript file. Returns lines removed."""
        if not path.exists():
            return 0

        lines_kept: list[str] = []
        lines_filtered = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line_stripped = line.strip()
                if line_stripped.startswith("#"):
                    lines_kept.append(line)
                    continue
                if self.is_hallucination(line_stripped):
                    lines_filtered += 1
                    print(f"[Final filter removed: {line_stripped[:60]}...]")
                else:
                    lines_kept.append(line)

        if lines_filtered > 0:
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(lines_kept)
            print(f"[Final filtering complete: {lines_filtered} hallucinated lines removed]")

        return lines_filtered
