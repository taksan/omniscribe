"""Detect and remove known Whisper hallucination patterns."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from .constants import DEFAULT_HALLUCINATION_PATTERNS


class HallucinationFilter:
    """Filter to detect and reject Whisper hallucination patterns."""

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

    def is_hallucination(self, text: str) -> bool:
        text_normalized = unicodedata.normalize("NFC", text.lower()).strip()
        for pattern in self.patterns:
            if pattern.search(text_normalized):
                return True
        words = text_normalized.split()
        if len(words) >= 4:
            for i in range(len(words) - 3):
                if words[i] == words[i + 1] == words[i + 2] == words[i + 3]:
                    return True
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
