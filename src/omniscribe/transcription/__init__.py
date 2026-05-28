"""Live Whisper transcription feature."""

from .audio import to_mono16k
from .buffer import SourceBuffer
from .constants import DEFAULT_HALLUCINATION_PATTERNS, WHISPER_SR
from .factory import create_live_transcriber
from .hallucination_filter import HallucinationFilter
from .live import LiveTranscriber

__all__ = [
    "DEFAULT_HALLUCINATION_PATTERNS",
    "HallucinationFilter",
    "LiveTranscriber",
    "SourceBuffer",
    "WHISPER_SR",
    "create_live_transcriber",
    "to_mono16k",
]

# Backward-compatible private aliases used by tests
_SourceBuffer = SourceBuffer
_to_mono16k = to_mono16k
