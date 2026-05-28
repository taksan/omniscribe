"""Backward-compatible re-exports for the transcription feature."""

from omniscribe.transcription import (
    DEFAULT_HALLUCINATION_PATTERNS,
    HallucinationFilter,
    LiveTranscriber,
    SourceBuffer,
    WHISPER_SR,
    create_live_transcriber,
    to_mono16k,
)
from omniscribe.transcription.transcript_metadata import get_git_sha as _get_git_sha

_SourceBuffer = SourceBuffer
_to_mono16k = to_mono16k

__all__ = [
    "DEFAULT_HALLUCINATION_PATTERNS",
    "HallucinationFilter",
    "LiveTranscriber",
    "WHISPER_SR",
    "_SourceBuffer",
    "_get_git_sha",
    "_to_mono16k",
    "create_live_transcriber",
]
