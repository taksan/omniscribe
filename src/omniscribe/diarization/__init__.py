"""Speaker diarization feature."""

from .diarizer import (
    PYANNOTE_AVAILABLE,
    SpeakerSegment,
    check_pyannote_available,
    diarize_audio,
    transcribe_split_with_diarization,
    transcribe_with_diarization,
)

__all__ = [
    "PYANNOTE_AVAILABLE",
    "SpeakerSegment",
    "check_pyannote_available",
    "diarize_audio",
    "transcribe_split_with_diarization",
    "transcribe_with_diarization",
]
