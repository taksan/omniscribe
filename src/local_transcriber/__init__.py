"""Local Transcriber: Record meetings with system audio and microphone.

This package provides tools for simultaneous recording of system audio
(what you hear) and microphone input, with optional live transcription
using faster-whisper.

Example:
    >>> from local_transcriber import record_meeting
    >>> record_meeting(
    ...     output_path="meeting.wav",
    ...     transcribe=True,
    ...     language="pt",
    ... )
"""

__version__ = "0.1.0"
__all__ = ["record", "LiveTranscriber", "check_inputs", "list_devices"]

from local_transcriber.recorder import record, check_inputs, list_devices
from local_transcriber.transcriber import LiveTranscriber
