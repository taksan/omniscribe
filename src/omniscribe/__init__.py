"""OmniScribe: Record meetings with system audio and microphone.

This package provides tools for simultaneous recording of system audio
(what you hear) and microphone input, with optional live transcription
using faster-whisper.

Example:
    >>> from omniscribe import record_meeting
    >>> record_meeting(
    ...     output_path="meeting.wav",
    ...     transcribe=True,
    ...     language="pt",
    ... )
"""

__version__ = "0.1.0"
__all__ = ["record", "LiveTranscriber", "check_inputs", "list_devices", "Config", "load_config", "OmniScribeTUI"]

from omniscribe.config import Config, load_config
from omniscribe.recording import check_inputs, list_devices, record
from omniscribe.transcription import LiveTranscriber
from omniscribe.ui import OmniScribeTUI
