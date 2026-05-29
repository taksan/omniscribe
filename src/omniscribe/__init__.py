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


def __getattr__(name: str):
    if name in ("Config", "load_config"):
        from omniscribe.config import Config, load_config
        globals()["Config"] = Config
        globals()["load_config"] = load_config
        return globals()[name]
    if name == "record":
        from omniscribe.recording.session import record
        globals()["record"] = record
        return record
    if name == "check_inputs":
        from omniscribe.recording.input_check import check_inputs
        globals()["check_inputs"] = check_inputs
        return check_inputs
    if name == "list_devices":
        from omniscribe.recording.devices import list_devices
        globals()["list_devices"] = list_devices
        return list_devices
    if name == "LiveTranscriber":
        from omniscribe.transcription import LiveTranscriber
        globals()["LiveTranscriber"] = LiveTranscriber
        return LiveTranscriber
    if name == "OmniScribeTUI":
        from omniscribe.ui import OmniScribeTUI
        globals()["OmniScribeTUI"] = OmniScribeTUI
        return OmniScribeTUI
    raise AttributeError(f"module 'omniscribe' has no attribute {name!r}")
