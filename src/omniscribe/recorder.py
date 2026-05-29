"""Backward-compatible re-exports for the recording feature."""

from omniscribe.recording.audio import to_stereo
from omniscribe.recording.capture import PulseMonitorRecorder, StreamRecorder
from omniscribe.recording.cli import main, transcribe_file
from omniscribe.recording.devices import (
    detect_default_mic_source,
    detect_monitor_source,
    list_devices,
    looks_like_pulse_source as _looks_like_pulse_source,
    resolve_mic_device,
    resolve_system_device,
)
from omniscribe.recording.input_check import check_inputs
from omniscribe.recording.meters import dbfs as _dbfs, meter_bar as _meter_bar
from omniscribe.recording.session import record

__all__ = [
    "PulseMonitorRecorder",
    "StreamRecorder",
    "_dbfs",
    "_looks_like_pulse_source",
    "_meter_bar",
    "check_inputs",
    "detect_default_mic_source",
    "detect_monitor_source",
    "list_devices",
    "main",
    "record",
    "resolve_mic_device",
    "resolve_system_device",
    "to_stereo",
]
