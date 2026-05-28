"""Backward-compatible re-exports for the recording feature."""

from omniscribe.recording import (
    PulseMonitorRecorder,
    StreamRecorder,
    check_inputs,
    detect_default_mic_source,
    detect_monitor_source,
    list_devices,
    main,
    record,
    resolve_mic_device,
    resolve_system_device,
    to_stereo,
)
from omniscribe.recording.devices import looks_like_pulse_source as _looks_like_pulse_source
from omniscribe.recording.meters import dbfs as _dbfs, meter_bar as _meter_bar

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
