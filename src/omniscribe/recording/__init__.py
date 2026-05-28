"""Meeting recording feature."""

from .audio import to_stereo
from .capture import PulseMonitorRecorder, StreamRecorder
from .cli import main, transcribe_file
from .devices import (
    detect_default_mic_source,
    detect_monitor_source,
    device_channels,
    list_devices,
    looks_like_pulse_source,
    resolve_mic_device,
    resolve_system_device,
)
from .input_check import check_inputs
from .meters import dbfs, meter_bar
from .session import record

__all__ = [
    "PulseMonitorRecorder",
    "StreamRecorder",
    "check_inputs",
    "detect_default_mic_source",
    "detect_monitor_source",
    "list_devices",
    "main",
    "record",
    "resolve_mic_device",
    "resolve_system_device",
    "to_stereo",
    "transcribe_file",
]

# Backward-compatible private aliases used by tests
_looks_like_pulse_source = looks_like_pulse_source
_dbfs = dbfs
_meter_bar = meter_bar
