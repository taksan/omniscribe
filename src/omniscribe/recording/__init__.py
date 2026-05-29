"""Meeting recording feature."""

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

__all__ = [
    "detect_default_mic_source",
    "detect_monitor_source",
    "list_devices",
    "main",
    "resolve_mic_device",
    "resolve_system_device",
    "transcribe_file",
]

# Backward-compatible private alias used by tests
_looks_like_pulse_source = looks_like_pulse_source
