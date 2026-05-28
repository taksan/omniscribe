"""Backward-compatible re-exports for the TUI."""

from omniscribe.ui.tui import (
    OmniScribeTUI,
    TUIAudioCallback,
    create_tui,
    destroy_tui,
    get_tui,
)

__all__ = [
    "OmniScribeTUI",
    "TUIAudioCallback",
    "create_tui",
    "destroy_tui",
    "get_tui",
]
