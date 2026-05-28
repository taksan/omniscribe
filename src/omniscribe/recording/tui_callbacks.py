"""Connect audio capture callbacks to the TUI level meters."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from omniscribe.ui.tui import OmniScribeTUI


def make_audio_level_callback(
    tui: OmniScribeTUI,
    source: str,
) -> Callable[[np.ndarray], None]:
    """Return a callback that updates mic or system level on the TUI."""

    def callback(arr: np.ndarray) -> None:
        rms = np.sqrt(np.mean(arr**2))
        db = 20 * np.log10(max(rms, 1e-10))
        if source == "mic":
            tui.update_audio_levels(db, tui.state.sys_level)
        else:
            tui.update_audio_levels(tui.state.mic_level, db)

    return callback
