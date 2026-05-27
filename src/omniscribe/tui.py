"""Terminal User Interface for OmniScribe using Rich.

Provides a live-updating dashboard with:
- Device status panel
- Transcription settings panel  
- VU meters for mic and system audio
- Message log
- Live transcription display
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.style import Style
from rich.text import Text


@dataclass
class TUIState:
    """State container for the TUI."""
    
    # Device info
    mic_device: str = "Auto"
    system_device: str = "Auto"
    mic_gain: float = 1.0
    sys_gain: float = 1.0
    
    # Transcription settings
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    language: str | None = None
    initial_prompt: str | None = None
    gpu_detected: bool = False
    
    # Audio levels (in dBFS)
    mic_level: float = -60.0
    sys_level: float = -60.0
    
    # Messages
    messages: deque[str] = field(default_factory=lambda: deque(maxlen=50))
    
    # Transcription lines
    transcript_lines: deque[str] = field(default_factory=lambda: deque(maxlen=100))
    
    # Status
    recording: bool = False
    transcribing: bool = False
    
    def add_message(self, msg: str) -> None:
        """Add a message with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        self.messages.append(f"[{timestamp}] {msg}")
    
    def add_transcript(self, line: str) -> None:
        """Add a transcript line."""
        self.transcript_lines.append(line)


class OmniScribeTUI:
    """Rich-based Terminal User Interface for OmniScribe."""
    
    def __init__(self) -> None:
        self.state = TUIState()
        self._state_lock = threading.Lock()
        self.console = Console()
        self._live: Live | None = None
        self._stop_event = threading.Event()
        self._update_thread: threading.Thread | None = None
        
    def _make_devices_panel(self) -> Panel:
        """Create the devices panel (top left)."""
        content = Group(
            Text(f"Audio: {self.state.system_device}", style="cyan"),
            Text(f"Mic: {self.state.mic_device}", style="cyan"),
            Text(f"Gain: {self.state.mic_gain:.1f}x / {self.state.sys_gain:.1f}x", style="cyan"),
        )
        return Panel(content, title="[b blue]Devices", border_style="blue")
    
    def _make_transcription_panel(self) -> Panel:
        """Create the transcription settings panel (top middle-left)."""
        gpu_status = "[green]Yes[/green]" if self.state.gpu_detected else "[red]No[/red]"
        lang = self.state.language or "Auto"
        prompt = self.state.initial_prompt or "None"
        
        content = Group(
            Text(f"Whisper model: {self.state.whisper_model}", style="cyan"),
            Text(f"Device: {self.state.whisper_device}", style="cyan"),
            Text(f"Language: {lang}", style="cyan"),
            Text(f"Initial prompt: {prompt[:30]}{'...' if len(str(prompt)) > 30 else ''}", style="cyan"),
            Text(f"GPU detected: {gpu_status}"),
        )
        return Panel(content, title="[b blue]Transcription", border_style="blue")
    
    def _make_vu_meter(self, label: str, level_db: float, color: str) -> Panel:
        """Create a VU meter panel."""
        # Convert dB to percentage for display (-60dB = 0%, 0dB = 100%)
        # Clamp between 0 and 100
        percentage = max(0, min(100, (level_db + 60) / 60 * 100))
        
        # Create a simple bar representation
        bar_width = 30
        filled = int(bar_width * percentage / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Color based on level
        if level_db > -6:
            bar_color = "red"
        elif level_db > -20:
            bar_color = "yellow"
        else:
            bar_color = color
        
        # Build the display
        lines = [
            Text(f"{label}", style=f"bold {color}"),
            Text(),
            Text(f"+50dB", style="dim"),
            Text(f"│[{bar_color}]{bar[:15]}[/{bar_color}]│", style="white"),
            Text(f"│[{bar_color}]{bar[15:]}[/{bar_color}]│", style="white"),
            Text(f"-50dB", style="dim"),
            Text(),
            Text(f"Peak: {level_db:+.1f}dB", style=f"bold {bar_color}"),
        ]
        
        content = Group(*lines)
        return Panel(content, title=f"[b {color}]{label}", border_style=color)
    
    def _make_audio_panel(self) -> Panel:
        """Create the audio VU meters panel (top center)."""
        mic_meter = self._make_vu_meter("MIC", self.state.mic_level, "red")
        sys_meter = self._make_vu_meter("AUDIO", self.state.sys_level, "green")
        
        # Side by side layout
        content = Group(
            Text("    MIC                AUDIO", style="bold"),
            Text(),
            f"[red] {self.state.mic_level:+5.1f}dB[/red]           [green]{self.state.sys_level:+5.1f}dB[/green]",
        )
        
        # Create a more visual representation
        mic_bar = self._db_to_bar(self.state.mic_level, "red")
        sys_bar = self._db_to_bar(self.state.sys_level, "green")
        
        layout_text = Text()
        layout_text.append("MIC\n", style="bold red")
        layout_text.append(mic_bar + f" {self.state.mic_level:+5.1f}dB\n\n", style="red")
        layout_text.append("AUDIO\n", style="bold green")
        layout_text.append(sys_bar + f" {self.state.sys_level:+5.1f}dB", style="green")
        
        return Panel(layout_text, title="[b blue]Audio Levels", border_style="blue")
    
    def _db_to_bar(self, db: float, color: str) -> str:
        """Convert dB level to a visual bar."""
        # Scale: -60dB to +6dB
        percentage = max(0, min(100, (db + 60) / 66 * 100))
        width = 20
        filled = int(width * percentage / 100)
        
        if db > -6:
            bar_char = "█"
        elif db > -20:
            bar_char = "▓"
        elif db > -40:
            bar_char = "▒"
        else:
            bar_char = "░"
        
        return bar_char * filled + "░" * (width - filled)
    
    def _make_messages_panel(self) -> Panel:
        """Create the messages panel (top right)."""
        if not self.state.messages:
            content = Text("No messages yet...", style="dim")
        else:
            lines = list(self.state.messages)[-10:]  # Show last 10
            content = Group(*[Text(line) for line in lines])
        
        return Panel(content, title="[b green]Messages", border_style="green")
    
    def _make_transcript_panel(self) -> Panel:
        """Create the live transcription panel (bottom)."""
        if not self.state.transcript_lines:
            content = Text("Waiting for transcription...", style="dim italic")
        else:
            lines = list(self.state.transcript_lines)[-20:]  # Show last 20 lines
            content = Group(*[Text(line) for line in lines])
        
        return Panel(
            content, 
            title="[b cyan]Live Transcription", 
            border_style="cyan",
            height=20
        )
    
    def _make_layout(self) -> Layout:
        """Create the full TUI layout (thread-safe)."""
        with self._state_lock:
            layout = Layout()
            
            # Split into header and body
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="body"),
            )
            
            # Header with title
            title = Align.center(
                Text("OmniScribe", style="bold blue"),
                vertical="middle"
            )
            layout["header"].update(
                Panel(title, border_style="blue")
            )
            
            # Body split into top row and transcript
            layout["body"].split_column(
                Layout(name="top_row", size=12),
                Layout(name="transcript"),
            )
            
            # Top row with 4 panels
            layout["top_row"].split_row(
                Layout(name="devices", ratio=1),
                Layout(name="transcription", ratio=1),
                Layout(name="audio", ratio=1),
                Layout(name="messages", ratio=1),
            )
            
            # Update each panel
            layout["devices"].update(self._make_devices_panel())
            layout["transcription"].update(self._make_transcription_panel())
            layout["audio"].update(self._make_audio_panel())
            layout["messages"].update(self._make_messages_panel())
            layout["transcript"].update(self._make_transcript_panel())
            
            return layout
    
    def _refresh_loop(self) -> None:
        """Background thread to refresh the TUI."""
        while self.state.recording and self._live:
            try:
                self.update()
                time.sleep(0.033)  # ~30 FPS refresh for smoother transcript
            except Exception:
                break

    def start(self) -> None:
        """Start the TUI."""
        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=10,
            screen=True  # Use alternate screen buffer
        )
        self._live.start()
        self.state.recording = True
        self.state.add_message("TUI started. Recording...")
        
        # Start background refresh thread
        self._refresh_thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._refresh_thread.start()
    
    def stop(self) -> None:
        """Stop the TUI."""
        self.state.recording = False
        # Wait for refresh thread to stop
        if hasattr(self, '_refresh_thread') and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=0.5)
        if self._live:
            self._live.stop()
            self._live = None
    
    def update(self) -> None:
        """Force a refresh of the display."""
        if self._live:
            self._live.update(self._make_layout())
    
    def update_audio_levels(self, mic_db: float, sys_db: float) -> None:
        """Update the audio level displays (thread-safe)."""
        with self._state_lock:
            self.state.mic_level = mic_db
            self.state.sys_level = sys_db
    
    def add_message(self, msg: str) -> None:
        """Add a message to the log (thread-safe)."""
        with self._state_lock:
            self.state.add_message(msg)
    
    def add_transcript(self, line: str) -> None:
        """Add a transcript line (thread-safe)."""
        with self._state_lock:
            self.state.add_transcript(line)
    
    def set_devices(
        self, 
        mic: str, 
        system: str, 
        mic_gain: float = 1.0, 
        sys_gain: float = 1.0
    ) -> None:
        """Set device information."""
        self.state.mic_device = mic
        self.state.system_device = system
        self.state.mic_gain = mic_gain
        self.state.sys_gain = sys_gain
    
    def set_transcription_config(
        self,
        model: str,
        device: str,
        language: str | None = None,
        initial_prompt: str | None = None,
        gpu_detected: bool = False
    ) -> None:
        """Set transcription configuration."""
        self.state.whisper_model = model
        self.state.whisper_device = device
        self.state.language = language
        self.state.initial_prompt = initial_prompt
        self.state.gpu_detected = gpu_detected
        self.state.transcribing = True


class TUIAudioCallback:
    """Adapter to connect audio callbacks to the TUI."""
    
    def __init__(self, tui: OmniScribeTUI) -> None:
        self.tui = tui
        self._mic_peak = -60.0
        self._sys_peak = -60.0
        self._decay = 0.9  # Peak decay factor
    
    def update(self, mic_block: np.ndarray | None, sys_block: np.ndarray | None) -> None:
        """Update audio levels from audio blocks."""
        if mic_block is not None and mic_block.size > 0:
            # Calculate RMS in dB
            rms = np.sqrt(np.mean(mic_block ** 2))
            db = 20 * np.log10(max(rms, 1e-10))
            # Update peak with decay
            self._mic_peak = max(db, self._mic_peak * self._decay)
        else:
            self._mic_peak *= self._decay
            
        if sys_block is not None and sys_block.size > 0:
            rms = np.sqrt(np.mean(sys_block ** 2))
            db = 20 * np.log10(max(rms, 1e-10))
            self._sys_peak = max(db, self._sys_peak * self._decay)
        else:
            self._sys_peak *= self._decay
        
        # Update TUI
        self.tui.update_audio_levels(self._mic_peak, self._sys_peak)


# Global TUI instance for easy access from recorder
_tui_instance: OmniScribeTUI | None = None


def get_tui() -> OmniScribeTUI | None:
    """Get the global TUI instance if active."""
    return _tui_instance


def create_tui() -> OmniScribeTUI:
    """Create and set the global TUI instance."""
    global _tui_instance
    _tui_instance = OmniScribeTUI()
    return _tui_instance


def destroy_tui() -> None:
    """Destroy the global TUI instance."""
    global _tui_instance
    if _tui_instance:
        _tui_instance.stop()
        _tui_instance = None
