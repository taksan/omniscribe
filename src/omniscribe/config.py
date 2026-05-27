"""Configuration management for OmniScribe.

Supports JSON configuration files with CLI override.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATHS = [
    Path.home() / ".config" / "omniscribe" / "config.json",
    Path.cwd() / "config.json",
]


@dataclass
class Config:
    """Configuration dataclass matching all CLI arguments."""

    # Output settings
    output: str | None = None

    # Device settings
    mic: str | None = None
    system: str | None = None
    mic_gain: float = 1.0
    sys_gain: float = 1.0

    # Recording options
    separate: bool = False

    # Transcription settings
    transcribe: bool = False
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    language: str | None = None
    chunk_seconds: float = 6.0
    beam_size: int = 5
    condition_on_previous_text: bool = True
    initial_prompt: str | None = None
    vad_min_silence_ms: int = 500

    # Transcription quality filtering
    enable_hallucination_filter: bool = True
    hallucination_blocklist: str | None = None
    silence_threshold_db: float = -50.0
    per_segment_output: bool = True
    min_logprob: float = -1.0
    max_no_speech_prob: float = 0.5
    enable_repetition_filter: bool = True

    # Channel layout
    split_channels: bool = False

    # Labeling
    mic_label: str = "You"
    system_label: str = "Them"

    # Silence alert
    silence_alert_seconds: float = 30.0

    # UI options
    show_meters: bool = True
    check_seconds: float = 3.0

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def merge_with_args(self, args: Any) -> None:
        """Merge CLI arguments into config (CLI overrides config)."""
        # Map of CLI arg names to their default values
        arg_defaults = {
            "mic_gain": 1.0,
            "sys_gain": 1.0,
            "separate": False,
            "transcribe": False,
            "whisper_model": "small",
            "whisper_device": "cpu",
            "whisper_compute_type": "int8",
            "language": None,
            "chunk_seconds": 6.0,
            "beam_size": 5,
            "condition_on_previous_text": True,
            "initial_prompt": None,
            "vad_min_silence_ms": 500,
            "mic_label": "You",
            "system_label": "Them",
            "silence_alert_seconds": 30.0,
            "show_meters": True,
            "check_seconds": 3.0,
            "enable_hallucination_filter": True,
            "hallucination_blocklist": None,
            "silence_threshold_db": -50.0,
            "per_segment_output": True,
            "min_logprob": -1.0,
            "max_no_speech_prob": 0.5,
            "enable_repetition_filter": True,
            "split_channels": False,
        }

        for key in arg_defaults:
            cli_value = getattr(args, key, None)
            # If CLI value differs from default, user explicitly set it
            if cli_value != arg_defaults[key]:
                setattr(self, key, cli_value)

        # Special handling for boolean flags that store True when present
        # These don't have defaults in argparse sense, we check if they were explicitly set
        for flag in ["separate", "transcribe", "no_context", "no_meters",
                     "no_hallucination_filter", "no_per_segment", "no_repetition_filter"]:
            if hasattr(args, flag):
                value = getattr(args, flag)
                if flag == "no_context":
                    if value:  # --no-context was used
                        self.condition_on_previous_text = False
                elif flag == "no_meters":
                    if value:  # --no-meters was used
                        self.show_meters = False
                elif flag == "no_hallucination_filter":
                    if value:
                        self.enable_hallucination_filter = False
                elif flag == "no_per_segment":
                    if value:
                        self.per_segment_output = False
                elif flag == "no_repetition_filter":
                    if value:
                        self.enable_repetition_filter = False
                else:
                    if value:  # Flag was used
                        setattr(self, flag, True)

        # Map --split-channels flag to split_channels config
        if hasattr(args, "split_channels"):
            if args.split_channels:
                self.split_channels = True

        # Positional/string args: if set, they override config
        for key in ["mic", "system", "language", "initial_prompt", "output"]:
            value = getattr(args, key, None)
            if value is not None:
                setattr(self, key, value)


def find_config_file(explicit_path: Path | None = None) -> Path | None:
    """Find config file from explicit path or default locations."""
    if explicit_path:
        return explicit_path if explicit_path.exists() else None

    for path in DEFAULT_CONFIG_PATHS:
        if path.exists():
            return path
    return None


def load_config(path: Path | None = None) -> Config:
    """Load configuration from JSON file."""
    config_file = find_config_file(path)
    if config_file is None:
        return Config()

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Config(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        raise SystemExit(f"Error loading config from {config_file}: {e}")


def save_default_config(path: Path | None = None) -> Path:
    """Save a default configuration file."""
    if path is None:
        path = Path.cwd() / "config.json"

    config = Config()
    config_data = config.to_dict()

    # Add comments as a special key (will be at top after json sorting)
    config_with_help = {
        "_comment": "OmniScribe configuration file. CLI flags override these settings.",
        "_help": {
            "mic": "Microphone device name or index",
            "system": "System audio device (monitor source)",
            "mic_gain": "Microphone gain multiplier (0.0 - 2.0)",
            "sys_gain": "System audio gain multiplier (0.0 - 2.0)",
            "separate": "Save separate mic.wav and system.wav files",
            "transcribe": "Enable live transcription with Whisper",
            "whisper_model": "Whisper model: tiny, base, small, medium, large-v2, large-v3",
            "whisper_device": "Device for Whisper: cpu, cuda, auto",
            "whisper_compute_type": "Computation type: int8, float16, float32",
            "language": "Language code (pt, en, es, etc). Omit for auto-detect",
            "chunk_seconds": "Audio chunk length for transcription (6-15 recommended)",
            "beam_size": "Beam search width (1=greedy, 5=default, 10=slow/best)",
            "condition_on_previous_text": "Use prior text as context for next chunk",
            "initial_prompt": "Vocabulary hints for Whisper (names, jargon)",
            "vad_min_silence_ms": "VAD silence threshold in milliseconds",
            "enable_hallucination_filter": "Filter known Whisper hallucination patterns",
            "hallucination_blocklist": "Path to custom hallucination patterns file",
            "silence_threshold_db": "Skip audio chunks below this dB threshold",
            "per_segment_output": "Output each Whisper segment with its own timestamp",
            "min_logprob": "Minimum avg_logprob for segment acceptance (-1.0)",
            "max_no_speech_prob": "Maximum no_speech_prob for segment acceptance (0.5)",
            "enable_repetition_filter": "Filter repeated identical segments",
            "split_channels": "Save main WAV as split stereo (L=Mic, R=System)",
            "mic_label": "Label for mic in transcript",
            "system_label": "Label for system in transcript",
            "silence_alert_seconds": "Seconds of silence before alert tone (0=disabled)",
            "show_meters": "Show ASCII peak meters during recording",
            "check_seconds": "Duration of --check command in seconds",
            "output": "Output file path (optional, default: meetings/meeting-<timestamp>.wav)",
        },
        **config_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_with_help, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return path
