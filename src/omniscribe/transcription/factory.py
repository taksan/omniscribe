"""Factory for constructing LiveTranscriber from shared settings."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from omniscribe.config import Config

from .live import LiveTranscriber


def create_live_transcriber(
    output_path: Path,
    *,
    config: Config,
    source_sample_rate: int,
    on_output: Optional[Callable[[str], None]] = None,
    condition_on_previous_text: bool | None = None,
) -> LiveTranscriber:
    """Build a LiveTranscriber from Config and runtime overrides."""
    blocklist = (
        Path(config.hallucination_blocklist) if config.hallucination_blocklist else None
    )
    return LiveTranscriber(
        output_path=output_path,
        model_name=config.whisper_model,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
        language=config.language,
        source_sample_rate=source_sample_rate,
        chunk_seconds=config.chunk_seconds,
        beam_size=config.beam_size,
        condition_on_previous_text=(
            config.condition_on_previous_text
            if condition_on_previous_text is None
            else condition_on_previous_text
        ),
        initial_prompt=config.initial_prompt,
        vad_min_silence_ms=config.vad_min_silence_ms,
        on_output=on_output,
        enable_hallucination_filter=config.enable_hallucination_filter,
        hallucination_blocklist=blocklist,
        silence_threshold_db=config.silence_threshold_db,
        per_segment_output=config.per_segment_output,
        min_logprob=config.min_logprob,
        max_no_speech_prob=config.max_no_speech_prob,
        enable_repetition_filter=config.enable_repetition_filter,
    )
