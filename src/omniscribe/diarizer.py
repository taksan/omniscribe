"""Speaker diarization using pyannote.audio for legacy files.

This module provides speaker diarization to separate speakers in mixed
audio files (legacy recordings without split channels).
"""

from __future__ import annotations

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

if TYPE_CHECKING:
    from collections.abc import Sequence

# pyannote is optional - handle import errors gracefully
try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    Pipeline = None  # type: ignore[misc,assignment]
    Segment = None  # type: ignore[misc,assignment]


@dataclass
class SpeakerSegment:
    """A transcribed segment with speaker label."""

    start: float
    end: float
    speaker: str
    text: str


def check_pyannote_available() -> bool:
    """Check if pyannote.audio is installed and available."""
    return PYANNOTE_AVAILABLE


def diarize_audio(
    audio_path: Path,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "cpu",
) -> list[tuple[float, float, str]]:
    """Run speaker diarization on an audio file.

    Args:
        audio_path: Path to audio file
        num_speakers: Exact number of speakers (if known)
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        device: "cpu" or "cuda" for pyannote

    Returns:
        List of (start_time, end_time, speaker_label) tuples

    Raises:
        RuntimeError: If pyannote.audio is not installed
    """
    if not PYANNOTE_AVAILABLE:
        raise RuntimeError(
            "pyannote.audio is required for diarization. "
            "Install with: pip install 'omniscribe[diarize]'"
        )

    # Load diarization pipeline
    # Uses pyannote/speaker-diarization-3.1 by default
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # Suppress torch warnings
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=False,  # Model is public
        )
        if device == "cuda" and torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

    # Run diarization
    if num_speakers is not None:
        diarization = pipeline(str(audio_path), num_speakers=num_speakers)
    else:
        diarization = pipeline(
            str(audio_path),
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    # Convert to list of segments
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append((turn.start, turn.end, speaker))

    return results


def transcribe_with_diarization(
    audio_path: Path,
    output_path: Path,
    transcriber,
    num_speakers: int | None = 2,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "cpu",
) -> list[SpeakerSegment]:
    """Transcribe audio with speaker diarization.

    Args:
        audio_path: Path to audio file
        output_path: Path for transcript output
        transcriber: LiveTranscriber instance (for Whisper settings)
        num_speakers: Expected number of speakers (default: 2)
        min_speakers: Minimum speakers (overrides num_speakers if set)
        max_speakers: Maximum speakers (overrides num_speakers if set)
        device: "cpu" or "cuda" for pyannote

    Returns:
        List of SpeakerSegment with transcription results
    """
    print("Running speaker diarization...")
    diarization = diarize_audio(
        audio_path,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        device=device,
    )

    if not diarization:
        print("No speech detected in audio.")
        return []

    # Get unique speakers and assign friendly names
    unique_speakers = sorted(set(speaker for _, _, speaker in diarization))
    speaker_map = {s: f"Speaker {chr(65 + i)}" for i, s in enumerate(unique_speakers)}

    print(f"Detected {len(unique_speakers)} speakers: {list(speaker_map.values())}")

    # Load audio
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    # Convert to mono for transcription (pyannote already separated speakers)
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)

    results: list[SpeakerSegment] = []

    # Transcribe each speaker segment
    for start, end, speaker_id in diarization:
        # Extract segment audio
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < sr * 0.1:  # Skip very short segments (<100ms)
            continue

        # Create temporary file for this segment
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            sf.write(str(tmp_path), segment_audio, sr)

        try:
            # Transcribe segment
            # Note: We create a fresh transcriber for each segment to avoid
            # context contamination between speakers
            text = _transcribe_segment(tmp_path, transcriber)

            if text.strip():
                results.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker=speaker_map[speaker_id],
                    text=text.strip(),
                ))
        finally:
            tmp_path.unlink(missing_ok=True)

    # Write output
    _write_diarized_output(results, output_path)

    return results


def _transcribe_segment(audio_path: Path, transcriber) -> str:
    """Transcribe a single audio segment.

    Uses the Whisper model from the transcriber but processes
    the segment in isolation.
    """
    import soundfile as sf

    from .transcriber import LiveTranscriber

    # Load audio
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)  # Make stereo

    # Create a temporary transcriber for this segment
    # We don't want context from previous segments when doing diarization
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
        tmp_output = Path(tmp.name)

    try:
        # Create isolated transcriber (no condition_on_previous_text)
        segment_transcriber = LiveTranscriber(
            output_path=tmp_output,
            model_name=transcriber.model_name,
            device=transcriber.device,
            compute_type=transcriber.compute_type,
            language=transcriber.language,
            source_sample_rate=sr,
            chunk_seconds=transcriber.chunk_seconds,
            beam_size=transcriber.beam_size,
            condition_on_previous_text=False,  # No context for segments
            initial_prompt=transcriber.initial_prompt,
            vad_min_silence_ms=transcriber.vad_min_silence_ms,
            on_output=None,
            enable_hallucination_filter=transcriber._hallucination_filter is not None,
            hallucination_blocklist=None,
            silence_threshold_db=transcriber.silence_threshold_db,
            per_segment_output=True,
            min_logprob=transcriber.min_logprob,
            max_no_speech_prob=transcriber.max_no_speech_prob,
            enable_repetition_filter=transcriber.enable_repetition_filter,
        )

        segment_transcriber.start()
        segment_transcriber.feed("segment", audio)

        import time
        time.sleep(0.5)  # Let it process

        segment_transcriber.stop()

        # Read output
        if tmp_output.exists():
            with open(tmp_output, "r") as f:
                lines = f.readlines()
            # Extract just the text from lines
            texts = []
            for line in lines:
                if ":" in line:
                    text = line.split(":", 1)[1].strip()
                    if text:
                        texts.append(text)
            return " ".join(texts)
        return ""

    finally:
        tmp_output.unlink(missing_ok=True)


def _write_diarized_output(segments: Sequence[SpeakerSegment], output_path: Path) -> None:
    """Write diarized transcription to file."""
    with open(output_path, "w") as f:
        f.write("# Speaker Diarization Transcription\n")
        f.write("# Format: [HH:MM:SS] Speaker: Text\n\n")

        for seg in segments:
            # Format timestamp
            hours = int(seg.start // 3600)
            minutes = int((seg.start % 3600) // 60)
            seconds = int(seg.start % 60)
            timestamp = f"[{hours:02d}:{minutes:02d}:{seconds:02d}]"

            f.write(f"{timestamp} {seg.speaker}: {seg.text}\n")

    print(f"Saved diarized transcript: {output_path}")


def transcribe_split_with_diarization(
    audio_path: Path,
    output_path: Path,
    transcriber,
    you_label: str = "You",
    them_label: str = "Them",
    num_speakers: int | None = 2,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    device: str = "cpu",
) -> list[SpeakerSegment]:
    """Transcribe split stereo file with diarization on system channel.

    Processes left channel (You) as single speaker and right channel (Them)
    with diarization to detect multiple speakers.

    Args:
        audio_path: Path to stereo WAV file (split channels)
        output_path: Path for transcript output
        transcriber: LiveTranscriber instance (for Whisper settings)
        you_label: Label for left channel (mic)
        them_label: Label prefix for right channel speakers
        num_speakers: Expected number of speakers on system channel
        min_speakers: Minimum speakers (overrides num_speakers if set)
        max_speakers: Maximum speakers (overrides num_speakers if set)
        device: "cpu" or "cuda" for pyannote

    Returns:
        List of SpeakerSegment with merged transcription results
    """
    print("Loading split stereo audio...")
    audio, sr = sf.read(str(audio_path))
    if audio.ndim == 1:
        print("Error: Expected stereo file for split-channel diarization", file=sys.stderr)
        return []

    if audio.shape[1] < 2:
        print("Error: File must have at least 2 channels", file=sys.stderr)
        return []

    # Extract channels
    you_audio = audio[:, 0]  # Left = Mic = You
    them_audio = audio[:, 1]  # Right = System = Them (multi-speaker)

    results: list[SpeakerSegment] = []

    # Process You channel (single speaker)
    print(f"Processing {you_label} channel...")
    you_segments = _transcribe_mono_channel(
        you_audio, sr, transcriber, you_label
    )
    results.extend(you_segments)

    # Process Them channel with diarization
    print(f"Running diarization on {them_label} channel...")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), them_audio, sr)

    try:
        diarization = diarize_audio(
            tmp_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            device=device,
        )

        if diarization:
            # Get unique speakers and assign friendly names
            unique_speakers = sorted(set(speaker for _, _, speaker in diarization))
            speaker_map = {
                s: f"{them_label} {chr(65 + i)}" for i, s in enumerate(unique_speakers)
            }
            print(f"Detected {len(unique_speakers)} speakers: {list(speaker_map.values())}")

            # Transcribe each speaker segment
            for start, end, speaker_id in diarization:
                start_sample = int(start * sr)
                end_sample = int(end * sr)
                segment_audio = them_audio[start_sample:end_sample]

                if len(segment_audio) < sr * 0.1:  # Skip very short segments
                    continue

                text = _transcribe_audio_segment(segment_audio, sr, transcriber)
                if text.strip():
                    results.append(SpeakerSegment(
                        start=start,
                        end=end,
                        speaker=speaker_map[speaker_id],
                        text=text.strip(),
                    ))
    finally:
        tmp_path.unlink(missing_ok=True)

    # Sort all results by start time
    results.sort(key=lambda x: x.start)

    # Write output
    _write_diarized_output(results, output_path)

    return results


def _transcribe_mono_channel(
    audio: np.ndarray,
    sr: int,
    transcriber,
    label: str,
) -> list[SpeakerSegment]:
    """Transcribe a mono audio channel and return segments."""
    segments: list[SpeakerSegment] = []

    # Create temp file for this channel
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), audio, sr)

    try:
        text = _transcribe_segment(tmp_path, transcriber)
        if text.strip():
            # For continuous mono transcription, we treat it as one big segment
            # starting at time 0 (will be refined later with VAD if needed)
            duration = len(audio) / sr
            segments.append(SpeakerSegment(
                start=0.0,
                end=duration,
                speaker=label,
                text=text.strip(),
            ))
    finally:
        tmp_path.unlink(missing_ok=True)

    return segments


def _transcribe_audio_segment(
    audio: np.ndarray,
    sr: int,
    transcriber,
) -> str:
    """Transcribe a numpy audio segment."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        sf.write(str(tmp_path), audio, sr)

    try:
        return _transcribe_segment(tmp_path, transcriber)
    finally:
        tmp_path.unlink(missing_ok=True)


# Avoid torch import issues when pyannote not installed
if PYANNOTE_AVAILABLE:
    import torch
