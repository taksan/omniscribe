"""Core logic: correlate audio energy with transcript segments."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

# Conservative speaking rate for duration mismatch detection
WORDS_PER_SECOND = 2.5
# RMS window size for energy curve (in seconds)
ENERGY_WINDOW_S = 0.25


@dataclass
class TranscriptSegment:
    timestamp_secs: int
    label: str
    text: str
    word_count: int = field(init=False)
    expected_speech_secs: float = field(init=False)

    def __post_init__(self) -> None:
        self.word_count = len(self.text.split())
        self.expected_speech_secs = self.word_count / WORDS_PER_SECOND


@dataclass
class SegmentAnalysis:
    segment: TranscriptSegment
    audio_duration_secs: float   # actual window length in audio
    rms_db: float                # RMS energy of the window
    silence_ratio: float         # fraction of sub-frames below silence threshold
    peak_db: float               # peak dB in window
    energy_std_db: float         # std-dev of frame-level energy — low = flat/noise-like
    words_per_sec: float         # transcript density relative to audio window
    suspicion_score: float       # 0.0–1.0
    flags: list[str]             # human-readable reasons


def _rms_db(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio ** 2)))
    return float(20 * np.log10(max(rms, 1e-10)))


def _peak_db(audio: np.ndarray) -> float:
    peak = float(np.abs(audio).max()) if audio.size else 1e-10
    return float(20 * np.log10(max(peak, 1e-10)))


def _energy_std_db(audio: np.ndarray, sr: int, frame_s: float = 0.05) -> float:
    """Std-dev of frame-level RMS energy (dB). Low value = flat, noise-like profile."""
    frame = int(sr * frame_s)
    if audio.size < frame * 2:
        return 0.0
    n = audio.size // frame
    frames = audio[: n * frame].reshape(n, frame)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)).clip(1e-10))
    return float(np.std(frame_db))


def _silence_ratio(audio: np.ndarray, sr: int, threshold_db: float) -> float:
    """Fraction of 50ms sub-frames below threshold."""
    frame = int(sr * 0.05)
    if audio.size < frame:
        return 1.0
    n_frames = audio.size // frame
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)).clip(1e-10))
    return float((frame_db < threshold_db).mean())


def parse_transcript(path: Path) -> tuple[dict, list[TranscriptSegment]]:
    """Return (headers_dict, segments_list)."""
    pattern = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s+(\w+):\s+(.+)$")
    headers: dict[str, str] = {}
    segments: list[TranscriptSegment] = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("#"):
                if ":" in line:
                    key, _, val = line[2:].partition(":")
                    headers[key.strip()] = val.strip()
                continue
            m = pattern.match(line)
            if m:
                hh, mm, ss, label, text = m.groups()
                secs = int(hh) * 3600 + int(mm) * 60 + int(ss)
                segments.append(TranscriptSegment(secs, label, text.strip()))

    return headers, segments


def energy_curve(audio_mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (times, rms_db_values) sampled every ENERGY_WINDOW_S seconds."""
    frame = int(sr * ENERGY_WINDOW_S)
    n = audio_mono.size // frame
    if n == 0:
        return np.array([0.0]), np.array([_rms_db(audio_mono)])
    frames = audio_mono[: n * frame].reshape(n, frame)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)).clip(1e-10)
    db = 20 * np.log10(rms)
    times = (np.arange(n) + 0.5) * ENERGY_WINDOW_S
    return times, db


def analyze(
    wav_path: Path,
    transcript_path: Path,
    silence_threshold_db: float = -45.0,
    mic_label: str = "You",
    system_label: str = "Them",
) -> tuple[dict, list[SegmentAnalysis], dict[str, np.ndarray], int, float | None]:
    """
    Returns (headers, analyses, channel_audio_dict, sample_rate, drift_scale).
    drift_scale is non-None when timestamps were wall-clock inflated (old recordings).
    channel_audio_dict keys: mic_label, system_label.
    """
    audio, sr = sf.read(str(wav_path), dtype="float32")
    audio_duration = audio.shape[0] / sr

    # Channel assignment (split-channel layout: ch0=mic, ch1=system)
    if audio.ndim == 2 and audio.shape[1] >= 2:
        channels = {mic_label: audio[:, 0], system_label: audio[:, 1]}
    else:
        mono = audio if audio.ndim == 1 else audio[:, 0]
        channels = {mic_label: mono, system_label: mono}

    headers, segments = parse_transcript(transcript_path)

    # Detect timestamp drift from old wall-clock approach.
    # GPU transcription time was included in timestamps, causing non-linear drift
    # that cannot be corrected reliably with a linear scale factor.
    drift_scale: float | None = None
    if segments:
        transcript_span = segments[-1].timestamp_secs
        if transcript_span > 0:
            ratio = transcript_span / audio_duration
            if ratio > 1.05:  # more than 5% overshoot → old recording
                drift_scale = ratio  # returned as a warning signal, not corrected

    # Build per-label index for same-channel next-timestamp lookup
    label_timestamps: dict[str, list[int]] = {}
    for seg in segments:
        label_timestamps.setdefault(seg.label, []).append(seg.timestamp_secs)

    analyses: list[SegmentAnalysis] = []
    skipped_beyond_audio = 0
    for i, seg in enumerate(segments):
        start_sample = int(seg.timestamp_secs * sr)
        ch_audio = channels.get(seg.label, channels[mic_label])

        # Find next timestamp for this label that is strictly later
        same_label_ts = label_timestamps.get(seg.label, [])
        later = [t for t in same_label_ts if t > seg.timestamp_secs]
        if later:
            end_sample = int(min(later) * sr)
        else:
            end_sample = start_sample + int(15 * sr)

        end_sample = min(end_sample, ch_audio.size)
        if start_sample >= ch_audio.size:
            # Timestamp beyond audio length — likely timestamp drift in old recordings
            skipped_beyond_audio += 1
            continue
        if start_sample >= end_sample:
            audio_duration = 0.1
            window = np.zeros(1, dtype=np.float32)
        else:
            audio_duration = (end_sample - start_sample) / sr
            window = ch_audio[start_sample:end_sample]

        rdb = _rms_db(window)
        pdb = _peak_db(window)
        sil = _silence_ratio(window, sr, silence_threshold_db)
        estd = _energy_std_db(window, sr)
        wps = seg.word_count / max(audio_duration, 0.1)

        # Suspicion scoring
        flags: list[str] = []
        score = 0.0

        if rdb < silence_threshold_db:
            score += 0.5
            flags.append(f"silent audio ({rdb:.1f} dB < {silence_threshold_db:.1f} dB threshold)")
        elif rdb < silence_threshold_db + 8:
            score += 0.25
            flags.append(f"low energy ({rdb:.1f} dB)")

        if sil > 0.85:
            score += 0.3
            flags.append(f"{sil*100:.0f}% of window is silence")

        # Flat energy profile: low std-dev means steady noise, not speech.
        # Real speech fluctuates ≥ 8 dB as syllables and pauses alternate.
        # Only apply when audio is clearly audible (above threshold, low silence).
        if (rdb > silence_threshold_db
                and sil < 0.5
                and seg.word_count > 0
                and estd < 5.0):
            score += 0.35
            flags.append(
                f"flat energy profile: {estd:.1f} dB std-dev "
                f"(speech typically > 8 dB — likely steady background noise)"
            )

        # active_duration = portion of the window that had real energy
        active_duration = (1.0 - sil) * audio_duration

        # Check 1: too many words for available active audio
        # Uses high thresholds because natural in-sentence pauses compress active_duration.
        # Only fire when density is extreme relative to the energy that was actually present.
        if seg.word_count > 3 and active_duration > 0:
            active_wps = seg.word_count / active_duration
            if active_wps > 8.0:
                score += 0.4
                flags.append(
                    f"very dense for active audio: {active_wps:.1f} w/s over "
                    f"{active_duration:.1f}s active ({seg.word_count} words)"
                )
            elif active_wps > 6.0:
                score += 0.2
                flags.append(
                    f"dense for active audio: {active_wps:.1f} w/s over "
                    f"{active_duration:.1f}s active"
                )

        # Check 2: few words over long high-energy window
        # Catches e.g. "Legenda por Sônia Ruberti" over 13s of real audio.
        # Cap at 25s to ignore long silences between speaker turns.
        # Skip truncated lines (mid-sentence "...") and very short utterances.
        active_window_capped = (1.0 - sil) * min(audio_duration, 25.0)
        is_truncated = seg.text.rstrip().endswith("...")
        if (active_window_capped >= 5.0
                and seg.word_count >= 4
                and not is_truncated
                and seg.word_count / active_window_capped < 0.5):
            score += 0.4
            flags.append(
                f"sparse text for active audio: {seg.word_count} words over "
                f"{active_window_capped:.1f}s of energy "
                f"({seg.word_count / active_window_capped:.2f} w/s)"
            )

        score = min(score, 1.0)
        analyses.append(SegmentAnalysis(
            segment=seg,
            audio_duration_secs=audio_duration,
            rms_db=rdb,
            silence_ratio=sil,
            peak_db=pdb,
            energy_std_db=estd,
            words_per_sec=wps,
            suspicion_score=score,
            flags=flags,
        ))

    return headers, analyses, channels, sr, skipped_beyond_audio, drift_scale
