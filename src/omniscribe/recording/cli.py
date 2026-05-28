"""CLI entry point for meeting recording."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from omniscribe.config import Config, load_config, save_default_config
from omniscribe.diarization import check_pyannote_available, transcribe_with_diarization
from omniscribe.transcription.factory import create_live_transcriber

from .devices import list_devices, resolve_mic_device, resolve_system_device
from .input_check import check_inputs
from .session import record

CLI_DOC = """Meeting recorder: captures system audio (what you hear) and microphone
simultaneously, then writes a mixed WAV file (and optionally separate tracks).

Works on Linux with PulseAudio or PipeWire (pulse compatibility layer) by
recording from the default sink's "monitor" source. Also works on systems
where the user manually selects a loopback/monitor input device.

Usage:
    python recorder.py --list                       # list devices
    python recorder.py                              # record to meeting.wav
    python recorder.py -o talk.wav                  # custom output
    python recorder.py --mic 3 --system 7           # explicit device indices
    python recorder.py --separate                   # also save mic.wav/system.wav
    python recorder.py --mic-gain 1.0 --sys-gain 1.0

Stop recording with Ctrl+C.
"""


def transcribe_file(
    input_path: Path,
    config: Config,
    split_channels: bool = False,
    diarize: bool = False,
    num_speakers: int = 2,
) -> int:
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Transcribing: {input_path}")

    if diarize:
        if not check_pyannote_available():
            print(
                "Error: pyannote.audio is required for diarization.\n"
                "Install with: pip install 'omniscribe[diarize]'",
                file=sys.stderr,
            )
            return 1

        output_path = input_path.with_suffix(".diarized.txt")
        transcriber = create_live_transcriber(
            output_path,
            config=config,
            source_sample_rate=48000,
            condition_on_previous_text=False,
        )
        transcribe_with_diarization(
            audio_path=input_path,
            output_path=output_path,
            transcriber=transcriber,
            num_speakers=num_speakers,
            device=config.whisper_device,
        )
        return 0

    audio, sr = sf.read(str(input_path))
    if audio.ndim == 1:
        audio = audio[:, None]

    num_channels = audio.shape[1]

    if split_channels:
        if num_channels < 2:
            print("Error: --input-split-channels requires a stereo file", file=sys.stderr)
            return 1
        print(f"Split-channel mode: Ch0={config.mic_label}, Ch1={config.system_label}")
        channels_to_process = [
            (0, config.mic_label),
            (1, config.system_label),
        ]
    else:
        channels_to_process = [(None, config.mic_label)]

    output_path = input_path.with_suffix(".txt")
    transcriber = create_live_transcriber(
        output_path,
        config=config,
        source_sample_rate=sr,
    )
    transcriber.start()

    chunk_samples = int(config.chunk_seconds * sr)
    for ch_idx, label in channels_to_process:
        print(f"  Processing {label}...")
        if ch_idx is None:
            ch_audio = audio[:, 0] if num_channels == 1 else audio.mean(axis=1)
        else:
            ch_audio = audio[:, ch_idx]

        for i in range(0, len(ch_audio), chunk_samples):
            chunk = ch_audio[i : i + chunk_samples]
            if chunk.ndim == 1:
                chunk = np.stack([chunk, chunk], axis=1)
            transcriber.feed(label, chunk)

    print("  Finalizing...")
    transcriber.stop()
    print(f"Saved transcript: {output_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=CLI_DOC, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                 help="path to config JSON file (default: ~/.config/local-transcriber/config.json or ./config.json)")
    p.add_argument("--init-config", action="store_true",
                 help="create a default config.json file and exit")
    p.add_argument("--list", action="store_true", help="list audio devices and exit")
    p.add_argument("--check", action="store_true",
                   help="run a short mic/system noise check and exit")
    p.add_argument("--check-seconds", type=float, default=3.0,
                   help="duration of the --check measurement window")
    p.add_argument("--transcribe-file", type=Path, default=None,
                   help="transcribe an existing WAV file (offline mode, no recording)")
    p.add_argument("--input-split-channels", action="store_true",
                   help="treat input file as split stereo: Left=Mic/You, Right=System/Them "
                        "(transcribes each channel separately with labels)")
    p.add_argument("--diarize", action="store_true",
                   help="run speaker diarization on the audio file (requires 'omniscribe[diarize]' extra)")
    p.add_argument("--diarize-them", action="store_true",
                   help="detect multiple speakers in system channel (Them) of split stereo recording")
    p.add_argument("--num-speakers", type=int, default=2,
                   help="expected number of speakers for diarization (default: 2)")
    default_name = f"meeting-{dt.datetime.now():%Y%m%d-%H%M%S}.wav"
    p.add_argument("-o", "--output", type=Path, default=Path("meetings") / default_name)
    p.add_argument("--mic", help="microphone device (index or name)")
    p.add_argument("--system", help="system audio device (monitor/loopback index or name)")
    p.add_argument("--mic-gain", type=float, default=1.0)
    p.add_argument("--sys-gain", type=float, default=1.0)
    p.add_argument("--separate", action="store_true", help="also save mic.wav and system.wav")
    p.add_argument("--transcribe", action="store_true", help="live transcribe with Whisper")
    p.add_argument("--whisper-model", default="small",
                   help="faster-whisper model name (tiny, base, small, medium, large-v3, ...)")
    p.add_argument("--whisper-device", default="cpu", help="cpu|cuda|auto")
    p.add_argument("--whisper-compute-type", default="int8",
                   help="int8|int8_float16|float16|float32|auto")
    p.add_argument("--language", default=None, help="language code, e.g. en, pt (auto-detect if omitted)")
    p.add_argument("--chunk-seconds", type=float, default=10.0,
                   help="audio chunk length fed to Whisper (longer = better context, more latency)")
    p.add_argument("--beam-size", type=int, default=5,
                   help="Whisper beam search width (1=greedy, 5=default, 10=slower/better)")
    p.add_argument("--no-context", action="store_true",
                   help="disable conditioning on previously transcribed text "
                        "(use if you see hallucination / repetition)")
    p.add_argument("--initial-prompt", default=None,
                   help="vocabulary hint string fed before each chunk; great for proper "
                        "nouns, names, jargon, e.g. 'Reuni\u00e3o sobre Kubernetes, Ansible, Tiago'")
    p.add_argument("--vad-min-silence-ms", type=int, default=500,
                   help="silence (ms) required to split speech in the VAD filter")
    p.add_argument("--no-hallucination-filter", action="store_true",
                   help="disable filtering of known Whisper hallucination patterns")
    p.add_argument("--hallucination-blocklist", type=Path, default=None,
                   help="path to custom file with hallucination patterns (one per line)")
    p.add_argument("--silence-threshold-db", type=float, default=-50.0,
                   help="skip audio chunks below this RMS dB threshold (default: -50)")
    p.add_argument("--no-per-segment", action="store_true",
                   help="output one line per chunk instead of per-segment timestamps")
    p.add_argument("--min-logprob", type=float, default=-1.0,
                   help="minimum avg_logprob for segment acceptance (default: -1.0)")
    p.add_argument("--max-no-speech-prob", type=float, default=0.5,
                   help="maximum no_speech_prob for segment acceptance (default: 0.5)")
    p.add_argument("--no-repetition-filter", action="store_true",
                   help="disable filtering of repeated identical segments")
    p.add_argument("--no-split-channels", action="store_true",
                   help="mix mic+system into both channels (default: split channels)")
    p.add_argument("--split-channels", action="store_true",
                   help=argparse.SUPPRESS)
    p.add_argument("--mic-label", default="You")
    p.add_argument("--system-label", default="Them")
    p.add_argument("--silence-alert-seconds", type=float, default=30.0,
                   help="play an alert tone after this many seconds of silence "
                        "on a source (0 to disable)")
    p.add_argument("--no-meters", action="store_true",
                   help="disable the live ASCII level meters")
    p.add_argument("--tui", action="store_true",
                   help="use fancy terminal UI with live transcription display")
    args = p.parse_args()

    if args.init_config:
        path = save_default_config(args.config)
        print(f"Created default config file: {path}")
        print("Edit this file to customize your default settings.")
        return 0

    config = load_config(args.config)
    config.merge_with_args(args)

    if args.list:
        list_devices()
        return 0

    def _coerce(v):
        if v is None:
            return None
        try:
            return int(v)
        except ValueError:
            return v

    if args.check:
        mic_dev = resolve_mic_device(_coerce(config.mic))
        sys_dev = resolve_system_device(_coerce(config.system))
        check_inputs(mic_dev, sys_dev, duration=config.check_seconds)
        return 0

    if args.transcribe_file:
        return transcribe_file(
            args.transcribe_file,
            config,
            split_channels=args.input_split_channels,
            diarize=args.diarize,
            num_speakers=args.num_speakers,
        )

    mic_dev = resolve_mic_device(_coerce(config.mic))
    sys_dev = resolve_system_device(_coerce(config.system))
    output = Path(config.output) if config.output else args.output

    try:
        record(
            output=output,
            mic_device=mic_dev,
            system_device=sys_dev,
            mic_gain=config.mic_gain,
            sys_gain=config.sys_gain,
            separate=config.separate,
            transcribe=config.transcribe,
            whisper_model=config.whisper_model,
            whisper_device=config.whisper_device,
            whisper_compute_type=config.whisper_compute_type,
            language=config.language,
            chunk_seconds=config.chunk_seconds,
            mic_label=config.mic_label,
            system_label=config.system_label,
            silence_alert_seconds=config.silence_alert_seconds,
            show_meters=config.show_meters,
            beam_size=config.beam_size,
            condition_on_previous_text=config.condition_on_previous_text,
            initial_prompt=config.initial_prompt,
            vad_min_silence_ms=config.vad_min_silence_ms,
            tui=args.tui,
            enable_hallucination_filter=config.enable_hallucination_filter,
            hallucination_blocklist=Path(config.hallucination_blocklist) if config.hallucination_blocklist else None,
            silence_threshold_db=config.silence_threshold_db,
            per_segment_output=config.per_segment_output,
            min_logprob=config.min_logprob,
            max_no_speech_prob=config.max_no_speech_prob,
            enable_repetition_filter=config.enable_repetition_filter,
            split_channels=config.split_channels,
            diarize_them=config.diarize_them or args.diarize_them,
            num_speakers=args.num_speakers,
        )
    except KeyboardInterrupt:
        return 130
    return 0
