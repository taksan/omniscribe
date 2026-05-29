"""Live recording session orchestration."""

from __future__ import annotations

import queue
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from omniscribe.transcription.factory import create_live_transcriber
from omniscribe.transcription.transcript_metadata import get_git_sha
from omniscribe.ui import create_tui

from .alerts import play_alert_tone
from .audio import to_stereo
from .capture import PulseMonitorRecorder, StreamRecorder
from .constants import (
    CHANNELS,
    DTYPE,
    METER_REFRESH_S,
    SAMPLE_RATE,
    SILENCE_THRESHOLD,
)
from .devices import (
    device_channels,
    device_label,
    looks_like_pulse_source,
    pulse_source_channels,
    pulse_source_description,
)
from .meters import meter_bar
from .tui_callbacks import make_audio_level_callback


def record(
    output: Path,
    mic_device,
    system_device,
    mic_gain: float,
    sys_gain: float,
    separate: bool,
    transcribe: bool = False,
    whisper_model: str = "small",
    whisper_device: str = "cpu",
    whisper_compute_type: str = "int8",
    language: str | None = None,
    chunk_seconds: float = 6.0,
    mic_label: str = "You",
    system_label: str = "Them",
    silence_alert_seconds: float = 30.0,
    show_meters: bool = True,
    beam_size: int = 5,
    condition_on_previous_text: bool = True,
    initial_prompt: str | None = None,
    vad_min_silence_ms: int = 500,
    tui: bool = False,
    enable_hallucination_filter: bool = True,
    hallucination_blocklist: Path | None = None,
    silence_threshold_db: float = -45.0,
    per_segment_output: bool = True,
    min_logprob: float = -1.0,
    max_no_speech_prob: float = 0.4,
    enable_repetition_filter: bool = True,
    split_channels: bool = True,
    diarize_them: bool = False,
    num_speakers: int = 2,
) -> None:
    use_parec_for_mic = isinstance(mic_device, str) and looks_like_pulse_source(mic_device)
    use_parec_for_system = isinstance(system_device, str) and looks_like_pulse_source(system_device)

    if use_parec_for_mic:
        mic_ch = pulse_source_channels(mic_device, fallback=1)
    else:
        mic_ch = device_channels(mic_device)

    if use_parec_for_system:
        sys_ch = pulse_source_channels(system_device, fallback=CHANNELS)
    else:
        sys_ch = device_channels(system_device)

    mic_name = (
        pulse_source_description(mic_device)
        if use_parec_for_mic
        else device_label(mic_device, "input")
    )
    sys_name = (
        pulse_source_description(system_device)
        if use_parec_for_system
        else device_label(system_device, "input")
    )

    tui_instance = None
    if tui:
        tui_instance = create_tui()
        # Set initialization status BEFORE starting TUI display
        tui_instance.state.initialization_status = "Initializing..."
        tui_instance.set_devices(mic_name, sys_name, mic_gain, sys_gain)
        tui_instance.set_transcription_config(
            model=whisper_model,
            device=whisper_device,
            language=language,
            initial_prompt=initial_prompt,
            gpu_detected=False,
        )
        # Start TUI display - now with initialization status visible
        tui_instance.start()
    else:
        print()
        print("=" * 72)
        print("Recording from:")
        print(
            f"  Mic    : {mic_name}  ({mic_ch}ch"
            + (", via parec" if use_parec_for_mic else "")
            + ")"
        )
        print(
            f"  System : {sys_name}  ({sys_ch}ch"
            + (", via parec" if use_parec_for_system else "")
            + ")"
        )
        print()
        print(f"Output WAV : {output}")
        if separate:
            print(f"             + {output.with_name(output.stem + '.mic.wav')}")
            print(f"             + {output.with_name(output.stem + '.system.wav')}")
        if silence_alert_seconds > 0:
            print(f"Silence alert: after {int(silence_alert_seconds)}s of silence on a source")
        print("=" * 72)
        print("Recording... press Ctrl+C to stop (Ctrl+C twice to force quit).")
    print()

    stop = threading.Event()
    if tui_instance:
        tui_instance.set_stop_callback(stop.set)
    interrupt_count = {"n": 0}

    def _on_signal(signum, _frame):
        interrupt_count["n"] += 1
        if interrupt_count["n"] == 1:
            sig = "Ctrl+C" if signum == signal.SIGINT else f"signal {signum}"
            print(
                f"\n{sig} received. Stopping recording, draining audio buffers"
                + (", and finalizing transcript" if transcribe else "")
                + "... (press Ctrl+C again to force quit)"
            )
            stop.set()
        else:
            print("\nForce quit requested. Files may be truncated.", file=sys.stderr)
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    mic_callback = make_audio_level_callback(tui_instance, "mic") if tui_instance else None
    sys_callback = make_audio_level_callback(tui_instance, "system") if tui_instance else None

    if use_parec_for_mic:
        mic = PulseMonitorRecorder(mic_device, mic_ch, "mic", stop, tui_callback=mic_callback)
    else:
        mic = StreamRecorder(mic_device, mic_ch, "mic", stop, tui_callback=mic_callback)
    if use_parec_for_system:
        sysrec = PulseMonitorRecorder(system_device, sys_ch, "system", stop, tui_callback=sys_callback)
    else:
        sysrec = StreamRecorder(system_device, sys_ch, "system", stop, tui_callback=sys_callback)

    output.parent.mkdir(parents=True, exist_ok=True)
    mixed = sf.SoundFile(
        str(output), mode="w", samplerate=SAMPLE_RATE, channels=CHANNELS, subtype="PCM_16"
    )
    mic_file = sys_file = None
    if separate:
        mic_file = sf.SoundFile(
            str(output.with_name(output.stem + ".mic.wav")),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            subtype="PCM_16",
        )
        sys_file = sf.SoundFile(
            str(output.with_name(output.stem + ".system.wav")),
            mode="w",
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            subtype="PCM_16",
        )

    transcriber = None
    if transcribe:
        from omniscribe.config import Config

        transcript_path = output.with_suffix(".txt")
        on_output = tui_instance.add_transcript if tui_instance else None
        whisper_config = Config(
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            whisper_compute_type=whisper_compute_type,
            language=language,
            chunk_seconds=chunk_seconds,
            beam_size=beam_size,
            condition_on_previous_text=condition_on_previous_text,
            initial_prompt=initial_prompt,
            vad_min_silence_ms=vad_min_silence_ms,
            enable_hallucination_filter=enable_hallucination_filter,
            hallucination_blocklist=str(hallucination_blocklist) if hallucination_blocklist else None,
            silence_threshold_db=silence_threshold_db,
            per_segment_output=per_segment_output,
            min_logprob=min_logprob,
            max_no_speech_prob=max_no_speech_prob,
            enable_repetition_filter=enable_repetition_filter,
        )
        if tui_instance:
            tui_instance.state.initialization_status = "Loading Whisper model..."

        transcriber = create_live_transcriber(
            transcript_path,
            config=whisper_config,
            source_sample_rate=SAMPLE_RATE,
            on_output=on_output,
        )
        transcriber.start()

        if tui_instance:
            # Update TUI with actual recording configuration
            tui_instance.state.gpu_detected = transcriber.gpu_detected
            tui_instance.state.whisper_model = whisper_config.whisper_model
            tui_instance.state.whisper_device = whisper_config.whisper_device
            tui_instance.state.language = whisper_config.language
            tui_instance.state.sample_rate = SAMPLE_RATE
            tui_instance.state.chunk_duration = whisper_config.chunk_seconds
            tui_instance.state.vad_min_silence_ms = whisper_config.vad_min_silence_ms
            tui_instance.state.silence_threshold_db = silence_threshold_db
            tui_instance.state.hallucination_filter_enabled = whisper_config.enable_hallucination_filter
            tui_instance.state.repetition_filter_enabled = enable_repetition_filter
            tui_instance.state.min_logprob = whisper_config.min_logprob
            tui_instance.state.max_no_speech_prob = whisper_config.max_no_speech_prob
            tui_instance.state.per_segment_output = whisper_config.per_segment_output
            tui_instance.state.version = get_git_sha()
            tui_instance.state.initialization_status = ""  # Clear when done - now shows real config
        else:
            print(f"Transcript: {transcript_path}")

    mic.start()
    sysrec.start()

    total_frames = 0
    last_render = 0.0
    mic_peak_acc = 0.0
    sys_peak_acc = 0.0
    now0 = time.monotonic()
    last_signal_time = {"mic": now0, "system": now0}
    alerted = {"mic": False, "system": False}

    try:
        while not stop.is_set() or not mic.queue.empty() or not sysrec.queue.empty():
            try:
                m = mic.queue.get(timeout=0.2)
            except queue.Empty:
                m = None
            try:
                s = sysrec.queue.get_nowait()
            except queue.Empty:
                s = None

            if m is None and s is None:
                if stop.is_set():
                    break
                continue

            if m is not None and s is not None:
                n = min(len(m), len(s))
                m = m[:n]
                s = s[:n]
            elif m is None:
                m = np.zeros((len(s), mic_ch), dtype=DTYPE)
            elif s is None:
                s = np.zeros((len(m), sys_ch), dtype=DTYPE)

            m2 = to_stereo(m) * mic_gain
            s2 = to_stereo(s) * sys_gain
            block_frames = len(m2)

            if split_channels:
                split = np.stack([m2[:, 0], s2[:, 0]], axis=1)
                mixed.write(split)
            else:
                mix = np.clip(m2 + s2, -1.0, 1.0)
                mixed.write(mix)

            if mic_file is not None:
                mic_file.write(m2)
            if sys_file is not None:
                sys_file.write(s2)
            if transcriber is not None:
                transcriber.feed(mic_label, m2)
                transcriber.feed(system_label, s2)

            total_frames += block_frames
            seconds = total_frames / SAMPLE_RATE

            mic_peak_block = float(np.abs(m2).max()) if m2.size else 0.0
            sys_peak_block = float(np.abs(s2).max()) if s2.size else 0.0
            mic_peak_acc = max(mic_peak_acc, mic_peak_block)
            sys_peak_acc = max(sys_peak_acc, sys_peak_block)

            now = time.monotonic()
            mic_muted = bool(tui_instance and tui_instance.state.mic_muted)
            sys_muted = bool(tui_instance and tui_instance.state.sys_muted)
            if mic_muted:
                last_signal_time["mic"] = now
                alerted["mic"] = False
            elif mic_peak_block > SILENCE_THRESHOLD:
                last_signal_time["mic"] = now
                if alerted["mic"]:
                    print(f"\n  >> mic audio resumed at {seconds:6.1f}s", flush=True)
                    alerted["mic"] = False
            if sys_muted:
                last_signal_time["system"] = now
                alerted["system"] = False
            elif sys_peak_block > SILENCE_THRESHOLD:
                last_signal_time["system"] = now
                if alerted["system"]:
                    print(f"\n  >> system audio resumed at {seconds:6.1f}s", flush=True)
                    alerted["system"] = False

            if silence_alert_seconds > 0 and not stop.is_set():
                for src in ("mic", "system"):
                    if alerted[src]:
                        continue
                    if src == "mic" and mic_muted:
                        continue
                    if src == "system" and sys_muted:
                        continue
                    if now - last_signal_time[src] >= silence_alert_seconds:
                        alerted[src] = True
                        label = "microphone" if src == "mic" else "system audio"
                        print(
                            f"\n  !! ALERT: no {label} detected for "
                            f"{int(silence_alert_seconds)}s",
                            flush=True,
                        )
                        play_alert_tone()

            if (
                show_meters
                and not tui_instance
                and not stop.is_set()
                and (now - last_render) >= METER_REFRESH_S
            ):
                last_render = now
                mm, ss = divmod(int(seconds), 60)
                hh, mm = divmod(mm, 60)
                line = (
                    f"  {hh:02d}:{mm:02d}:{ss:02d}  "
                    f"MIC {meter_bar(mic_peak_acc)}  "
                    f"SYS {meter_bar(sys_peak_acc)}"
                )
                print(f"\r{line}\033[K", end="", flush=True)
                mic_peak_acc *= 0.4
                sys_peak_acc *= 0.4
    finally:
        stop.set()
        if tui_instance:
            tui_instance.stop()
        else:
            print()

        print("  [1/3] Stopping audio capture threads...", flush=True)
        mic.join(timeout=2)
        sysrec.join(timeout=2)

        print("  [2/3] Closing WAV file(s)...", flush=True)
        mixed.close()
        if mic_file is not None:
            mic_file.close()
        if sys_file is not None:
            sys_file.close()

        if transcriber is not None:
            print(
                "  [3/3] Finalizing transcript "
                "(transcribing remaining audio, this may take a few seconds)...",
                flush=True,
            )
            transcriber.stop()
        else:
            print("  [3/3] Done.", flush=True)

        if mic.error:
            print(f"Mic stream error: {mic.error}", file=sys.stderr)
        if sysrec.error:
            print(f"System stream error: {sysrec.error}", file=sys.stderr)

        seconds = total_frames / SAMPLE_RATE
        mm, ss = divmod(int(seconds), 60)
        hh, mm = divmod(mm, 60)
        print(f"\nSaved {output}  ({hh:02d}:{mm:02d}:{ss:02d}, {seconds:.1f}s)")
        if transcriber is not None:
            print(f"Saved transcript {output.with_suffix('.txt')}")

        if diarize_them and split_channels:
            from omniscribe.diarization import check_pyannote_available, transcribe_split_with_diarization
            from omniscribe.config import Config

            if not check_pyannote_available():
                print(
                    "Warning: pyannote.audio required for diarization.\n"
                    "Install with: pip install 'omniscribe[diarize]'",
                    file=sys.stderr,
                )
            else:
                print("\n[Post-processing] Running speaker diarization on system channel...")
                diarized_path = output.with_suffix(".diarized.txt")
                diarize_config = Config(
                    whisper_model=whisper_model,
                    whisper_device=whisper_device,
                    whisper_compute_type=whisper_compute_type,
                    language=language,
                    chunk_seconds=chunk_seconds,
                    beam_size=beam_size,
                    initial_prompt=initial_prompt,
                    vad_min_silence_ms=vad_min_silence_ms,
                    enable_hallucination_filter=enable_hallucination_filter,
                    hallucination_blocklist=str(hallucination_blocklist) if hallucination_blocklist else None,
                    silence_threshold_db=silence_threshold_db,
                    min_logprob=min_logprob,
                    max_no_speech_prob=max_no_speech_prob,
                    enable_repetition_filter=enable_repetition_filter,
                )
                diarize_transcriber = create_live_transcriber(
                    diarized_path,
                    config=diarize_config,
                    source_sample_rate=SAMPLE_RATE,
                    condition_on_previous_text=False,
                )
                transcribe_split_with_diarization(
                    audio_path=output,
                    output_path=diarized_path,
                    transcriber=diarize_transcriber,
                    you_label=mic_label,
                    them_label=system_label,
                    num_speakers=num_speakers,
                    device=whisper_device,
                )
                print(f"Saved diarized transcript: {diarized_path}")
