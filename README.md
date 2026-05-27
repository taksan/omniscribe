# local-transcriber: meeting recorder

Records **system audio (what you hear)** + **microphone** simultaneously and
saves a single mixed WAV file. Works with any meeting app (Google Meet,
Microsoft Teams, Zoom, Discord, ...) since it captures at the OS audio layer.

Includes optional live transcription using faster-whisper with optimized
settings for non-English languages.

Tested on Linux (PulseAudio / PipeWire).

## Project Structure

```
local-transcriber/
├── src/local_transcriber/     # Main package
│   ├── __init__.py
│   ├── recorder.py            # Audio recording logic
│   └── transcriber.py         # Whisper transcription
├── tests/                       # Test suite
├── scripts/                     # Helper scripts
│   └── start.sh                 # Example startup script
├── meetings/                    # Default output directory
├── pyproject.toml              # Project configuration
└── README.md                    # This file
```

## Install

System dependencies (Debian/Ubuntu):

```bash
sudo apt install python3-venv portaudio19-dev pulseaudio-utils ffmpeg
```

Python package (editable install for development):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[transcribe]"  # Include transcription dependencies
```

## Quick start

```bash
# 1. See devices and the auto-detected monitor source
python -m local_transcriber --list

# 2. Record. Plays back through your normal speakers/headphones meanwhile.
python -m local_transcriber  # saves to meetings/meeting-YYYYMMDD-HHMMSS.wav

# Stop with Ctrl+C.
```

On Linux the script auto-detects the **monitor** source of your current default
sink via `pactl get-default-sink` and uses that to capture system audio. Your
microphone defaults to the system default input.

## Useful options

```bash
python -m local_transcriber -o my-meeting.wav          # custom output path
python -m local_transcriber --mic 3 --system 7         # pick devices by index
python -m local_transcriber --separate                 # also write *.mic.wav and *.system.wav
python -m local_transcriber --mic-gain 1.2 --sys-gain 0.9
```

## Configuration File

Instead of passing flags every time, you can use a JSON configuration file.
CLI flags override config file settings.

### Create a config file

```bash
# Create default config in current directory
python -m local_transcriber --init-config

# Or specify a custom location
python -m local_transcriber --init-config --config ~/.config/local-transcriber/config.json
```

### Config file locations (searched in order)

1. `--config <path>` (explicit path)
2. `~/.config/local-transcriber/config.json`
3. `./config.json` (current directory)

### Example config.json

```json
{
  "mic": "alsa_input.usb-HP__Inc_HyperX_Cloud_III_Wireless_0000000000000000-00.mono-fallback",
  "transcribe": true,
  "language": "pt",
  "whisper_model": "large-v2",
  "whisper_device": "cuda",
  "whisper_compute_type": "float16",
  "chunk_seconds": 12,
  "beam_size": 5,
  "initial_prompt": "Reunião sobre AstraZeneca, speaker, CRM",
  "mic_label": "Eu",
  "system_label": "Eles"
}
```

Then simply run:

```bash
python -m local_transcriber  # Uses settings from config.json
```

## Tips for meetings

- Use **headphones** so the meeting audio coming out of your speakers is not
  picked up again by the microphone (otherwise you get echo / double voice).
- If you switch your default output device during recording, the monitor
  source will not follow. Set the desired sink before starting.
- If `--list` does not show any `*.monitor` input device, enable PulseAudio's
  Pulse PortAudio backend (default on Ubuntu) or pick a loopback device
  manually.

## Pre-flight Check

Before recording, verify your microphone is working and not too noisy:

```bash
python -m local_transcriber --check
```

This measures background noise, peak levels, and checks for clipping.
Run with `--check-seconds 5` for a longer sample.

## Live transcription (Whisper)

The `--transcribe` flag runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
locally on each audio source in real time. The transcript is written to a
`.txt` file next to the WAV and printed to the terminal as it is produced.

```bash
python -m local_transcriber --transcribe                              # default model: small
python -m local_transcriber --transcribe --whisper-model medium       # better quality
python -m local_transcriber --transcribe --whisper-model large-v2 \
                   --whisper-device cuda --whisper-compute-type float16
python -m local_transcriber --transcribe --language en                # skip auto-detect
python -m local_transcriber --transcribe --mic-label "Alice" --system-label "Bob"
```

Each line in the transcript is timestamped relative to the start of the
recording and tagged by source:

```
[00:00:12] You: Hi everyone, can you hear me?
[00:00:15] Them: Yes, loud and clear.
```

Transcription Tips:

- The first run downloads the model into `~/.cache/huggingface`. `tiny`/`base`
  are fast on CPU; `small`/`medium` give noticeably better quality. `large-v2`
  often works better than `large-v3` for non-English languages.
- Tune `--chunk-seconds` (default 10 s). Longer = better context and quality,
  slightly more latency. 10-15s is good for meetings.
- Use `--initial-prompt` to provide vocabulary hints: names, jargon, acronyms.
- Use `--beam-size 5` (default) for better quality than greedy decoding.
- Transcription runs in a worker thread and does not block recording. If
  Whisper falls behind, audio simply queues up and catches up at the end.

## Using the convenience script

A sample startup script is provided in `scripts/start.sh`. Copy and edit it
to set your preferred options:

```bash
cp scripts/start.sh my-meeting.sh
chmod +x my-meeting.sh
# Edit my-meeting.sh with your preferred mic, language, etc.
./my-meeting.sh
```

## Development

Install in editable mode with dev dependencies:

```bash
pip install -e ".[transcribe,dev]"
```

Run tests:

```bash
pytest
```

Format and lint:

```bash
ruff check src tests
ruff format src tests
```

Type checking:

```bash
mypy src/local_transcriber
```

## Convert to MP3 / M4A

```bash
ffmpeg -i meetings/meeting.wav -c:a libmp3lame -q:a 4 meetings/meeting.mp3
```

## Notes on legality

Recording meetings may be subject to laws and policies in your jurisdiction
and organization. Get consent from participants.
