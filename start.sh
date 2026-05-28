#!/bin/bash
# Convenience script to start recording with preferred settings.
# Edit this file to customize your default recording options.

# Navigate to project root (where this script is located)
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Run the transcriber with your preferred settings
# Edit these options to match your setup:
#   --mic: your microphone PulseAudio source name (find with: python -m local_transcriber --list)
#   --language: pt, en, es, etc.
#   --whisper-model: tiny, base, small, medium, large-v2, large-v3
#   --whisper-device: cpu or cuda
#   --chunk-seconds: 6-15 (longer = better context)

# NOTE: --language is intentionally omitted to allow per-chunk auto-detection.
# This avoids Whisper translating English speech into Portuguese (and vice versa)
# during mixed-language meetings. Add `--language pt` back if your meetings are
# 100% Portuguese.
omniscribe --tui --mic alsa_input.usb-HP__Inc_HyperX_Cloud_III_Wireless_0000000000000000-00.mono-fallback \
    --transcribe \
    --whisper-device cuda \
    --whisper-compute-type float16 \
    --whisper-model large-v3 \
    --initial-prompt "Reunião em português, mas podem haver alguns termos em inglês, especialmente em nomes próprios de sistemas, aplicações ou estrangeirismos."


# omniscribe --transcribe-file meetings/meeting-20260527-103422.wav --language pt --diarize --num-speakers 2 \
#     --whisper-device cuda \
#     --whisper-compute-type float16 \
#     --whisper-model large-v3