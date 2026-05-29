#!/bin/bash
# Convenience script to start recording with preferred settings.

# Navigate to project root (where this script is located)
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# NOTE: --language is intentionally omitted to allow per-chunk auto-detection.
# This avoids Whisper translating English speech into Portuguese (and vice versa)
# during mixed-language meetings. Add `--language pt` if meetings are 100% Portuguese.
#
# To find your microphone source name: omniscribe --list
omniscribe --tui \
    --mic alsa_input.usb-HP__Inc_HyperX_Cloud_III_Wireless_0000000000000000-00.mono-fallback \
    --transcribe \
    --whisper-model large-v3 \
    --whisper-device cuda \
    --whisper-compute-type float16 \
    --chunk-seconds 10 \
    --initial-prompt "Reunião em português, mas podem haver alguns termos em inglês, especialmente em nomes próprios de sistemas, aplicações ou estrangeirismos."

# To re-transcribe a recorded file:
# omniscribe --transcribe-file meetings/meeting-YYYYMMDD-HHMMSS.wav \
#     --whisper-model large-v3 --whisper-device cuda --whisper-compute-type float16