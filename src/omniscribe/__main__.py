#!/usr/bin/env python3
"""Entry point for running OmniScribe as a module.

Usage:
    python -m omniscribe --help
    python -m omniscribe --list
    python -m omniscribe -o meeting.wav --transcribe
"""

from omniscribe.recording.cli import main
import sys

if __name__ == "__main__":
    sys.exit(main())
