#!/usr/bin/env python3
"""Entry point for running local-transcriber as a module.

Usage:
    python -m local_transcriber --help
    python -m local_transcriber --list
    python -m local_transcriber -o meeting.wav --transcribe
"""

from local_transcriber.recorder import main
import sys

if __name__ == "__main__":
    sys.exit(main())
