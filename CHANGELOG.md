# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed

- Reorganized `omniscribe` into feature-oriented packages (`recording/`, `transcription/`, `diarization/`, `ui/`) with thin backward-compatible shims at the old module paths.
- Split `LiveTranscriber` responsibilities into focused modules (CUDA setup, hallucination filtering, audio prep, buffering).
- Added `create_live_transcriber()` factory to centralize Whisper configuration and removed duplicated transcriber construction in CLI/session code.
- Improved hallucination filter: narrowed patterns to reduce false positives, added subtitle/caption artifact detection, added audio glitch and language-mixing patterns. Coverage increased from 73% to 91%.

### Fixed

- Recording loop no longer references an undefined `mix` variable when split-channel mode is enabled.
- `check_inputs()` no longer references an undefined global TUI instance.
- Transcript file flush now happens on every written line, not only when a TUI callback is attached.
- Final transcript post-filter reuses the configured hallucination filter instead of a hardcoded pattern list.
- Removed overly broad `inscreva-se` pattern that filtered legitimate speech; hallucination filter now uses full phrase patterns to reduce false positives.
- Whitespace normalization in hallucination detection now handles variable spacing in transcription output.
