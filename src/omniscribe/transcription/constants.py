"""Shared constants for Whisper transcription."""

WHISPER_SR = 16000

DEFAULT_HALLUCINATION_PATTERNS = [
    # YouTube/streaming metadata
    "obrigado por assistir",
    "muito obrigado por assistir",
    "muito obrigado, até a próxima",
    "obrigado, até a próxima",
    "thanks for watching",
    "subscribe to the channel",
    "like and subscribe",
    "click the bell",
    "check the description",
    "se inscreva no canal",
    "ative o sininho",
    "acompanhe a avaliação",
    "legendas disponíveis",
    "acesse o nosso site",
    "www.opusdei.pt",

    # Subtitle/captions artifacts
    "webvtt",
    "kind: captions",
    "language: pt",
    "language: en",
    "legenda por",
    "timestamp",

    # Common filler and audio glitches
    "um er um",
    "ah eh ah",
    "[inaudible]",
    "[background noise]",
    "[music]",

    # Bye phrases (often hallucinated)
    "tchau, tchau",
    "tchau, tchau.",
]
