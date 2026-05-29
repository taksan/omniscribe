"""Shared constants for Whisper transcription."""

WHISPER_SR = 16000

DEFAULT_HALLUCINATION_PATTERNS = [
    # YouTube/streaming metadata
    "obrigado por assistir",
    "muito obrigado por assistir",
    "muito obrigado, até a próxima",
    "obrigado, até a próxima",
    "thanks for watching",
    "thank you for watching",
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
    "transcription by castingwords",
    "subtitles by castingwords",

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

    # Closing phrases (often hallucinated on silence)
    "tchau, tchau",
    "tchau, tchau.",
    "bye-bye",
    "bye bye",
    "see you next time",
    "be right back",
    "продолжение следует",  # Russian "to be continued"

    # Recording/platform artifacts
    "this meeting is being recorded",
    "esta reunião está sendo gravada",
    "recording has started",
    "recording has stopped",
    "marketing has stopped",
    "has left the meeting",
    "joined the meeting",

    # Whisper language detection tags (emitted on near-silence)
    "aplicações em português",
    "aplicação em português",
]
