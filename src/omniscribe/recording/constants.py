"""Shared constants for audio recording."""

SAMPLE_RATE = 48000
CHANNELS = 2
BLOCKSIZE = 1024
DTYPE = "float32"

# Below this peak amplitude, a block is treated as silent (~ -46 dBFS).
SILENCE_THRESHOLD = 0.005
METER_WIDTH = 20
METER_REFRESH_S = 0.25
