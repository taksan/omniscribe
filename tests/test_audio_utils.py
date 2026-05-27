"""Tests for audio utility functions."""

import numpy as np
import pytest

from omniscribe.recorder import to_stereo, _dbfs, _meter_bar
from omniscribe.transcriber import _to_mono16k


class TestToStereo:
    """Tests for the to_stereo function."""

    def test_mono_to_stereo(self):
        """Test converting mono to stereo duplicates the channel."""
        mono = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = to_stereo(mono[:, None], target=2)
        assert result.shape == (3, 2)
        assert np.allclose(result[:, 0], result[:, 1])
        assert np.allclose(result[:, 0], mono)

    def test_stereo_unchanged(self):
        """Test stereo input with target=2 is unchanged."""
        stereo = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        result = to_stereo(stereo, target=2)
        assert result.shape == (2, 2)
        assert np.allclose(result, stereo)

    def test_4ch_to_2ch(self):
        """Test downsampling 4 channels to 2."""
        quad = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], dtype=np.float32)
        result = to_stereo(quad, target=2)
        assert result.shape == (2, 2)
        assert np.allclose(result, quad[:, :2])


class TestDbfs:
    """Tests for the _dbfs function."""

    def test_silence(self):
        """Test that near-zero returns -120 dBFS."""
        assert _dbfs(0.0) == -120.0
        assert _dbfs(1e-7) == -120.0

    def test_full_scale(self):
        """Test that 1.0 returns 0 dBFS."""
        assert _dbfs(1.0) == pytest.approx(0.0, abs=0.01)

    def test_half_amplitude(self):
        """Test that 0.5 returns ~-6 dBFS."""
        assert _dbfs(0.5) == pytest.approx(-6.02, abs=0.01)

    def test_tenth_amplitude(self):
        """Test that 0.1 returns ~-20 dBFS."""
        assert _dbfs(0.1) == pytest.approx(-20.0, abs=0.01)


class TestMeterBar:
    """Tests for the _meter_bar function."""

    def test_silence(self):
        """Test meter bar for silence."""
        bar = _meter_bar(0.0)
        assert "-120.0dB" in bar
        assert "[" in bar and "]" in bar

    def test_full_scale(self):
        """Test meter bar for full scale."""
        bar = _meter_bar(1.0)
        assert "0.0dB" in bar or "-0.0dB" in bar
        # Should be full of #
        assert bar.count("#") >= 15

    def test_half_amplitude(self):
        """Test meter bar for half amplitude."""
        bar = _meter_bar(0.5)
        assert "-6.0" in bar or "-6.1" in bar


class TestToMono16k:
    """Tests for the _to_mono16k function."""

    def test_mono_48k_to_16k(self):
        """Test resampling mono 48kHz to 16kHz."""
        # Generate 48000 samples at 48kHz = 1 second
        sr = 48000
        t = np.linspace(0, 1, sr, endpoint=False)
        # 440 Hz sine wave
        mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        result = _to_mono16k(mono, src_sr=48000)
        
        # Should be 1/3 the samples (48000 -> 16000)
        assert result.shape[0] == 16000
        assert result.dtype == np.float32

    def test_stereo_48k_to_mono_16k(self):
        """Test down-mixing stereo to mono and resampling."""
        sr = 48000
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        stereo = np.column_stack([left, right]).astype(np.float32)
        
        result = _to_mono16k(stereo, src_sr=48000)
        
        # Should be mono
        assert result.ndim == 1
        # Length should be 1/3 of original
        expected_len = int(len(t) / 3)
        assert abs(result.shape[0] - expected_len) <= 1

    def test_already_16k(self):
        """Test that 16kHz input passes through unchanged."""
        sr = 16000
        t = np.linspace(0, 0.1, int(sr * 0.1), endpoint=False)
        mono = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        result = _to_mono16k(mono, src_sr=16000)
        
        # Should be unchanged
        assert result.shape == mono.shape
        assert np.allclose(result, mono)
