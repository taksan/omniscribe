"""Tests for device detection functions."""

import pytest
from unittest.mock import patch, MagicMock

from omniscribe.recording.devices import (
    looks_like_pulse_source as _looks_like_pulse_source,
    detect_monitor_source,
    detect_default_mic_source,
)


class TestLooksLikePulseSource:
    """Tests for the _looks_like_pulse_source function."""

    def test_monitor_source(self):
        """Test that .monitor suffix is detected."""
        assert _looks_like_pulse_source("alsa_output.monitor") is True
        assert _looks_like_pulse_source("something.monitor") is True

    def test_alsa_input(self):
        """Test that alsa_input prefix is detected."""
        assert _looks_like_pulse_source("alsa_input.usb-Logitech") is True

    def test_alsa_output(self):
        """Test that alsa_output prefix is detected."""
        assert _looks_like_pulse_source("alsa_output.pci-0000") is True

    def test_portaudio_name(self):
        """Test that PortAudio device names are not detected as Pulse."""
        assert _looks_like_pulse_source("default") is False
        assert _looks_like_pulse_source("sysdefault") is False
        assert _looks_like_pulse_source("USB Audio") is False


class TestDetectMonitorSource:
    """Tests for detect_monitor_source function."""

    @patch("omniscribe.recording.devices.shutil.which")
    def test_no_pactl(self, mock_which):
        """Test returns None when pactl is not available."""
        mock_which.return_value = None
        result = detect_monitor_source()
        assert result is None

    @patch("omniscribe.recording.devices.shutil.which")
    @patch("omniscribe.recording.devices.subprocess.check_output")
    def test_success(self, mock_check_output, mock_which):
        """Test successful detection of monitor source."""
        mock_which.return_value = "/usr/bin/pactl"
        mock_check_output.return_value = "alsa_output.pci-0000.analog-stereo\n"
        
        result = detect_monitor_source()
        
        assert result == "alsa_output.pci-0000.analog-stereo.monitor"

    @patch("omniscribe.recording.devices.shutil.which")
    @patch("omniscribe.recording.devices.subprocess.check_output")
    def test_pactl_error(self, mock_check_output, mock_which):
        """Test returns None when pactl fails."""
        mock_which.return_value = "/usr/bin/pactl"
        import subprocess
        mock_check_output.side_effect = subprocess.CalledProcessError(1, "pactl")
        
        result = detect_monitor_source()
        
        assert result is None


class TestDetectDefaultMicSource:
    """Tests for detect_default_mic_source function."""

    @patch("omniscribe.recording.devices.shutil.which")
    def test_no_pactl(self, mock_which):
        """Test returns None when pactl is not available."""
        mock_which.return_value = None
        result = detect_default_mic_source()
        assert result is None

    @patch("omniscribe.recording.devices.shutil.which")
    @patch("omniscribe.recording.devices.subprocess.check_output")
    def test_success(self, mock_check_output, mock_which):
        """Test successful detection of default mic source."""
        mock_which.return_value = "/usr/bin/pactl"
        mock_check_output.return_value = "alsa_input.usb-Logitech_Webcam_C930e.analog-mono\n"
        
        result = detect_default_mic_source()
        
        assert result == "alsa_input.usb-Logitech_Webcam_C930e.analog-mono"
