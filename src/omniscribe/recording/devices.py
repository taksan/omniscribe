"""PulseAudio/PortAudio device detection and resolution."""

from __future__ import annotations

import shutil
import subprocess

import sounddevice as sd

from .constants import CHANNELS


def list_devices() -> None:
    print("PortAudio devices:")
    print(sd.query_devices())
    print()
    print(f"Default input : {sd.default.device[0]}")
    print(f"Default output: {sd.default.device[1]}")
    monitor = detect_monitor_source()
    mic_src = detect_default_mic_source()
    if monitor:
        print(f"Pulse default monitor (system audio): {monitor}")
    if mic_src:
        print(f"Pulse default source (microphone):    {mic_src}")
    if shutil.which("pactl"):
        print()
        print("All Pulse/PipeWire sources (use any of these with --mic or --system):")
        try:
            out = subprocess.check_output(
                ["pactl", "list", "sources", "short"], text=True
            )
            print(out.rstrip())
        except subprocess.CalledProcessError:
            pass


def detect_monitor_source() -> str | None:
    if not shutil.which("pactl"):
        return None
    try:
        sink = subprocess.check_output(
            ["pactl", "get-default-sink"], text=True
        ).strip()
        if not sink:
            return None
        return f"{sink}.monitor"
    except subprocess.CalledProcessError:
        return None


def detect_default_mic_source() -> str | None:
    if not shutil.which("pactl"):
        return None
    try:
        src = subprocess.check_output(
            ["pactl", "get-default-source"], text=True
        ).strip()
        return src or None
    except subprocess.CalledProcessError:
        return None


def pulse_source_channels(name: str, fallback: int = 1) -> int:
    if not shutil.which("pactl"):
        return fallback
    try:
        out = subprocess.check_output(
            ["pactl", "list", "sources", "short"], text=True
        )
    except subprocess.CalledProcessError:
        return fallback
    for line in out.splitlines():
        cols = line.split("\t")
        if len(cols) >= 5 and cols[1] == name:
            for tok in cols[4].split():
                if tok.endswith("ch"):
                    try:
                        return int(tok[:-2])
                    except ValueError:
                        pass
    return fallback


def looks_like_pulse_source(name: str) -> bool:
    n = name.lower()
    if ".monitor" in n or n.endswith("monitor"):
        return True
    return n.startswith(("alsa_input.", "alsa_output.", "input."))


def resolve_system_device(explicit: int | str | None) -> int | str:
    if explicit is not None:
        if isinstance(explicit, str) and looks_like_pulse_source(explicit):
            if not shutil.which("parec"):
                raise RuntimeError(
                    "Got a PulseAudio source name but 'parec' is not installed. "
                    "Install pulseaudio-utils (Debian/Ubuntu) or pass a PortAudio index."
                )
            return explicit
        return explicit

    monitor = detect_monitor_source()
    if monitor and shutil.which("parec"):
        return monitor

    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        n = dev["name"].lower()
        if "monitor" in n or "loopback" in n or "stereo mix" in n:
            return idx

    raise RuntimeError(
        "Could not auto-detect a system-audio (monitor/loopback) input. "
        "Install pulseaudio-utils for the 'parec' tool, or run with --list "
        "and pass --system <index>."
    )


def resolve_mic_device(explicit: int | str | None) -> int | str:
    if explicit is not None:
        return explicit

    src = detect_default_mic_source()
    if src and shutil.which("parec"):
        return src

    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        return default_in
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            return idx
    raise RuntimeError("No input (microphone) device found.")


def device_channels(device: int | str, desired: int = CHANNELS) -> int:
    info = sd.query_devices(device, "input")
    return min(desired, info["max_input_channels"]) or 1


def device_label(device, kind: str = "input") -> str:
    try:
        info = sd.query_devices(device, kind)
        return f"{info['name']} (idx {device if isinstance(device, int) else '?'})"
    except Exception:  # noqa: BLE001
        return str(device)


def pulse_source_description(name: str) -> str:
    if not shutil.which("pactl"):
        return name
    try:
        out = subprocess.check_output(["pactl", "list", "sources"], text=True)
    except subprocess.CalledProcessError:
        return name
    desc = None
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Description:"):
            desc = s.split(":", 1)[1].strip()
        if s == f"Name: {name}" and desc:
            return desc
        if s == f"Name: {name}":
            for follow in out.splitlines()[out.splitlines().index(line) + 1 :]:
                fs = follow.strip()
                if fs.startswith("Description:"):
                    return fs.split(":", 1)[1].strip()
                if fs.startswith("Source #"):
                    break
    return name
