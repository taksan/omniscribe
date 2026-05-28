"""CUDA library preloading and availability checks for faster-whisper."""

from __future__ import annotations

import ctypes
import glob
import shutil
import site
import subprocess
import sys
from pathlib import Path


def preload_cuda_libs() -> list[str]:
    """Preload cuBLAS + cuDNN from nvidia pip wheels for ctranslate2."""
    loaded: list[str] = []
    search_roots: list[str] = []

    for base in {sys.prefix, *site.getsitepackages(), site.getusersitepackages()}:
        if base:
            search_roots.append(base)

    for site_path in site.getsitepackages():
        nvidia_base = Path(site_path) / "nvidia"
        if nvidia_base.exists():
            search_roots.append(str(nvidia_base))

    common_paths = [
        Path.home() / ".local" / "lib" / "python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
        Path(sys.executable).parent.parent / "lib" / "python" / f"{sys.version_info.major}.{sys.version_info.minor}" / "site-packages",
    ]
    for path in common_paths:
        if path.exists():
            search_roots.append(str(path))

    patterns = [
        "**/nvidia/cublas/lib/libcublas.so.12",
        "**/nvidia/cublas/lib/libcublasLt.so.12",
        "**/nvidia/cudnn/lib/libcudnn_graph.so.9",
        "**/nvidia/cudnn/lib/libcudnn.so.9",
        "**/nvidia/cudnn/lib/libcudnn_ops.so.9",
        "**/nvidia/cudnn/lib/libcudnn_cnn.so.9",
        "**/nvidia/cuda_nvrtc/lib/libnvrtc.so.12",
        "nvidia/cublas/lib/libcublas.so.12",
        "nvidia/cudnn/lib/libcudnn*.so.9",
    ]

    seen: set[str] = set()
    for root in search_roots:
        for pat in patterns:
            for lib_path in glob.glob(f"{root}/{pat}", recursive=True):
                if lib_path in seen:
                    continue
                seen.add(lib_path)
                try:
                    ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(lib_path)
                except OSError:
                    pass

    return loaded


def check_cuda_available() -> tuple[bool, str]:
    """Check if CUDA is available and return diagnostics."""
    checks: list[str] = []

    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                checks.append(f"GPU detected: {result.stdout.strip()}")
            else:
                checks.append("nvidia-smi found but failed to query GPU")
        except OSError as e:
            checks.append(f"nvidia-smi error: {e}")
    else:
        checks.append("nvidia-smi not found (NVIDIA driver not installed?)")

    try:
        import pkg_resources

        nvidia_packages = [
            d for d in pkg_resources.working_set if d.project_name.startswith("nvidia-")
        ]
        if nvidia_packages:
            checks.append(f"NVIDIA pip packages: {', '.join(p.project_name for p in nvidia_packages)}")
        else:
            checks.append("No nvidia-* pip packages found (try: pip install nvidia-cublas-cu12 nvidia-cudnn-cu12)")
    except Exception as e:
        checks.append(f"Could not check pip packages: {e}")

    cuda_available = any("GPU detected" in c for c in checks)
    return cuda_available, "\n".join(checks)
