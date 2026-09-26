"""Registers NVIDIA CUDA DLL directories on Windows before anything needs them.

CTranslate2 (faster-whisper) looks up ``cublas64_12.dll`` and
``cudnn_ops64_9.dll`` via the OS DLL search path, but pip/uv installs the
NVIDIA wheels' DLLs under ``site-packages/nvidia/*/bin`` where Windows never
looks. This module must be imported first — before any other import in this
package, and before any third-party package touches CUDA — so
``ultron/__init__.py`` imports it as its very first line.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger


def _find_nvidia_dll_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in sys.path:
        nvidia_root = Path(entry) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for package_dir in nvidia_root.iterdir():
            bin_dir = package_dir / "bin"
            if bin_dir.is_dir():
                dirs.append(bin_dir)
    return dirs


def add_cuda_dll_directories() -> list[Path]:
    """Add every ``nvidia/*/bin`` directory found on sys.path to the DLL
    search path (Windows only). Returns the directories that were added."""
    if os.name != "nt":
        return []

    added: list[Path] = []
    for bin_dir in _find_nvidia_dll_dirs():
        try:
            os.add_dll_directory(str(bin_dir))
        except OSError:
            continue
        added.append(bin_dir)

    if added:
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        new_entries = [str(d) for d in added if str(d) not in path_entries]
        if new_entries:
            os.environ["PATH"] = os.pathsep.join([*new_entries, *path_entries])

    return added


_added = add_cuda_dll_directories()
if _added:
    logger.debug("CUDA DLL directories registered: {}", _added)
