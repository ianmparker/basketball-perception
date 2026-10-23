"""Warn when a machine has an NVIDIA GPU but PyTorch was installed without CUDA.

This bit the project for real. `torch` resolved to a CPU-only wheel on a machine with an
RTX 5070 Ti sitting idle, and nothing said so: every run worked, just 6.8x slower than it
had to be. A silent slow path is worse than a loud failure because nobody investigates it.

`pip install -r requirements.txt` on Windows pulls the CPU wheel by default, so this is
the common case, not an exotic one.
"""

from __future__ import annotations

import shutil
import warnings

CUDA_FIX = (
    "PyTorch was installed without CUDA support, but this machine has an NVIDIA GPU. "
    "Everything will run on the CPU (measured 6.8x slower for ball inference). "
    "Reinstall with:\n"
    "  pip install --index-url https://download.pytorch.org/whl/cu128 "
    "\"torch>=2.7\" torchvision\n"
    "Use cu128 or newer; Blackwell cards (RTX 50xx) need it."
)


def gpu_idle_reason() -> str | None:
    """Return why the GPU is unusable, or None when there is nothing to warn about.

    Deliberately checks for the *NVIDIA driver tool* rather than torch alone: on a
    machine with no GPU at all this stays quiet, so CI and laptops are not spammed.
    """
    try:
        import torch
    except Exception:                                  # torch missing entirely
        return None

    if torch.cuda.is_available():
        return None
    if getattr(torch.version, "cuda", None) is not None:
        # CUDA build but no usable device: a driver or container problem, not ours.
        return None
    if shutil.which("nvidia-smi") is None:
        return None
    return CUDA_FIX


def warn_if_gpu_idle(once: bool = True) -> str | None:
    """Emit the CPU-fallback warning at most once per process."""
    reason = gpu_idle_reason()
    if reason and once:
        global _already_warned
        if _already_warned:
            return reason
        _already_warned = True
    if reason:
        warnings.warn(reason, RuntimeWarning, stacklevel=2)
    return reason


_already_warned = False
