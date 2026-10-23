"""The CPU-fallback warning exists because it silently cost this project 6.8x speed.

These tests pin the discrimination that makes the warning useful: shout when a real GPU
is idle, stay silent on machines without one.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import device_guard


@pytest.fixture(autouse=True)
def _reset_once_flag():
    device_guard._already_warned = False
    yield
    device_guard._already_warned = False


class _FakeCudaVersion:
    cuda = None


class _FakeTorch:
    def __init__(self, available: bool, cuda_build: str | None):
        self.cuda = mock.Mock()
        self.cuda.is_available.return_value = available
        self.version = _FakeCudaVersion()
        self.version.cuda = cuda_build


def _with(torch_obj, smi_present: bool):
    return (
        mock.patch.dict(sys.modules, {"torch": torch_obj}),
        mock.patch("shutil.which", return_value="/usr/bin/nvidia-smi" if smi_present else None),
    )


def test_warns_when_cpu_build_sits_on_an_nvidia_machine():
    fake = _FakeTorch(available=False, cuda_build=None)
    with _with(fake, smi_present=True)[0], _with(fake, smi_present=True)[1]:
        reason = device_guard.gpu_idle_reason()
    assert reason is not None
    assert "cu128" in reason


def test_silent_when_cuda_actually_works():
    fake = _FakeTorch(available=True, cuda_build="12.8")
    with _with(fake, smi_present=True)[0], _with(fake, smi_present=True)[1]:
        assert device_guard.gpu_idle_reason() is None


def test_silent_on_machines_with_no_nvidia_gpu():
    # CI and laptops must not get nagged.
    fake = _FakeTorch(available=False, cuda_build=None)
    with _with(fake, smi_present=False)[0], _with(fake, smi_present=False)[1]:
        assert device_guard.gpu_idle_reason() is None


def test_silent_when_cuda_build_cannot_see_a_device():
    # A CUDA wheel with no visible device is a driver/container problem, not a wrong
    # install, so the reinstall advice would be misleading.
    fake = _FakeTorch(available=False, cuda_build="12.8")
    with _with(fake, smi_present=True)[0], _with(fake, smi_present=True)[1]:
        assert device_guard.gpu_idle_reason() is None


def test_warning_is_emitted_only_once_per_process():
    import warnings

    fake = _FakeTorch(available=False, cuda_build=None)
    with _with(fake, smi_present=True)[0], _with(fake, smi_present=True)[1]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            device_guard.warn_if_gpu_idle()
            device_guard.warn_if_gpu_idle()

    assert len(caught) == 1, "the CPU-fallback notice must not spam every call"
