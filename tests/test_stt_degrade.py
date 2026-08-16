"""Tests for STT degradation tracking (the CUDA-OOM silent-death fix)."""

from __future__ import annotations

import pytest
import torch

from local_voice_ai.services.nemotron import server as stt


@pytest.fixture(autouse=True)
def _reset() -> None:
    stt._degraded = None
    stt._consecutive_failures = 0
    yield
    stt._degraded = None
    stt._consecutive_failures = 0


class TestFatalClassification:
    @pytest.mark.parametrize("msg", [
        "CUDA out of memory. Tried to allocate 14.00 MiB",
        "'NoneType' object has no attribute 'replay'",
        "CUDA error: an illegal memory access was encountered",
        "cuBLAS error",
    ])
    def test_fatal_markers(self, msg: str) -> None:
        assert stt._is_fatal(RuntimeError(msg)) is True

    @pytest.mark.parametrize("msg", [
        "Failed to process audio: unknown format",
        "Empty audio file",
        "invalid sample rate",
    ])
    def test_ordinary_errors_are_not_fatal(self, msg: str) -> None:
        # Bad input must never take the process down.
        assert stt._is_fatal(ValueError(msg)) is False

    def test_torch_oom_type_is_fatal(self) -> None:
        assert stt._is_fatal(torch.cuda.OutOfMemoryError("boom")) is True


class TestDegradation:
    def test_single_fatal_error_degrades_immediately(self) -> None:
        stt._note_failure(RuntimeError("CUDA out of memory"))
        assert stt._degraded is not None

    def test_single_ordinary_error_does_not_degrade(self) -> None:
        stt._note_failure(ValueError("bad audio"))
        assert stt._degraded is None

    def test_repeated_ordinary_errors_degrade(self) -> None:
        for _ in range(stt._MAX_CONSECUTIVE_FAILURES):
            stt._note_failure(ValueError("bad audio"))
        assert stt._degraded is not None

    def test_success_resets_the_run(self) -> None:
        # Intermittent bad clips must not accumulate into a false restart.
        for _ in range(stt._MAX_CONSECUTIVE_FAILURES - 1):
            stt._note_failure(ValueError("bad audio"))
            stt._note_success()
        assert stt._degraded is None
        assert stt._consecutive_failures == 0

    def test_degradation_is_sticky(self) -> None:
        # Nothing in-process can clear it: the CUDA context is unrecoverable.
        stt._note_failure(RuntimeError("CUDA out of memory"))
        first = stt._degraded
        stt._note_success()
        assert stt._degraded == first
