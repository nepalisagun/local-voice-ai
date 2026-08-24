"""Native Nemotron launcher behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

from local_voice_ai.services.nemotron_cpp.launcher import ensure_model, server_argv


def test_model_download_is_atomic_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.gguf"
    source.write_bytes(b"small test model")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = tmp_path / "cache" / "model.gguf"

    result = ensure_model(
        url=source.as_uri(),
        expected_sha256=expected,
        destination=destination,
    )

    assert result == destination
    assert destination.read_bytes() == b"small test model"
    assert not destination.with_suffix(".gguf.part").exists()


def test_verified_cached_model_does_not_touch_source(tmp_path: Path) -> None:
    destination = tmp_path / "model.gguf"
    destination.write_bytes(b"cached")
    expected = hashlib.sha256(destination.read_bytes()).hexdigest()

    assert ensure_model(
        url="file:///does/not/exist",
        expected_sha256=expected,
        destination=destination,
    ) == destination


def test_server_uses_cuda_and_low_latency_right_context(tmp_path: Path) -> None:
    model = tmp_path / "nemotron.gguf"

    argv = server_argv(model, host="127.0.0.1", port=8000, right_context=1)

    assert argv[:2] == ["nemo-speech", "serve"]
    assert argv[argv.index("--asr-model") + 1] == str(model)
    assert argv[argv.index("--asr.backend.gpu") + 1] == "0"
    assert argv[argv.index("--asr.streaming.rnnt_right_context") + 1] == "1"
