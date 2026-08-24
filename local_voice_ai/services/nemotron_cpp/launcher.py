"""Download the pinned Nemotron GGUF and replace this process with NeMo-Speech.cpp."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger("nemotron-cpp")
logging.basicConfig(level=logging.INFO)

MODEL_REVISION = "ebe59e5a817142986528bbbee5dba8db7b38ed50"
MODEL_FILENAME = "nemotron-speech-streaming-en-0.6b.q8_0.gguf"
MODEL_URL = (
    "https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b/resolve/"
    f"{MODEL_REVISION}/{MODEL_FILENAME}"
)
MODEL_SHA256 = "d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d"


def cache_dir() -> Path:
    base = os.getenv("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "nemo-speech"


def model_path() -> Path:
    return cache_dir() / MODEL_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_model(
    *,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    destination: Path | None = None,
) -> Path:
    """Return a verified local model, downloading it atomically when absent."""
    destination = destination or model_path()
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    partial.unlink(missing_ok=True)
    logger.info("downloading %s", url)
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual_sha256 = _sha256(partial)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Nemotron model checksum mismatch: expected {expected_sha256}, "
                f"received {actual_sha256}"
            )
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    logger.info("saved %s (%.1f MB)", destination, destination.stat().st_size / 1e6)
    return destination


def server_argv(
    model: Path,
    *,
    host: str,
    port: int,
    right_context: int,
) -> list[str]:
    return [
        "nemo-speech",
        "serve",
        "--asr-model",
        str(model),
        "--host",
        host,
        "--port",
        str(port),
        "--no-ui",
        "--asr.backend.gpu",
        "0",
        "--asr.streaming.rnnt_right_context",
        str(right_context),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch native streaming Nemotron Speech")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--right-context",
        type=int,
        choices=(0, 1, 6, 13),
        default=int(os.getenv("NEMOTRON_CPP_RIGHT_CONTEXT", "1")),
        help="80 ms frames of right context; 1 gives about 160 ms model latency",
    )
    args = parser.parse_args(argv)

    model = ensure_model(
        url=os.getenv("NEMOTRON_CPP_MODEL_URL", MODEL_URL),
        expected_sha256=os.getenv("NEMOTRON_CPP_MODEL_SHA256", MODEL_SHA256),
    )
    command = server_argv(
        model,
        host=args.host,
        port=args.port,
        right_context=args.right_context,
    )
    logger.info("starting NeMo-Speech.cpp")
    os.execvp(command[0], command)
    return 0  # pragma: no cover - execvp either replaces the process or raises


if __name__ == "__main__":
    raise SystemExit(main())
