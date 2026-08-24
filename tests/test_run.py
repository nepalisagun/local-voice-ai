"""Small, deterministic tests for the repository launcher boundary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import run as run_launcher
from local_voice_ai.profiles import (
    DEFAULT_CATALOG_PATH,
    HardwareInfo,
    load_catalog,
    load_selection,
    resolve_profile,
)


def _cuda_profile():
    hardware = HardwareInfo(
        system="Linux",
        machine="x86_64",
        platform_key="nvidia-cuda",
        device_name="NVIDIA GPU",
        accelerator="cuda",
        memory_topology="discrete",
        total_memory_gib=32.0,
        accelerator_memory_gib=12.0,
    )
    return resolve_profile(load_catalog(DEFAULT_CATALOG_PATH), hardware, requested="compact")


def _jetson_profile(*, jetpack: str = "6.2.1", l4t: str = "36.4.3"):
    hardware = HardwareInfo(
        system="Linux",
        machine="aarch64",
        platform_key="jetson-orin-nano",
        device_name="NVIDIA Jetson Orin Nano",
        accelerator="cuda",
        memory_topology="shared",
        total_memory_gib=7.4,
        jetpack_version=jetpack,
        l4t_version=l4t,
    )
    return resolve_profile(load_catalog(DEFAULT_CATALOG_PATH), hardware, requested="compact")


def test_runtime_environment_precedence() -> None:
    profile = _cuda_profile()

    environment, overrides = run_launcher.runtime_environment(
        profile,
        local_overrides={"LLAMA_CTX_SIZE": "8192", "LOCAL_ONLY": "yes"},
        shell_environment={"DEVICE": "cpu", "SHELL_ONLY": "yes"},
    )

    assert environment["LLAMA_CTX_SIZE"] == "8192"
    assert environment["DEVICE"] == "cpu"
    assert environment["LLAMA_HF_REPO"].startswith("unsloth/gemma-4-E2B")
    assert environment["LOCAL_ONLY"] == "yes"
    assert environment["SHELL_ONLY"] == "yes"
    assert overrides == {"DEVICE": "cpu", "LLAMA_CTX_SIZE": "8192"}


def test_compose_command_uses_platform_overlays() -> None:
    command = run_launcher.compose_command(_cuda_profile(), "up", "-d", "--build")

    assert command == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.gpu.yml",
        "up",
        "-d",
        "--build",
    ]


def test_jetson_compose_command_uses_jetpack_overlay() -> None:
    command = run_launcher.compose_command(_jetson_profile(), "up", "-d", "--build")

    assert command == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.jetson.yml",
        "up",
        "-d",
        "--build",
    ]


def test_supported_jetson_release_and_nvidia_runtime_pass_preflight_check() -> None:
    assert run_launcher.jetson_container_errors(_jetson_profile(), {"nvidia", "runc"}) == []


def test_docker_runtime_parser_accepts_jetson_docker_info_output() -> None:
    raw = (
        '{"io.containerd.runc.v2":{"path":"runc"},'
        '"nvidia":{"path":"nvidia-container-runtime"},'
        '"runc":{"path":"runc"}}'
    )

    assert run_launcher._docker_runtime_names(raw) == {
        "io.containerd.runc.v2",
        "nvidia",
        "runc",
    }


def test_jetson_container_check_explains_an_unsupported_l4t_release() -> None:
    errors = run_launcher.jetson_container_errors(
        _jetson_profile(jetpack="6.0", l4t="36.3.0"),
        {"nvidia", "runc"},
    )

    assert any("L4T 36.3.0" in error and "36.4" in error for error in errors)
    assert any("JetPack 6.0" in error and "6.2" in error for error in errors)


def test_jetson_container_check_requires_nvidia_docker_runtime() -> None:
    errors = run_launcher.jetson_container_errors(_jetson_profile(), {"runc"})

    assert errors == [
        "Docker's nvidia runtime is unavailable. Install or repair nvidia-container-runtime"
    ]


def test_read_env_file_handles_comments_export_and_quotes(tmp_path: Path) -> None:
    path = tmp_path / ".env.local"
    path.write_text(
        "# comment\nexport DEVICE=cuda\nLLAMA_MODEL='custom'\nEMPTY=\n",
        encoding="utf-8",
    )

    assert run_launcher.read_env_file(path) == {
        "DEVICE": "cuda",
        "LLAMA_MODEL": "custom",
        "EMPTY": "",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("192.168.1.40", "http://192.168.1.40:8080"),
        ("orin.local", "http://orin.local:8080"),
        ("http://192.168.1.40:9000/", "http://192.168.1.40:9000"),
        ("https://voice.example/base/", "https://voice.example/base"),
    ],
)
def test_client_backend_url_normalizes_host_or_url(value: str, expected: str) -> None:
    assert run_launcher._client_backend_url(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "ftp://192.168.1.40", "http:///missing-host", "http://user:pass@orin.local"],
)
def test_client_backend_url_rejects_unsafe_or_invalid_values(value: str) -> None:
    with pytest.raises(ValueError):
        run_launcher._client_backend_url(value)


def test_client_stops_cleanly_on_keyboard_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    frontend = tmp_path / "frontend"
    (frontend / "node_modules").mkdir(parents=True)
    monkeypatch.setattr(run_launcher, "FRONTEND", frontend)
    monkeypatch.setattr(run_launcher.shutil, "which", lambda _name: "/usr/bin/pnpm")
    monkeypatch.setattr(
        run_launcher,
        "_client_connection_details",
        lambda _url: {"serverUrl": "ws://192.168.1.40:7880"},
    )
    monkeypatch.setattr(
        run_launcher,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt),
    )

    assert run_launcher._client("192.168.1.40", 3000) == 0
    assert "Client stopped" in capsys.readouterr().out


def test_saved_budget_only_persists_a_manual_override() -> None:
    profile = _cuda_profile()

    automatic = run_launcher.selection_for(profile)
    manual = run_launcher.selection_for(profile, memory_budget_was_set=True)

    assert automatic.memory_budget_gib is None
    assert manual.memory_budget_gib == profile.memory_budget_gib
    assert automatic.detected_memory_gib == 12.0


def test_current_python_is_supported_for_native_runtime() -> None:
    assert run_launcher._native_python_version_error(Path(sys.executable)) is None


def test_noninteractive_configure_saves_the_explicit_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection_path = tmp_path / ".local-voice-ai.toml"
    monkeypatch.setattr(run_launcher, "SELECTION_PATH", selection_path)
    monkeypatch.setattr(run_launcher, "detect_hardware", lambda: _cuda_profile().hardware)

    result = run_launcher.main(["configure", "--profile", "compact", "--memory-gb", "5.5", "--yes"])

    assert result == 0
    selection = load_selection(selection_path)
    assert selection is not None
    assert selection.profile == "compact"
    assert selection.memory_budget_gib == 5.5
