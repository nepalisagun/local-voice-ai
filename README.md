<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice Agent</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A real-time voice AI assistant — STT, LLM, TTS — running in <strong>one container</strong>, supervised by a single Python parent process. Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
  <p>To keep up with what I'm building or request new features <a href="https://x.com/intent/follow?screen_name=ShayneParlo">send me a DM on X</a></p>
</div>

## Overview

Everything runs as managed children of one Python supervisor (`python -m local_voice_ai serve`):

- **LiveKit server** (Go binary subprocess) for WebRTC signaling — skipped if `LIVEKIT_URL` points at LiveKit Cloud.
- **llama.cpp** (`llama-server` binary subprocess) for the LLM — default model is Gemma 4 E2B (quantization-aware-trained 4-bit, ~2.6 GB); swap it with `LLAMA_HF_REPO=org/repo:quant`. Skipped if `LLAMA_BASE_URL` points elsewhere.
- **Nemotron STT** or **Whisper (faster-whisper)** — Python uvicorn child, OpenAI-compatible.
- **Kokoro TTS** — Python uvicorn child, OpenAI-compatible.
- **LiveKit Agents worker** — the orchestrator child.
- **FastAPI** in the supervisor itself, serving `POST /api/connection-details` (token minting) and the statically-exported Next.js frontend.

Children speak HTTP only over `127.0.0.1`. The image exposes four ports: `8080` (web), `7880`, `7881`, `7882/udp` (LiveKit WebRTC, only if running locally).

## Getting started

From a source checkout, the recommended entry point is the hardware-aware
launcher. It uses only the Python standard library, so it can inspect the
machine and show the model plan before it installs dependencies or downloads
weights:

```bash
python3 run.py
```

On the first run it detects Apple Silicon, a desktop NVIDIA GPU, Jetson, or CPU;
accounts for unified versus discrete memory; and recommends a model profile.
Press Enter to accept, or choose a different profile or memory budget. The
choice is saved in the gitignored `.local-voice-ai.toml`, so later starts do not
repeat the questions.

```bash
python3 run.py configure                         # revisit the setup screen
python3 run.py plan                              # inspect without starting
python3 run.py start --profile auto --yes        # non-interactive auto selection
python3 run.py start --profile compact --memory-gb 5.5 --yes
python3 run.py status
python3 run.py logs
python3 run.py down
```

The initial catalog has two conservative variants of the proven v2 stack:

| Profile | Planning target | Configuration |
|---|---:|---|
| `compact` | about 5.5 GB | Gemma 4 E2B Q4 + Nemotron 0.6B FP16 + Kokoro, 4K context |
| `balanced` | about 6.5 GB | Same models with a 16K context |

These numbers are planning targets, not hard limits; allocator, driver, and
conversation state add workload-dependent overhead. Auto selection reserves
25% (at least 2 GB) on unified-memory systems and 8% on discrete GPUs. CPU and
Jetson Orin Nano are capped at `compact` for latency and system headroom.

The model and platform definitions live in
`local_voice_ai/profiles.toml`. Platform profiles decide the runtime and device;
model profiles decide weights, context, and estimated memory. Shell variables
and `.env.local` still override profile values.

### Jetson

Jetson is detected separately from a desktop NVIDIA GPU because its GPU shares
system RAM and its PyTorch build must match JetPack. The launcher chooses the
native JetPack path and verifies CUDA-enabled PyTorch, `llama-server`, and
`livekit-server` before starting. It deliberately does not apply the desktop
`docker-compose.gpu.yml` image to Jetson. A fully automated JetPack container is
still needed; until then, provision the native dependencies for the installed
JetPack release first.

### Direct container run

To bypass the launcher, run the prebuilt CPU image directly (amd64 + arm64):

```bash
docker run --rm -it \
  -p 8080:8080 -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  -v local-voice-ai-models:/models \
  ghcr.io/shaynep/local-voice-ai:latest
```

Or build from source (also the path for GPU builds):

```bash
docker compose up --build
```

Open <http://localhost:8080>. The first boot downloads the Nemotron + LLM weights — the page shows per-service progress with download sizes, and the terminal logs a compact status heartbeat plus an unmissable “ready” banner when everything is up. Weights are cached in the `models` volume, so later boots are fast and work offline.

### GPU (NVIDIA)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

The overlay swaps in the CUDA llama.cpp binary + CUDA torch wheels, grants the
GPU to the container, and offloads the whole LLM (`LLAMA_N_GPU_LAYERS=999`,
override to partially offload). Requires the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) —
verify with `docker run --gpus all ubuntu nvidia-smi`.

### Apple Silicon

The prebuilt image runs natively (arm64), but **CPU-only** — Docker on macOS is a
VM with no Metal access. For GPU (Metal) inference, run bare-metal via
[Local development](#local-development-no-docker) below, where `llama-server`
picks up Metal automatically.

## Swapping in cloud providers

Each service has a single "manage" decision driven by its base URL — point it at a remote endpoint and the local subprocess is skipped:

| Goal                              | Set                                                                                  |
| --------------------------------- | ------------------------------------------------------------------------------------ |
| Use LiveKit Cloud                 | `LIVEKIT_URL=wss://your-project.livekit.cloud` (+ `LIVEKIT_API_KEY` / `…_SECRET`)   |
| Use OpenAI for the LLM            | `LLAMA_BASE_URL=https://api.openai.com/v1`, `LLAMA_MODEL=gpt-4o-mini`, `LLAMA_API_KEY=sk-…` |
| Use a remote OpenAI-compatible STT| `STT_BASE_URL=…`, `STT_MODEL=…`, `STT_API_KEY=…`                                     |
| Use a remote OpenAI-compatible TTS| `TTS_BASE_URL=…`, `TTS_API_KEY=…`                                                    |

The supervisor logs which children it manages on startup.

## Local development (no Docker)

Requires Python 3.11–3.13 (3.14 is not usable yet: NeMo's `kaldialign` dep ships
no cp314 wheel and no sdist — `.python-version` pins 3.13), plus the
`livekit-server` and `llama-server` binaries on your PATH
(macOS: `brew install livekit llama.cpp`).

On Linux there is no `livekit-server`/`llama-server` package — grab the
[LiveKit release](https://github.com/livekit/livekit/releases) tarball, and for
llama.cpp either take a [prebuilt Linux
binary](https://github.com/ggml-org/llama.cpp/releases) (the `ubuntu-x64` /
`ubuntu-vulkan-x64` tarballs) or build it yourself. There is no prebuilt Linux
CUDA binary — NVIDIA users who want CUDA have to compile:

```bash
cmake -B build -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_INSTALL_PREFIX=$HOME/.local
cmake --build build -j && cmake --install build
```

Set `CMAKE_CUDA_ARCHITECTURES` to your GPU (89 = Ada / RTX 40xx). `LLAMA_CURL=ON`
needs `libcurl4-openssl-dev` and is required — the supervisor starts llama-server
with `--hf-repo`. If CUDA rejects your system compiler, add
`-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`.

```bash
# Python side — uv sync installs from uv.lock into ./.venv
uv sync --extra ml --extra dev

# Prefetch the turn-detector / VAD models (the image does this at build time)
.venv/bin/python -m local_voice_ai.agent download-files

.venv/bin/python -m local_voice_ai serve

# Frontend side, in another shell (only needed if you're editing the UI)
cd frontend && pnpm install && pnpm run dev
```

To use an NVIDIA GPU, mirror what `docker-compose.gpu.yml` sets:

```bash
LLAMA_N_GPU_LAYERS=999 DEVICE=cuda .venv/bin/python -m local_voice_ai serve
```

## Architecture

```
┌──────────────────────── single container ────────────────────────┐
│  python -m local_voice_ai serve                                  │
│  │                                                                │
│  ├── child: livekit-server     (skipped if LIVEKIT_URL external) │
│  ├── child: llama-server       (skipped if LLAMA_BASE_URL ext.)  │
│  ├── child: nemotron | whisper (skipped if STT_BASE_URL ext.)    │
│  ├── child: kokoro             (skipped if TTS_BASE_URL ext.)    │
│  ├── child: livekit-agents worker                                │
│  └── in-process: FastAPI on :8080                                 │
│        ├── POST /api/connection-details  (token minting)         │
│        ├── GET  /api/status              (per-child readiness)   │
│        └── GET  /*                       (static frontend)       │
└───────────────────────────────────────────────────────────────────┘
```

## Project structure

```
.
├─ local_voice_ai/         # Python package: supervisor + agent + services
│  ├─ __main__.py          # python -m local_voice_ai serve
│  ├─ profiles.py          # hardware detection + profile resolution
│  ├─ profiles.toml        # model and platform profile catalog
│  ├─ launcher.py          # dependency-free terminal setup presentation
│  ├─ supervisor.py        # async process supervisor
│  ├─ config.py            # env-driven config + manage-X flags
│  ├─ api.py               # FastAPI: token route, status, static frontend
│  ├─ agent.py             # LiveKit Agents worker
│  ├─ wakeword.py          # optional "hey livekit" gate for the agent
│  └─ services/
│     ├─ nemotron/server.py
│     ├─ whisper/server.py
│     └─ kokoro/server.py
├─ frontend/               # Next.js (configured for static export)
├─ tests/                  # pytest suite
├─ run.py                  # first-run wizard + start/status/log controls
├─ Dockerfile              # multi-stage build
├─ docker-compose.yml      # one service (CPU default)
├─ docker-compose.gpu.yml  # NVIDIA overlay: CUDA build + GPU reservation
├─ .github/workflows/      # CI: tests + multi-arch image publish to GHCR
└─ pyproject.toml          # one Python package, one venv
```

## Environment variables

`serve` loads `.env.local` then `.env` from the working directory, so settings
apply to bare-metal runs as well as Docker. Precedence is real env > `.env.local`
> `.env`, so `FOO=bar python -m local_voice_ai serve` still wins over both.

Put per-machine choices (model, GPU settings) in **`.env.local`** — it's
gitignored, so it won't change the committed defaults or the CPU compose path.
Because children inherit the environment, llama.cpp's own `LLAMA_ARG_*` variables
work there too — e.g. `LLAMA_ARG_CPU_MOE=1` keeps a MoE model's expert weights on
the CPU, which is what makes a 26B-A4B fit alongside STT on a 12 GB card.

When starting through `run.py`, precedence is shell environment > `.env.local`
> selected profile > committed defaults. The launcher reports any values that
override its selected profile before startup.

See `.env` for the full list. The most important ones:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — local-default; override for cloud.
- `LLAMA_BASE_URL`, `LLAMA_MODEL`, `LLAMA_HF_REPO`, `LLAMA_N_GPU_LAYERS`
- `LLAMA_OFFLINE` — offline LLM startup. Auto by default: once the model is cached, it starts with no internet (skips the Hugging Face lookup); the first run still downloads. Set `LLAMA_OFFLINE=1` to force it, or `0` to always re-check. `LLAMA_MODEL_PATH=/models/…​.gguf` loads a local file directly instead.
- `WAKE_WORD=1` — the agent joins deaf and only starts listening after it hears **“Hey LiveKit”** (on-device detection via [livekit-wakeword](https://github.com/livekit/livekit-wakeword), model baked into the image). `WAKE_WORD_THRESHOLD` (default `0.5`) tunes sensitivity; scores are logged at DEBUG for calibration.
- `STT_PROVIDER` (`nemotron`|`whisper`), `STT_BASE_URL`, `STT_MODEL`; `WHISPER_MODEL` picks the faster-whisper model for the whisper provider.
- `TTS_BASE_URL`, `TTS_VOICE`
- `WEB_PORT` (default `8080`)
- `MANAGE_LIVEKIT`, `MANAGE_LLAMA`, `MANAGE_STT`, `MANAGE_TTS` — explicit overrides for the auto-detected "is the URL external?" logic.
- `GATEWAY=1` — serve the OpenAI-compatible `/v1/*` API on the web port, proxied to the LLM/STT/TTS children. Off by default. See below.
- `BIND_HOST` — where the managed LLM/STT/TTS children listen. Defaults to `127.0.0.1`, so they're loopback-only. See below.

## Serving the LLM / STT endpoints on your network

llama.cpp, the STT server, and Kokoro each expose an OpenAI-compatible API, but
all three bind `127.0.0.1`, so nothing else on the network can reach them. There
are two ways to open that up.

### Recommended: the gateway

`GATEWAY=1` puts the whole `/v1/*` surface on the web port (`8080`), proxied to
whichever child owns each route:

```bash
GATEWAY=1 .venv/bin/python -m local_voice_ai serve
```

One base URL — `http://<host>:8080/v1` — serves chat, transcription and speech.
The children stay on loopback, and `8080` is already published by
`docker-compose.yml`, so this needs no `BIND_HOST` and no new port mappings.

| Route | Goes to |
|---|---|
| `/v1/models` | union of all three listings, each tagged with a `backend` field |
| `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | LLM |
| `/v1/audio/transcriptions`, `/v1/audio/translations` | STT |
| `/v1/audio/speech` | TTS |

Streaming is passed through chunk by chunk, so SSE tokens arrive as they're
generated. A backend that's still warming up returns `502` and is omitted from
`/v1/models` rather than failing the whole listing. Any API key works; nothing
is checked.

### Alternative: bind the children directly

To skip the proxy and expose each service on its own port, set `BIND_HOST`:

```bash
BIND_HOST=0.0.0.0 .venv/bin/python -m local_voice_ai serve
```

Per-service `LLAMA_BIND_HOST`, `STT_BIND_HOST`, and `TTS_BIND_HOST` override it
individually — e.g. expose the LLM but keep TTS local:

```bash
BIND_HOST=0.0.0.0 TTS_BIND_HOST=127.0.0.1 .venv/bin/python -m local_voice_ai serve
```

Use `0.0.0.0` for every interface, or a specific NIC address to pick one. Then
point any OpenAI client at the box:

| Service | Endpoint |
|---|---|
| LLM | `http://<host>:11434/v1` — model `gemma-4-e2b` |
| STT | `http://<host>:8000/v1` — `POST /v1/audio/transcriptions` |
| TTS | `http://<host>:8880/v1` — `POST /v1/audio/speech` |

Any API key works; the servers don't check one.

Under Docker this also needs the ports published: compose maps only 8080/7880/7881/7882,
so add `11434:11434` and `8000:8000` to the `ports:` list alongside `BIND_HOST=0.0.0.0`.

> **Neither route adds authentication.** llama-server also runs with `CORS: *`.
> Anyone who can reach the host gets free inference on your GPU. Note the web
> port already defaults to `0.0.0.0`, so `GATEWAY=1` alone publishes inference to
> whatever network that port is reachable from — which is why it's opt-in. Only
> enable either on a network you trust, never on a public interface. Put a
> reverse proxy with auth in front if you need more than that.

## Credits

- LiveKit: <https://livekit.io/>
- LiveKit Agents: <https://docs.livekit.io/agents/>
- NVIDIA Nemotron Speech: <https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b>
- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Gemma 4 (default LLM, Unsloth QAT GGUF): <https://huggingface.co/unsloth/gemma-4-E2B-it-qat-GGUF>
- Kokoro TTS: <https://github.com/hexgrad/kokoro>
- faster-whisper (Whisper fallback): <https://github.com/SYSTRAN/faster-whisper>
- livekit-wakeword ("hey livekit" detection): <https://github.com/livekit/livekit-wakeword>
