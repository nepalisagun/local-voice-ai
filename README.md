<div align="center">
  <img src="./frontend/.github/assets/template-light.webp" alt="App Icon" width="80" />
  <h1>Local Voice Agent</h1>
  <p>This project's goal is to enable anyone to easily build a powerful, private, local voice AI agent.</p>
  <p>A real-time voice AI assistant — STT, LLM, TTS — running in <strong>one container</strong>, supervised by a single Python parent process. Powered by <a href="https://docs.livekit.io/agents?utm_source=local-voice-ai">LiveKit Agents</a>.</p>
  <p><strong>Local Voice Agent includes hardware-aware profiles, a Jetson-native streaming stack, and a localhost client for remote voice servers.</strong></p>
  <p>To keep up with what I'm building or request new features <a href="https://x.com/intent/follow?screen_name=ShayneParlo">send me a DM on X</a></p>
</div>

## At a glance


> [!TIP]
> Now gives realtime performance on a Jetson Nano!

Local Voice Agent uses a single supervisor with a hardware-aware startup layer.
The launcher detects the platform, memory topology, accelerator, and supported
runtime before it selects models.

- A terminal setup screen shows the detected hardware and the recommended model
  profile before the first download.
- CPU, desktop NVIDIA, Apple Silicon, and Jetson use separate platform settings.
- Jetson Orin uses Qwen3 1.7B Q4, native streaming Nemotron Q8, and Kokoro ONNX.
- A laptop can run the voice UI at `localhost` while the Jetson runs the voice
  stack.
- The supervisor monitors model services and restarts an unhealthy child.
- An optional gateway exposes the managed model APIs through one port.

## Overview

Everything runs as managed children of one Python supervisor (`python -m local_voice_ai serve`):

- **LiveKit server** (Go binary subprocess) for WebRTC signaling — skipped if `LIVEKIT_URL` points at LiveKit Cloud.
- **llama.cpp** (`llama-server` binary subprocess) for the LLM — default model is Gemma 4 E2B (quantization-aware-trained 4-bit, ~2.6 GB); swap it with `LLAMA_HF_REPO=org/repo:quant`. Skipped if `LLAMA_BASE_URL` points elsewhere.
- **Nemotron STT** or **Whisper (faster-whisper)**. Jetson uses the native,
  quantized NeMo-Speech.cpp server and its realtime WebSocket. Other profiles
  can use the Python OpenAI-compatible services.
- **Kokoro TTS** — Python uvicorn child, OpenAI-compatible.
- **LiveKit Agents worker** — the orchestrator child.
- **FastAPI** in the supervisor itself, serving `POST /api/connection-details` (token minting) and the statically-exported Next.js frontend.

Children speak HTTP only over `127.0.0.1`. The image exposes four ports: `8080` (web), `7880`, `7881`, `7882/udp` (LiveKit WebRTC, only if running locally).

## Choose a deployment pattern

| Pattern                         | Best use                                            | Start here                                                              |
| ------------------------------- | --------------------------------------------------- | ----------------------------------------------------------------------- |
| One computer                    | Desktop development or a workstation with a browser | `python3 run.py`                                                        |
| Jetson server and laptop client | Recommended for an Orin Nano voice appliance        | [Run the UI on another computer](#run-the-voice-ui-on-another-computer) |
| Local stack with cloud services | Replace one or more managed children                | [Swapping in cloud providers](#swapping-in-cloud-providers)             |

The built-in frontend remains available on port `8080`. FastAPI serves a static
export, so the Nano does not run a Next.js server in production.

The remote-client pattern runs the frontend on the laptop instead. This pattern
gives the browser a `localhost` microphone origin and keeps the device roles
separate.

## Getting started

From a source checkout, the recommended entry point is the hardware-aware
launcher. It uses only the Python standard library, so it can inspect the
machine and show the model plan before it installs dependencies or downloads
weights:

```bash
python3 run.py
```

The setup launcher supports Python 3.10+, including Jetson's system Python.
The supervised application runtime still requires Python 3.11–3.13. Native
startup checks this requirement separately and reports a clear error.

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
python3 run.py client --server orin.local        # run the UI here, use a remote server
python3 run.py status
python3 run.py logs
python3 run.py down
```

The profile catalog has four conservative model stacks:

| Profile           | Planning target | Configuration                                                                     |
| ----------------- | --------------: | --------------------------------------------------------------------------------- |
| `lean`            |    about 4.7 GB | Qwen3 1.7B Q4 + Whisper Small on CPU + Kokoro ONNX FP16, one 4K slot              |
| `jetson-realtime` |    about 4.7 GB | Qwen3 1.7B Q4 + native Nemotron 0.6B Q8 streaming + Kokoro ONNX FP16, one 4K slot |
| `compact`         |    about 5.5 GB | Gemma 4 E2B Q4 + Nemotron 0.6B FP16 + Kokoro, one 4K slot                         |
| `balanced`        |    about 6.5 GB | Same models with one 16K slot                                                     |

These numbers are planning targets, not hard limits; allocator, driver, and
conversation state add workload-dependent overhead. Auto selection reserves
25% (at least 2 GB) on unified-memory systems and 8% on discrete GPUs. CPU is
capped at `compact`. Jetson Orin Nano uses the Jetson-only `jetson-realtime`
profile for system headroom and low speech latency.

Auto selection makes two separate decisions:

1. Hardware identity selects the platform settings. For example, Jetson model
   data and L4T select the Jetson runtime.
2. The inference-memory budget selects the highest compatible model profile.
   A platform limit can cap this choice.

Thus, the card or device name does not select a model by itself. It selects the
runtime path, and the memory budget selects the model stack.

The model and platform definitions live in
`local_voice_ai/profiles.json`. Platform profiles decide the runtime and device;
model profiles decide weights, context, and estimated memory. Shell variables
and `.env.local` still override profile values.

### Jetson Orin

The Jetson profile supports JetPack 6.2.x and L4T 36.4.x. The launcher reads
both release values from the device. It also makes sure that Docker has the
`nvidia` runtime.

The launcher selects `docker-compose.jetson.yml` for a supported Jetson. This
overlay uses the shared GPU memory and compiles llama.cpp for Orin (SM 8.7).
It does not use the desktop CUDA image.

The Jetson overlay uses host networking for the build and application. This
avoids Docker 28 bridge rules that require the missing `iptable_raw` kernel
module on JetPack 6.2. The model services remain bound to loopback.

On Jetson, the managed llama.cpp server uses port 11435. Port 11434 remains
available for Ollama. The supervisor accepts llama.cpp only when its expected
model appears in the response.

The Jetson profile loads managed services one at a time. It runs native
NeMo-Speech.cpp on CUDA with the Nemotron Q8 model. Microphone PCM enters its
realtime WebSocket as it arrives, and the server returns interim transcripts.
The profile also uses the lower-memory Kokoro ONNX runtime, raw PCM output,
VAD-only endpointing, and one idle agent process. After startup, all services
remain loaded and run together.

The voice UI uses one Qwen inference slot and does not load the repository's
vision projector. These settings preserve memory for speech services.

The image uses Python 3.12 and CUDA-enabled PyTorch 2.7. Its public
[Jetson Containers](https://github.com/dusty-nv/jetson-containers) base is
pinned by digest. Thus, an NGC account is not necessary.

The first build downloads approximately 5.9 GB for the base image and compiles
the two native CUDA servers. The selected models use approximately 2.0 GB.
Keep at least 29 GB of free disk space for the image, models, and build cache.
An external SSD gives better build and model-load performance than a microSD
card. A clean NeMo-Speech.cpp build took approximately 42 minutes on the tested
Orin Nano. Later builds reused the cached native layer.

Start the recommended profile with this command:

```bash
python3 run.py start --profile auto --memory-gb 5.5
```

#### Observed Orin Nano results

These results came from an 8 GB Jetson Orin Nano Super. They are measurements,
not performance guarantees.

- Nemotron returned partial text while the question audio was still arriving.
- The final transcript arrived 1.408 seconds after the question audio ended.
- The agent entered spoken-reply state 3.522 seconds after the audio ended.
- A concurrent LLM, STT, and TTS stress test used 6,498 MB of 7,620 MB.
- All five managed services remained healthy with zero restarts after the test.

#### Run the voice UI on another computer

This pattern is recommended when the Nano acts as a voice appliance. The laptop
runs the frontend, and the Nano runs the voice stack.

```text
Laptop                                              Jetson Orin Nano
┌────────────────────────────┐                      ┌─────────────────────────┐
│ http://localhost:3000      │── token and status ─▶│ FastAPI :8080           │
│ Next.js development server │── WebSocket ────────▶│ LiveKit :7880           │
│ Browser and microphone     │══ WebRTC media ═════▶│ :7881/tcp or :7882/udp  │
└────────────────────────────┘                      │ Nemotron, Qwen, Kokoro  │
                                                    └─────────────────────────┘
```

The Nano still serves a built-in static frontend. The browser does the UI work
on the computer that opens that page. The current static export sends
approximately 335 KB of JavaScript on its first load.

Static file service is small compared with inference. The main benefits of the
remote pattern are the localhost origin and separate device roles.

The localhost client also gives the browser a secure microphone origin. Many
browsers reject microphone access from a plain HTTP address on the LAN.

##### 1. Configure the Nano address

On the Nano, find the LAN address:

```bash
ip -4 -brief address
```

Add the following values to `.env.local` on the Nano. Replace the example
address with the Nano address.

```env
LIVEKIT_URL=ws://192.168.1.40:7880
LIVEKIT_NODE_IP=192.168.1.40
MANAGE_LIVEKIT=1
```

Compose injects `.env.local` into the application container. These values make
LiveKit advertise an address that the laptop can reach.

##### 2. Permit the LAN traffic

The client needs these ports on the Nano:

| Port   | Protocol | Purpose                |
| ------ | -------- | ---------------------- |
| `8080` | TCP      | Token and status API   |
| `7880` | TCP      | LiveKit signaling      |
| `7881` | TCP      | WebRTC fallback media  |
| `7882` | UDP      | Preferred WebRTC media |

On an Ubuntu system, show the UFW state:

```bash
sudo ufw status
```

If UFW is active, permit traffic from the local subnet. Replace the example
subnet with the subnet of the Nano.

```bash
sudo ufw allow proto tcp from 192.168.1.0/24 to any port 7880,7881,8080 comment 'local voice ai'
sudo ufw allow proto udp from 192.168.1.0/24 to any port 7882 comment 'local voice ai media'
```

CAUTION: Do not expose these development ports to the public internet. The
local development credentials are not suitable for a public service.

##### 3. Start the Nano server

```bash
python3 run.py start --profile auto --memory-gb 5.5 --yes
```

Wait until the launcher reports that all five services are ready.

##### 4. Start the laptop client

Install Node.js 20 on the laptop. Then use a source checkout of this repository:

```bash
git clone https://github.com/ShayneP/local-voice-ai.git
cd local-voice-ai
corepack enable
python3 run.py client --server 192.168.1.40
```

If the repository already exists, run only the last two commands from its root
directory.

Open <http://localhost:3000>. Then permit microphone access. The client installs
the frontend packages on its first run.

The client performs a connection test before it starts the frontend. It also
rejects a remote server that advertises a loopback LiveKit address.

The Nano API accepts localhost client origins by default. Set `CLIENT_ORIGINS`
only when a client uses a separate HTTPS origin.

Use a different local UI port when port `3000` is in use:

```bash
python3 run.py client --server 192.168.1.40 --port 3001
```

##### 5. Make sure that the server is reachable

If the client cannot connect, request the Nano status from the laptop:

```bash
curl -fsS http://192.168.1.40:8080/api/status | python3 -m json.tool
```

If this command times out, make sure that UFW permits the laptop subnet. If the
UI connects without audio, make sure that UDP port `7882` is permitted.

To bypass the launcher, use both Compose files:

```bash
docker compose -f docker-compose.yml -f docker-compose.jetson.yml up -d --build
```

The host Python 3.10 process only runs the launcher. The application runs in
the Python 3.12 container.

### Direct container run

Build the current source checkout with Docker Compose:

```bash
docker compose up --build
```

Open <http://localhost:8080>. The first boot downloads the Nemotron + LLM weights — the page shows per-service progress with download sizes, and the terminal logs a compact status heartbeat plus an unmissable “ready” banner when everything is up. Weights are cached in the `models` volume, so later boots are fast and work offline.

### Desktop NVIDIA GPU

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

| Goal                               | Set                                                                                         |
| ---------------------------------- | ------------------------------------------------------------------------------------------- |
| Use LiveKit Cloud                  | `LIVEKIT_URL=wss://your-project.livekit.cloud` (+ `LIVEKIT_API_KEY` / `…_SECRET`)           |
| Use OpenAI for the LLM             | `LLAMA_BASE_URL=https://api.openai.com/v1`, `LLAMA_MODEL=gpt-4o-mini`, `LLAMA_API_KEY=sk-…` |
| Use a remote OpenAI-compatible STT | `STT_BASE_URL=…`, `STT_MODEL=…`, `STT_API_KEY=…`                                            |
| Use a remote OpenAI-compatible TTS | `TTS_BASE_URL=…`, `TTS_API_KEY=…`                                                           |

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

## Troubleshooting

### The startup line shows several gigabytes for a model

The startup detail shows the size of the model cache on disk. It does not show
the resident memory of that process.

Use `tegrastats` during a complete voice turn on Jetson. This command shows
shared memory, GPU activity, temperature, and power under load.

### The laptop client cannot reach the Nano

From the laptop, request `http://<nano-address>:8080/api/status`. If the request
times out, make sure that the Nano firewall permits the laptop subnet.

The Nano must listen on TCP ports `8080`, `7880`, and `7881`. It must also
listen on UDP port `7882`.

### The UI connects, but no audio moves

Make sure that UDP port `7882` is permitted between the laptop and the Nano.
LiveKit uses this port for the preferred WebRTC media path.

### Jetson reports a Docker `iptables` raw-table error

Use `python3 run.py`. The launcher selects the Jetson overlay, which avoids the
unavailable `iptable_raw` kernel module.

### Ollama already uses port 11434

The Jetson profile uses port `11435` for its managed llama.cpp server. Ollama
can continue to use port `11434`.

## Architecture

The supervisor owns the server stack in both deployment patterns. In remote
client mode, only the frontend moves to the laptop.

```
┌──────────────────────── single container ────────────────────────┐
│  python -m local_voice_ai serve                                  │
│  │                                                                │
│  ├── child: livekit-server     (skipped if LIVEKIT_URL external) │
│  ├── child: llama-server       (skipped if LLAMA_BASE_URL ext.)  │
│  ├── child: nemotron native | Python | whisper (if managed)      │
│  ├── child: kokoro native | ONNX (skipped if TTS URL external)   │
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
│  ├─ profiles.json        # model and platform profile catalog
│  ├─ launcher.py          # dependency-free terminal setup presentation
│  ├─ supervisor.py        # async process supervisor
│  ├─ config.py            # env-driven config + manage-X flags
│  ├─ api.py               # FastAPI: token route, status, static frontend
│  ├─ agent.py             # LiveKit Agents worker
│  ├─ nemotron_stt.py      # streaming STT adapter + VAD commit boundary
│  ├─ textnorm.py          # spoken-number normalization for transcripts
│  ├─ wakeword.py          # optional "hey livekit" gate for the agent
│  └─ services/
│     ├─ nemotron/server.py
│     ├─ nemotron_cpp/launcher.py
│     ├─ whisper/server.py
│     ├─ kokoro/server.py
│     └─ kokoro_onnx/server.py
├─ frontend/               # Next.js (configured for static export)
├─ tests/                  # pytest suite
├─ run.py                  # setup, service controls, and localhost client
├─ Dockerfile              # multi-stage build
├─ Dockerfile.jetson       # JetPack 6.2 / Orin CUDA build
├─ docker-compose.yml      # one service (CPU default)
├─ docker-compose.gpu.yml  # NVIDIA overlay: CUDA build + GPU reservation
├─ docker-compose.jetson.yml # Jetson Orin overlay
├─ .github/workflows/      # CI: tests + multi-arch image publish to GHCR
└─ pyproject.toml          # one Python package, one venv
```

## Environment variables

`serve` loads `.env.local` then `.env` from the working directory, so settings
apply to bare-metal runs. Compose also injects the optional `.env.local` file
into the container. Precedence is real environment > `.env.local` > `.env`.
Thus, `FOO=bar python -m local_voice_ai serve` wins over both files.

Put per-machine choices (model, GPU settings) in **`.env.local`** — it's
gitignored, so it won't change the committed defaults or the CPU compose path.
Because children inherit the environment, llama.cpp's own `LLAMA_ARG_*` variables
work there too — e.g. `LLAMA_ARG_CPU_MOE=1` keeps a MoE model's expert weights on
the CPU, which is what makes a 26B-A4B fit alongside STT on a 12 GB card.

When starting through `run.py`, the precedence order is:

1. The shell environment.
2. The `.env.local` file.
3. The selected profile.
4. The committed defaults.

The launcher reports values that override the selected profile before startup.

See `.env` for the full list. The most important ones:

- `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` — local-default; override for cloud.
- `LIVEKIT_NODE_IP` — address that a managed LiveKit server advertises to remote clients.
- `LLAMA_BASE_URL`, `LLAMA_MODEL`, `LLAMA_HF_REPO`, `LLAMA_N_GPU_LAYERS`
- `LLAMA_OFFLINE` — offline LLM startup. Auto by default: once the model is cached, it starts with no internet (skips the Hugging Face lookup); the first run still downloads. Set `LLAMA_OFFLINE=1` to force it, or `0` to always re-check. `LLAMA_MODEL_PATH=/models/…​.gguf` loads a local file directly instead.
- `WAKE_WORD=1` — the agent joins deaf and only starts listening after it hears **“Hey LiveKit”** (on-device detection via [livekit-wakeword](https://github.com/livekit/livekit-wakeword), model baked into the image). `WAKE_WORD_THRESHOLD` (default `0.5`) tunes sensitivity; scores are logged at DEBUG for calibration.
- `STT_PROVIDER` (`nemotron`|`nemotron-cpp`|`whisper`), `STT_BASE_URL`, `STT_MODEL`; `WHISPER_MODEL` picks the faster-whisper model. `STT_DEVICE` can keep Whisper on CPU while the LLM uses the GPU.
- `TTS_PROVIDER` (`kokoro`|`kokoro-onnx`), `TTS_BASE_URL`, `TTS_VOICE`
- `TURN_DETECTION` (`multilingual`|`vad`) and `AGENT_IDLE_PROCESSES`. The `jetson-realtime` profile uses `vad` and one idle process.
- `SEQUENTIAL_STARTUP=1` — load managed services one at a time to reduce temporary memory peaks.
- `WEB_PORT` (default `8080`)
- `CLIENT_ORIGINS` — comma-separated HTTPS origins for separately hosted clients. Localhost client origins work by default.
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

| Route                                                       | Goes to                                                         |
| ----------------------------------------------------------- | --------------------------------------------------------------- |
| `/v1/models`                                                | union of all three listings, each tagged with a `backend` field |
| `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings` | LLM                                                             |
| `/v1/audio/transcriptions`, `/v1/audio/translations`        | STT                                                             |
| `/v1/audio/speech`                                          | TTS                                                             |

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

| Service | Endpoint                                                  |
| ------- | --------------------------------------------------------- |
| LLM     | `http://<host>:11434/v1` — model `gemma-4-e2b`            |
| STT     | `http://<host>:8000/v1` — `POST /v1/audio/transcriptions` |
| TTS     | `http://<host>:8880/v1` — `POST /v1/audio/speech`         |

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
- Kokoro ONNX: <https://github.com/thewh1teagle/kokoro-onnx>
- faster-whisper (Whisper fallback): <https://github.com/SYSTRAN/faster-whisper>
- livekit-wakeword ("hey livekit" detection): <https://github.com/livekit/livekit-wakeword>
