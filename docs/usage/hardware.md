# Hardware for the probe router

The random router needs almost nothing. The probe router runs an encoder model
(Qwen3.5-9B, 9.5 GB as Q8_0) over every routed request, so where that model runs decides
the routing latency.

## Measured prefill time, Qwen3.5-9B Q8_0

Measured on an M3 Pro (Apple Silicon, 36 GB) with `opas-encode`:

| Backend | 20-token prompt | ~50-token prompt | 531-token prompt |
|---|---|---|---|
| Metal GPU (native binary) | 0.16 s | 0.2–0.3 s | 2.3 s |
| CPU only (11 cores) | 0.8–1.2 s | 2.8–3.9 s | 19.7 s |

Caveats:

- The CPU numbers were taken while other jobs used the machine, so they are pessimistic,
  but not by an order of magnitude.
- Routed requests add this latency before the upstream call starts.
- Server GPUs (NVIDIA L4, A100, H100) prefill several times faster than a laptop GPU.

**Bottom line:** CPU-only is not enough for the 9B encoder on interactive traffic. Use a
GPU, or make the CPU job much smaller (see "Options" below).

## Why containers are CPU-only by default

- **macOS:** containers run in a Linux VM (Docker Desktop, Podman, Colima) that has no
  access to Metal. The Apple GPU cannot be used from a container.
- **Linux:** the GPU is reachable from containers (NVIDIA Container Toolkit, device
  plugins), but llama.cpp must be compiled for that backend. The default `Dockerfile` is
  a CPU build. `Dockerfile.cuda` builds the CUDA variant.

## Options

| Setup | Encoder speed | Notes |
|---|---|---|
| **NVIDIA GPU, CUDA image** | fastest | `docker build -f Dockerfile.cuda`. Run with `--gpus all`, or in Kubernetes on GPU nodes (`gpu.enabled=true` in the Helm chart). Untested in this repository's CI: there are no GPU runners. |
| **Apple Silicon, bare metal** | fast | Run the `opas` binary natively (no container) on a Mac mini / Mac Studio / Mac Pro; Metal is used automatically. Put a Kubernetes service in front if needed (an `ExternalName` Service, or a Service with manual Endpoints pointing at the Mac). |
| **CPU, tuned** | 5–20× slower than GPU | Fine for batch or offline work, or low-volume traffic with short prompts. See the tuning below. |
| **Split deployment** | — | Run the random or proxy-only server (`--no-default-features`, tiny) where clients are, and the probe router only on GPU hosts. Clients can also use `/v1/route` there and call providers themselves. |

Other llama.cpp backends are exposed as cargo features, but have no Dockerfile in this
repository:

- `vulkan`: AMD / Intel GPUs, and Podman on macOS with GPU passthrough via `libkrun`
  (experimental).
- `rocm`: AMD GPUs.

Build them with `cargo build --release -p opas-server --features vulkan` (or `rocm`).

## Making CPU inference cheaper

1. **Build for the CPU you run on.** llama.cpp only uses the vector instructions Rust's
   `target-cpu` enables. The x86 image defaults to `x86-64-v3` (AVX2/FMA). Use
   `--build-arg TARGET_CPU=native` when building on the target machine, or
   `TARGET_CPU=x86-64-v4` for AVX-512 servers. For native builds outside Docker:
   `RUSTFLAGS="-C target-cpu=native" cargo build --release`.
2. **Shorter prompts:** the probe was trained on prompts truncated to 1024 tokens (head +
   tail). Prefill time grows with length, so long agent contexts cost the most.
3. **Smaller or lower-precision encoder:**
   - A Qwen3.5-4B or -2B encoder needs a probe retrained on its activations (see
     "Training the probe" in the README).
   - A Q4_K_M quantization of the 9B is about 40% smaller and faster on CPU, but its
     activations differ slightly from Q8_0, so retrain or at least re-evaluate the probe.
4. **Enough cores and memory bandwidth:** prefill is compute-bound. Give the pod whole
   cores (`resources.requests.cpu` of 8–16 or more), and about 11 GiB of memory for the
   9B Q8_0 model.
5. **Route less often:** the probe router runs the encoder on every routed request,
   continued conversations included. For agent loops, route once per task with
   `/v1/route` and then call the chosen model directly, or send follow-up turns with the
   chosen model id instead of `auto`.
