# Running opas with Docker

The image contains the `opas` server and the `opas-encode` tool, and runs as a
non-root user (uid 10001). Model weights and probe files are not baked in; mount them.

## Build

```sh
docker build -t opas .                                    # CPU build (x86 images default to AVX2)
docker build -t opas --build-arg TARGET_CPU=native .      # tuned for the build machine
docker build -f Dockerfile.cuda -t opas:cuda .            # NVIDIA GPU build (untested in CI)
```

Images from CI are published as `ghcr.io/panuhorsmalahti/opas:<tag>` (see
`.github/workflows/image.yml`). Prebuilt binaries are on the GitHub Releases page.

## Run it

The trained predictor ships in the image at `/opt/opas/probes/`. The encoder model
(9.5 GB) does not: download it once (see the README's "Quick start") and mount it at
`/models`. The image's default config is `opas.example.toml` with its paths pointing
there, routing between Claude Opus and Haiku:

```sh
docker run --rm -p 127.0.0.1:8080:8080 \
  -e ANTHROPIC_API_KEY \
  -v "$PWD/models:/models:ro" \
  --memory 12g \
  opas
```

To use your own models, tiers and prices, mount a config:

```sh
cp opas.example.toml opas.toml        # edit models, tiers, pricing
docker run --rm -p 127.0.0.1:8080:8080 -e ANTHROPIC_API_KEY \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/opas.toml:/etc/opas/opas.toml:ro" \
  --memory 12g \
  opas
```

In a mounted config, use the container paths:

```toml
[router]
kind = "probe"
encoder_model = "/models/gguf/Qwen3.5-9B-Q8_0.gguf"
probe = "/opt/opas/probes/llmrouterbench-qwen35-9b.json"   # shipped in the image
weights = { quality = 60, speed = 20, cost = 20 }
```

**Keep the port on localhost:** `-p 127.0.0.1:8080:8080` publishes it on the host's
loopback interface only. opas has no client authentication, so `-p 8080:8080` would
let anyone on your network use your provider API keys.

- **Why the image listens on `0.0.0.0`:** inside a container, `127.0.0.1` is the
  container's own loopback, so binding there would make the server unreachable through
  port publishing. The host-side `127.0.0.1:` prefix is what keeps it local.
- **To reach it from other machines:** put an authenticating reverse proxy in front.

| Environment variable | Default in the image | Meaning |
|---|---|---|
| `OPAS_CONFIG` | `/etc/opas/opas.toml` | Config file |
| `OPAS_LISTEN` | `0.0.0.0:8080` | Listen address (overrides `listen` in the config) |
| `RUST_LOG` | `info` | Log level, e.g. `debug`, `opas_server=debug` |
| API key variables | — | Whatever the config's `api_key_env` names, e.g. `ANTHROPIC_API_KEY` |

Health check: `curl localhost:8080/healthz`.

## GPU, speed and the plain proxy

With the CUDA image, add `--gpus all` (this needs the NVIDIA Container Toolkit on the
host).

**Expected speed:** on CPU the 9B encoder adds roughly 1–4 s to short prompts and much more
to long ones. On macOS, Docker cannot use the Apple GPU at all; run the binary natively
there. See [hardware](hardware.md) for measurements and options.

**Without the encoder:** remove the `[router]` section from your config. opas then runs
as a plain proxy (no model mount, about 20 MB of memory), and requests must name a
configured model instead of `auto`.

## Docker Compose

```yaml
services:
  opas:
    image: ghcr.io/panuhorsmalahti/opas:latest
    ports: ["127.0.0.1:8080:8080"]   # loopback only; opas has no client auth
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    volumes:
      - ./opas.toml:/etc/opas/opas.toml:ro
      - ./models:/models:ro          # the encoder model
    restart: unless-stopped
```

The slim image has no `curl`, so check health from outside the container
(`curl localhost:8080/healthz`).

## Extract activations in a container

`opas-encode` is in the image too, e.g. for producing training data on a GPU host:

```sh
docker run --rm --entrypoint opas-encode \
  -v "$PWD/models:/models:ro" -v "$PWD/data:/data" \
  opas --model /models/gguf/Qwen3.5-9B-Q8_0.gguf \
  --input /data/llmrouterbench/sample.jsonl --out /data/llmrouterbench/acts-9b --max-tokens 1024
```
