# Running opastin with Docker

The image contains the `opastin` server and the `opastin-encode` tool, and runs as a
non-root user (uid 10001). Model weights and probe files are not baked in; mount them.

## Build

```sh
docker build -t opastin .                                    # CPU build (x86 images default to AVX2)
docker build -t opastin --build-arg TARGET_CPU=native .      # tuned for the build machine
docker build -f Dockerfile.cuda -t opastin:cuda .            # NVIDIA GPU build (untested in CI)
```

Images from CI are published as `ghcr.io/OWNER/opastin:<tag>` (see
`.github/workflows/release.yml`).

## Run the proxy (random router)

```sh
cp opastin.example.toml opastin.toml        # edit models, tiers, pricing
docker run --rm -p 8080:8080 \
  -e ANTHROPIC_API_KEY \
  -v "$PWD/opastin.toml:/etc/opastin/opastin.toml:ro" \
  opastin
```

| Environment variable | Default in the image | Meaning |
|---|---|---|
| `OPASTIN_CONFIG` | `/etc/opastin/opastin.toml` | Config file |
| `OPASTIN_LISTEN` | `0.0.0.0:8080` | Listen address (overrides `listen` in the config) |
| `RUST_LOG` | `info` | Log level, e.g. `debug`, `opastin_server=debug` |
| API key variables | — | Whatever the config's `api_key_env` names, e.g. `ANTHROPIC_API_KEY` |

Health check: `curl localhost:8080/healthz`.

## Run the probe router

Put the encoder model and the probe in a directory, mount it, and point the config at
it:

```toml
[router]
kind = "probe"
encoder_model = "/models/gguf/Qwen3.5-9B-Q8_0.gguf"
probe = "/models/probes/llmrouterbench-qwen35-9b.json"
lambda = 10.0
```

```sh
docker run --rm -p 8080:8080 -e ANTHROPIC_API_KEY \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/opastin.toml:/etc/opastin/opastin.toml:ro" \
  --memory 12g \
  opastin
```

With the CUDA image, add `--gpus all` (this needs the NVIDIA Container Toolkit on the
host).

**Expected speed:** on CPU the 9B encoder adds roughly 1–4 s to short prompts and much more
to long ones. On macOS, Docker cannot use the Apple GPU at all; run the binary natively
there. See [hardware](hardware.md) for measurements and options.

## Docker Compose

```yaml
services:
  opastin:
    image: ghcr.io/OWNER/opastin:latest
    ports: ["8080:8080"]
    environment:
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    volumes:
      - ./opastin.toml:/etc/opastin/opastin.toml:ro
      - ./models:/models:ro          # only for the probe router
    restart: unless-stopped
```

The slim image has no `curl`, so check health from outside the container
(`curl localhost:8080/healthz`).

## Extract activations in a container

`opastin-encode` is in the image too, e.g. for producing training data on a GPU host:

```sh
docker run --rm --entrypoint opastin-encode \
  -v "$PWD/models:/models:ro" -v "$PWD/data:/data" \
  opastin --model /models/gguf/Qwen3.5-9B-Q8_0.gguf \
  --input /data/llmrouterbench/sample.jsonl --out /data/llmrouterbench/acts-9b --max-tokens 1024
```
