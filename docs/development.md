# Development

```sh
cargo test --workspace                                   # unit and end-to-end tests (mock upstream)
cargo clippy --workspace --all-targets -- -D warnings
OPAS_TEST_MODEL=models/gguf/Qwen3.5-0.8B-Q8_0.gguf cargo test --release -p opas-encoder
```

## CI and releases

`.github/workflows/ci.yml` runs on every push and PR:

- format and strict clippy;
- tests on Linux and macOS (Metal);
- a build without llama.cpp;
- the training tools;
- Helm and Kustomize rendering;
- a Docker build.

Other workflows:

- `image.yml` publishes multi-arch images to GHCR on pushes to `main` and on `v*` tags.
- `release.yml` builds `opas` and `opas-encode` for Linux (x86_64, arm64) and macOS
  (Apple Silicon) on `v*` tags, and publishes them with the trained predictor as a
  GitHub Release. To release, bump `version` in `Cargo.toml` and the Helm chart, then
  push a tag: `git tag v0.2.0 && git push origin v0.2.0`.

## Layout

| Path | What |
|---|---|
| `crates/opas` | Library / SDK: request parsing for four APIs, `Router` trait, tiers, cache-aware conversation tracking, config, gateway |
| `crates/opas-server` | The `opas` HTTP server |
| `crates/opas-encoder` | In-process llama.cpp encoder, probe format, `ProbeRouter`, `opas-encode` CLI |
| `training/` | Offline tooling (Python): dataset preparation, probe training; `reference/` MLX cross-check |
| `scripts/` | Installer and dataset download |
| `deploy/` | Kubernetes manifests and Helm chart |
| `docs/` | Usage guides, design notes, training and probe results |
