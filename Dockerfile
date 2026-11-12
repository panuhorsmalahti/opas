# opastin: LLM router.
#
# Builds the `opastin` server (with the activation-probe router, which links llama.cpp
# for CPU) and the `opastin-encode` tool. For NVIDIA GPUs use Dockerfile.cuda; see
# docs/usage/docker.md for hardware guidance. Model weights and probe files are not baked
# into the image: mount them (e.g. at /models) and point the config at them.
#
#   docker build -t opastin .
#   docker run -p 8080:8080 -v $PWD/opastin.toml:/etc/opastin/opastin.toml:ro \
#     -e ANTHROPIC_API_KEY opastin

ARG RUST_VERSION=1.94
ARG DEBIAN_RELEASE=bookworm

FROM rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS build
# cmake builds llama.cpp; clang/libclang are needed by bindgen for its bindings.
RUN apt-get update \
    && apt-get install -y --no-install-recommends cmake clang libclang-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src

# llama.cpp only uses the CPU instruction sets Rust's target-cpu enables. Without one, x86
# builds fall back to baseline SSE and run several times slower. Default for amd64:
# x86-64-v3 (AVX2/FMA, Intel Haswell / AMD Zen and newer). Use TARGET_CPU=native when the
# image is built on the machine it runs on, or x86-64-v4 for AVX-512 servers.
ARG TARGETARCH
ARG TARGET_CPU=""
RUN cpu="$TARGET_CPU"; \
    if [ -z "$cpu" ] && [ "$TARGETARCH" = "amd64" ]; then cpu=x86-64-v3; fi; \
    if [ -n "$cpu" ]; then echo "-C target-cpu=$cpu" > /tmp/rustflags; else : > /tmp/rustflags; fi

# Build dependencies first so source edits don't rebuild llama.cpp.
COPY Cargo.toml Cargo.lock ./
COPY crates/opastin/Cargo.toml crates/opastin/Cargo.toml
COPY crates/opastin-server/Cargo.toml crates/opastin-server/Cargo.toml
COPY crates/opastin-encoder/Cargo.toml crates/opastin-encoder/Cargo.toml
RUN for c in opastin opastin-server opastin-encoder; do \
        mkdir -p crates/$c/src/bin crates/$c/examples && touch crates/$c/src/lib.rs; \
    done \
    && echo 'fn main() {}' > crates/opastin-server/src/main.rs \
    && echo 'fn main() {}' > crates/opastin-encoder/src/bin/encode.rs \
    && for e in decide gateway; do echo 'fn main() {}' > crates/opastin/examples/$e.rs; done \
    && echo 'fn main() {}' > crates/opastin-encoder/examples/overhead.rs \
    && RUSTFLAGS="$(cat /tmp/rustflags)" cargo build --release -p opastin-server -p opastin-encoder \
    && rm -rf crates

COPY crates crates
# Touch sources so cargo rebuilds the workspace crates over the dependency-only stubs.
RUN find crates -name '*.rs' -exec touch {} + \
    && RUSTFLAGS="$(cat /tmp/rustflags)" cargo build --release -p opastin-server -p opastin-encoder --locked \
    && cp target/release/opastin target/release/opastin-encode /usr/local/bin/

FROM debian:${DEBIAN_RELEASE}-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libstdc++6 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --home /nonexistent --shell /usr/sbin/nologin opastin
COPY --from=build /usr/local/bin/opastin /usr/local/bin/opastin-encode /usr/local/bin/
COPY opastin.example.toml /etc/opastin/opastin.toml

USER opastin
ENV OPASTIN_CONFIG=/etc/opastin/opastin.toml \
    OPASTIN_LISTEN=0.0.0.0:8080 \
    RUST_LOG=info
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/opastin"]
