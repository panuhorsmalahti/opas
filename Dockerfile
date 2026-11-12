# opastin: LLM router.
#
# Builds the `opastin` server (with the activation-probe router, which links llama.cpp
# for CPU) and the `opastin-encode` tool. Model weights and probe files are not baked
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
    && cargo build --release -p opastin-server -p opastin-encoder \
    && rm -rf crates

COPY crates crates
# Touch sources so cargo rebuilds the workspace crates over the dependency-only stubs.
RUN find crates -name '*.rs' -exec touch {} + \
    && cargo build --release -p opastin-server -p opastin-encoder --locked \
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
