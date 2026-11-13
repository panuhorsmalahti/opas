#!/bin/sh
# Install opas: the server binary, the trained predictor, the encoder model and a config.
#
#   curl -fsSL https://raw.githubusercontent.com/panuhorsmalahti/opas/main/scripts/install.sh | sh
#
# Environment variables (all optional):
#   OPAS_VERSION     release tag to install, e.g. v0.1.0 (default: the latest release)
#   OPAS_BIN_DIR     where `opas` and `opas-encode` go (default: ~/.local/bin)
#   OPAS_HOME        predictor and encoder model (default: ~/.local/share/opas)
#   OPAS_CONFIG_DIR  config directory (default: ~/.config/opas)
#   OPAS_SKIP_MODEL  set to 1 to skip the 9.5 GB encoder model download
#
# Supported: Linux x86_64 / arm64 (glibc 2.39+), macOS on Apple Silicon. Elsewhere,
# build from source (see the README) or use the Docker image.
set -eu

REPO="panuhorsmalahti/opas"
MODEL_URL="https://huggingface.co/lmstudio-community/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf"
MODEL_FILE="Qwen3.5-9B-Q8_0.gguf"

BIN_DIR="${OPAS_BIN_DIR:-$HOME/.local/bin}"
OPAS_HOME="${OPAS_HOME:-${XDG_DATA_HOME:-$HOME/.local/share}/opas}"
CONFIG_DIR="${OPAS_CONFIG_DIR:-${XDG_CONFIG_HOME:-$HOME/.config}/opas}"

say() { printf '%s\n' "$*"; }
fail() { printf 'opas install: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || fail "needs \`$1\`"; }

need curl
need tar
need uname

case "$(uname -s)/$(uname -m)" in
  Linux/x86_64) target=x86_64-unknown-linux-gnu ;;
  Linux/aarch64 | Linux/arm64) target=aarch64-unknown-linux-gnu ;;
  Darwin/arm64) target=aarch64-apple-darwin ;;
  MINGW*/* | MSYS*/* | CYGWIN*/*) fail "on Windows, use PowerShell: irm https://raw.githubusercontent.com/$REPO/main/scripts/install.ps1 | iex" ;;
  *) fail "no prebuilt binary for $(uname -s) $(uname -m); build from source (README) or use the Docker image" ;;
esac

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
  fail "needs \`sha256sum\` or \`shasum\` to verify the download"
fi

# The latest release's tag, from the redirect of /releases/latest (no API rate limit).
tag="${OPAS_VERSION:-}"
if [ -z "$tag" ]; then
  url=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest") \
    || fail "could not look up the latest release"
  tag="${url##*/}"
  case "$tag" in v*) ;; *) fail "no release found at https://github.com/$REPO/releases" ;; esac
fi

name="opas-$tag-$target"
base="https://github.com/$REPO/releases/download/$tag"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

say "Installing opas $tag ($target)"
curl -fsSL -o "$tmp/$name.tar.gz" "$base/$name.tar.gz" || fail "download failed: $base/$name.tar.gz"
curl -fsSL -o "$tmp/SHA256SUMS" "$base/SHA256SUMS" || fail "download failed: $base/SHA256SUMS"
want=$(grep " $name.tar.gz\$" "$tmp/SHA256SUMS" | cut -d' ' -f1)
[ -n "$want" ] || fail "$name.tar.gz is not listed in SHA256SUMS"
[ "$(sha256 "$tmp/$name.tar.gz")" = "$want" ] || fail "checksum mismatch for $name.tar.gz"
tar -xzf "$tmp/$name.tar.gz" -C "$tmp"

mkdir -p "$BIN_DIR" "$OPAS_HOME/probes" "$OPAS_HOME/models/gguf" "$CONFIG_DIR"
for bin in opas opas-encode; do
  cp "$tmp/$name/$bin" "$BIN_DIR/$bin.new"
  chmod 755 "$BIN_DIR/$bin.new"
  mv "$BIN_DIR/$bin.new" "$BIN_DIR/$bin"
done
cp "$tmp/$name"/probes/* "$OPAS_HOME/probes/"
say "  binaries:  $BIN_DIR/opas, $BIN_DIR/opas-encode"
say "  predictor: $OPAS_HOME/probes/"

# Config: the example, with absolute paths to the encoder model and predictor. An
# existing config is left alone.
config="$CONFIG_DIR/opas.toml"
if [ -e "$config" ]; then
  say "  config:    $config (kept; the new example is at $CONFIG_DIR/opas.example.toml)"
  cp "$tmp/$name/opas.example.toml" "$CONFIG_DIR/opas.example.toml"
else
  sed -e "s|^encoder_model = \"models/gguf/|encoder_model = \"$OPAS_HOME/models/gguf/|" \
      -e "s|^probe = \"probes/|probe = \"$OPAS_HOME/probes/|" \
      "$tmp/$name/opas.example.toml" > "$config"
  say "  config:    $config"
fi

model="$OPAS_HOME/models/gguf/$MODEL_FILE"
if [ "${OPAS_SKIP_MODEL:-0}" = 1 ]; then
  say "  encoder:   skipped (OPAS_SKIP_MODEL=1); the router needs $model"
elif [ -s "$model" ] && [ ! -e "$model.part" ]; then
  say "  encoder:   $model (already present)"
else
  say "  encoder:   downloading $MODEL_FILE (9.5 GB; resumes if interrupted)"
  # A partial download is kept as .part, so running the installer again resumes it.
  [ -e "$model.part" ] || { [ -e "$model" ] && mv "$model" "$model.part"; } || :
  curl -fL --retry 5 -C - -o "$model.part" "$MODEL_URL" || fail "model download failed; run the installer again to resume"
  mv "$model.part" "$model"
  say "  encoder:   $model"
fi

if [ "$(uname -s)" = Linux ] && command -v ldconfig >/dev/null 2>&1 && ! ldconfig -p | grep -q libgomp.so.1; then
  say ""
  say "Note: opas needs the OpenMP runtime: install libgomp1 (Debian/Ubuntu) or libgomp (Fedora)."
fi

say ""
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "Add $BIN_DIR to your PATH, e.g.: export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
say "Edit models and keys in $config, then start the router:"
say ""
say "  export ANTHROPIC_API_KEY=...   # or leave unset to pass clients' own keys through"
say "  opas"
say ""
say "It listens on http://127.0.0.1:8080 (OpenAI: /v1, Anthropic: the root URL)."
