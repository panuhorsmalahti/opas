#!/usr/bin/env bash
# Download the LLMRouterBench results archive into data/ (git-ignored).
#
# Dataset: https://huggingface.co/datasets/NPULH/LLMRouterBench (MIT; the benchmark
# questions inside keep their original licenses). Paper: "LLMRouterBench: A Massive
# Benchmark and Unified Framework for LLM Routing" (Findings of ACL 2026),
# code: https://github.com/ynulihao/LLMRouterBench
#
# The archive is ~1.3 GB. training/prepare_llmrouterbench.py then turns it into a compact
# labels file without extracting the raw model outputs to disk.
set -euo pipefail

cd "$(dirname "$0")/.."
DEST=data/llmrouterbench
URL=https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz
EXPECTED_BYTES=1283503080

mkdir -p "$DEST"
if [ -f "$DEST/bench-release.tar.gz" ] && [ "$(stat -f%z "$DEST/bench-release.tar.gz" 2>/dev/null || stat -c%s "$DEST/bench-release.tar.gz")" = "$EXPECTED_BYTES" ]; then
  echo "already downloaded: $DEST/bench-release.tar.gz"
  exit 0
fi

echo "downloading $URL -> $DEST/bench-release.tar.gz"
# -C - resumes a partial download.
curl -L --fail --retry 5 --retry-delay 5 -C - -o "$DEST/bench-release.tar.gz" "$URL"
echo "done: $(du -h "$DEST/bench-release.tar.gz" | cut -f1)"
