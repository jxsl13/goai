#!/usr/bin/env bash
# Industry-standard inference baseline for the goai CUDA throughput comparison
# (worker sub-spec §PERF). Downloads a prebuilt llama.cpp *Vulkan* build (no nvcc
# needed — runs GPU-accelerated on the RTX 3060) and runs llama-bench on the same
# TinyLlama-1.1B GGUF that backend/cuda benchmarks use. Vulkan is chosen because
# the pip CUDA wheels ship only ptxas (no nvcc) so a CUDA llama.cpp can't be built
# here, and no Linux CUDA prebuilt is published; Vulkan on NVIDIA is a legit,
# well-supported GPU backend and goai itself has a Vulkan backend.
#
# NOTE ON FAIRNESS: llama.cpp runs the model QUANTIZED (Q8_0, ~1.1 GiB) whereas
# goai's CUDA path runs F32 (~4.4 GiB). Q8 has ~4x less weight bandwidth, which
# favours llama.cpp (especially memory-bound decode). Treat prefill (compute-
# bound) as the closer comparison; decode gap is expected until goai adds KV-cache
# + on-device quantized matmul.
set -euo pipefail
BUILD="${LLAMACPP_BUILD:-b10012}"
MODEL="${1:-$(dirname "$0")/../models/tinyllama-1.1b-chat-q8_0.gguf}"
DIR="${TMPDIR:-/tmp}/llamacpp-$BUILD"
BIN="$DIR/llama-b*/llama-bench"
if ! ls $BIN >/dev/null 2>&1; then
  mkdir -p "$DIR"
  URL="https://github.com/ggml-org/llama.cpp/releases/download/$BUILD/llama-$BUILD-bin-ubuntu-vulkan-x64.tar.gz"
  echo "downloading $URL"
  curl -sL "$URL" | tar xz -C "$DIR"
fi
BINDIR="$(dirname "$(ls $BIN | head -1)")"
export LD_LIBRARY_PATH="$BINDIR:${LD_LIBRARY_PATH:-}"
echo "=== llama.cpp $BUILD Vulkan — $(basename "$MODEL"), -ngl 99 (all layers on GPU) ==="
"$BINDIR/llama-bench" -m "$MODEL" -ngl 99 -p 32,128 -n 128 -r 3 2>&1 \
  | grep -viE 'ggml_vulkan|load_backend|^ggml_|warn'
