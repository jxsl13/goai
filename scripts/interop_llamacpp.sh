#!/usr/bin/env bash
# GoAI → llama.cpp GGUF interop verification (SPEC §B23 closure; docs/gguf.md
# "Exporting to llama.cpp"). Round-trips a GoAI-WRITTEN GGUF through a REAL
# llama.cpp build and demands it loads AND generates, then compares greedy
# decoding token-for-token against GoAI's own Generate on the same file:
#
#   level A — llama-cli loads + generates from the GoAI-written F32 file;
#   level B — a WriteQuantized Q8_0 file also loads + generates;
#   level C — llama.cpp's greedy completion (llama-simple) is BYTE-IDENTICAL
#             to GoAI's Generate over 9 tokens on the F32 file, and the prompt
#             tokenization matches (llama-tokenize vs BPEFromGGUF).
#
# Q8_0 greedy parity is NOT required: llama.cpp computes directly in quantized
# arithmetic (Q8_0 dot products) while GoAI dequantizes to f64, so near-tie
# argmaxes may legitimately diverge on a random-init tiny model (observed:
# first divergence at generated token 6). Level B is load+generate only.
#
# NOT wired into CI — needs network (clone) and a C++ toolchain (cmake build).
# The llama.cpp checkout+build is cached under $LLAMACPP_SRC (default under
# /tmp) and reused across runs; nothing is installed system-wide.
#
# Usage: scripts/interop_llamacpp.sh
#   LLAMACPP_SRC=/path/to/llama.cpp   reuse an existing checkout/build
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${LLAMACPP_SRC:-${TMPDIR:-/tmp}/goai-llamacpp-interop/llama.cpp}"
BIN="$SRC/build/bin"

# --- llama.cpp: clone + CPU-only build (cached) ------------------------------
if [ ! -d "$SRC/.git" ]; then
  mkdir -p "$(dirname "$SRC")"
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
fi
if [ ! -x "$BIN/llama-cli" ] || [ ! -x "$BIN/llama-simple" ] || [ ! -x "$BIN/llama-tokenize" ]; then
  cmake -S "$SRC" -B "$SRC/build" -DGGML_METAL=OFF -DLLAMA_CURL=OFF >/dev/null
  cmake --build "$SRC/build" -j "$(getconf _NPROCESSORS_ONLN)" \
    --target llama-cli llama-simple llama-tokenize | tail -2
fi

# --- GoAI side: write the tiny GGUFs + GoAI's own greedy outputs -------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/goai-interop.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/spike"
cp "$REPO/scripts/interop_llamacpp_spike.go.txt" "$WORK/spike/main.go"
cat > "$WORK/spike/go.mod" <<EOF
module interopspike

go 1.24

require github.com/jxsl13/goai v0.0.0

replace github.com/jxsl13/goai => $REPO
EOF
(cd "$WORK/spike" && go mod tidy >/dev/null 2>&1 && go run . "$WORK")
echo "goai wrote: tiny-f32.gguf tiny-q8_0.gguf (+ goai-*.bytes, goai-prompt-ids.txt)"

# --- level A: llama-cli loads + generates from the F32 file ------------------
"$BIN/llama-cli" -m "$WORK/tiny-f32.gguf" -p "hello" -n 8 --temp 0 \
  -no-cnv -st --no-warmup -c 64 --ignore-eos </dev/null >"$WORK/cli-f32.log" 2>&1
grep -q "Generation:" "$WORK/cli-f32.log" || {
  echo "FAIL level A: llama-cli produced no generation"; tail -20 "$WORK/cli-f32.log"; exit 1; }
echo "PASS level A: llama-cli loads + generates (F32)"

# --- level B: the WriteQuantized Q8_0 file also loads + generates ------------
"$BIN/llama-cli" -m "$WORK/tiny-q8_0.gguf" -p "hello" -n 8 --temp 0 \
  -no-cnv -st --no-warmup -c 64 --ignore-eos </dev/null >"$WORK/cli-q8.log" 2>&1
grep -q "Generation:" "$WORK/cli-q8.log" || {
  echo "FAIL level B: llama-cli produced no generation from Q8_0"; tail -20 "$WORK/cli-q8.log"; exit 1; }
"$BIN/llama-simple" -m "$WORK/tiny-q8_0.gguf" -n 9 "hello" 2>/dev/null >"$WORK/llama-q8.bytes"
[ -s "$WORK/llama-q8.bytes" ] || { echo "FAIL level B: llama-simple empty output (Q8_0)"; exit 1; }
echo "PASS level B: Q8_0 file loads + generates"

# --- level C: greedy byte parity + prompt tokenization parity (F32) ----------
"$BIN/llama-simple" -m "$WORK/tiny-f32.gguf" -n 9 "hello" 2>/dev/null >"$WORK/llama-f32.bytes"
printf '\n' >>"$WORK/goai-f32.bytes" # llama-simple ends its stdout with a newline
if ! cmp -s "$WORK/goai-f32.bytes" "$WORK/llama-f32.bytes"; then
  echo "FAIL level C: F32 greedy outputs differ"
  echo "  goai:      $(xxd -p "$WORK/goai-f32.bytes")"
  echo "  llama.cpp: $(xxd -p "$WORK/llama-f32.bytes")"
  exit 1
fi
LCPP_IDS="$("$BIN/llama-tokenize" -m "$WORK/tiny-f32.gguf" -p "hello" 2>/dev/null \
  | awk '/->/{printf "%s ", $1}' | sed 's/ $//')"
GOAI_IDS="$(cat "$WORK/goai-prompt-ids.txt")"
if [ "$LCPP_IDS" != "$GOAI_IDS" ]; then
  echo "FAIL level C: prompt tokenization differs (goai: $GOAI_IDS, llama.cpp: $LCPP_IDS)"
  exit 1
fi
echo "PASS level C: F32 greedy completion byte-identical over 9 tokens; tokenization matches ($GOAI_IDS)"

# informational: Q8_0 greedy parity (expected to diverge, not gating)
printf '\n' >>"$WORK/goai-q8_0.bytes"
if cmp -s "$WORK/goai-q8_0.bytes" "$WORK/llama-q8.bytes"; then
  echo "info: Q8_0 greedy outputs happen to match too"
else
  echo "info: Q8_0 greedy outputs differ (expected: quantized vs dequantized-f64 arithmetic)"
fi
echo "OK: GoAI-written GGUF verified against llama.cpp $(git -C "$SRC" rev-parse --short HEAD)"
