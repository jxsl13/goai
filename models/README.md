# Test model weights (local, gitignored)

Open-weight GGUF models fetched for **on-GPU inference testing** on the RTX 3060
(12 GB VRAM). The `.gguf` files themselves are **not committed** (`models/` is in
`.gitignore`) — this file just records what to fetch and why.

The CUDA backend (`backend/cuda`) runs inference in **f32** (weights uploaded to
the GPU as f32 via `ResidentB`), so the VRAM footprint is ≈ 4 bytes × params. All
choices below sit well under the ~10.7 GB free VRAM.

## Fetched (all open-weight, ungated, Q8_0 GGUF) — verified with `gguf.ReadFile`

| file | arch | layers | heads / kv | embd | ctx | f32 VRAM | CUDA path |
|------|------|--------|-----------|------|-----|----------|-----------|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | qwen2 | 24 | 14 / 2 (GQA) | 896 | 32768 | ~2 GB | needs QKV bias |
| `tinyllama-1.1b-chat-q8_0.gguf` | llama | 22 | 32 / 4 (GQA) | 2048 | 2048 | ~4.4 GB | **runs directly** |
| `qwen2.5-1.5b-instruct-q8_0.gguf` | qwen2 | 28 | 12 / 2 (GQA) | 1536 | 32768 | ~6 GB | needs QKV bias |
| `qwen2.5-3b-instruct-q8_0.gguf` | qwen2 | 36 | 16 / 2 (GQA) | 2048 | 32768 | ~3.4 GB (Q8 resident) | needs QKV bias |

All three parse **and dequantize** cleanly through goai's own `format/gguf`
reader. **TinyLlama-1.1B** (arch=llama, GQA 32:4, no attention bias) is the first
end-to-end GPU target — every op it needs (RMSNorm, RoPE, `GroupedQueryAttention`,
SwiGLU FFN, `Embed`) is already on the CUDA path.

Sources: `Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct-GGUF`,
`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` (HuggingFace `resolve/main`).

**Qwen2.5-3B** (36 layers, dim 2048) is the largest scale validated end-to-end on the
CUDA engine so far — `TestCUDAQwenGenerate` decodes it to coherent text
("Paris. The capital of Spain is Madrid…") at ≈56 tok/s (Q8, alloc-path), the same
config-driven path as 0.5B/1.5B (0.5B ≈210, 1.5B ≈96 tok/s), so the engine generalizes
across a 6× parameter range with no code changes. `TestCUDAQwenFixedMatchesAlloc` also
runs it on the FULL optimized decode (fixed-buffer + device-pos + Q8 + CUDA graph, bias
in the graph body) — **token-for-token identical to the alloc path at 62 tok/s (+10%)**,
proving CUDA-graph capture is correct at 36-layer scale (graph speedups shrink with size:
0.5B +29%, 1.5B +15%, 3B +10%, as larger models become GPU-compute- rather than
launch-bound).

## Fetch command

```sh
mkdir -p models
curl -L -o models/tinyllama-1.1b-chat-q8_0.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
# (see the table for the two Qwen URLs)
```

## Architecture notes for the CUDA path

goai's `nlp.LlamaFromGGUF` / `BPEFromGGUF` load these into the CPU reference
model (Llama/Qwen/Mistral family). For GPU inference the on-device op set
(`RMSNorm`, `RoPE`, `MultiHeadAttention`/`GroupedQueryAttention`, SwiGLU FFN,
`Embed`) already covers a Llama decoder layer.

- **TinyLlama** = arch=llama, GQA 32:4, no attention bias → the first end-to-end
  GPU target: load GGUF → dequant weights to f32 → resident layers → validate
  logits/tokens vs the CPU `nlp.Llama`. All its ops are already on the CUDA path.
- **Qwen2.5** = arch=qwen2, GQA + a **bias on the Q/K/V projections** — needs a
  resident bias-add after the projection matmuls (a small addition using the
  existing `Add`/`ResidentVec`) before it runs on the GPU path.

## Plan (subsequent fires)

1. TinyLlama single-layer parity: dequant one GGUF layer's weights → resident
   CUDA layer → compare against the CPU `nlp.Llama` layer (bit-tolerance).
2. Full-model forward: embedding → N resident layers → final norm → logits;
   compare token-for-token (greedy) against `nlp.Llama` decode.
3. Qwen bias-add, then Qwen models.
4. KV-cache decode on-device (positions via `RoPE` `PosOffset`).
