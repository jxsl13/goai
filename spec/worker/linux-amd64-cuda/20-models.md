## §MODELS — local model assets (models/, gitignored; formerly models/README.md)

M1: files (all open-weight, ungated, HF `resolve/main`):
  | file | arch | L | heads/kv | dim | source |
  |---|---|---|---|---|---|
  | tinyllama-1.1b-chat-q8_0.gguf | llama | 22 | 32/4 | 2048 | TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF |
  | qwen2.5-0.5b-instruct-q8_0.gguf | qwen2 | 24 | 14/2 | 896 | Qwen/Qwen2.5-0.5B-Instruct-GGUF |
  | qwen2.5-1.5b-instruct-q8_0.gguf | qwen2 | 28 | 12/2 | 1536 | Qwen/Qwen2.5-1.5B-Instruct-GGUF |
  | qwen2.5-3b-instruct-q8_0.gguf | qwen2 | 36 | 16/2 | 2048 | Qwen/Qwen2.5-3B-Instruct-GGUF |
  | mistral-7b-instruct-v0.2.Q8_0.gguf | llama | 32 | 32/8 | 4096 | TheBloke/Mistral-7B-Instruct-v0.2-GGUF |

M2: `*q4_k_m*`/`*Q4_K_M*` siblings = LOCAL requants (`llama-quantize --allow-requantize … Q4_K_M`) = the llama.cpp side of the fair Q4-class compare ONLY; goai always loads the Q8_0 files + quantizes to its own resident Q8/Q4 at build.

M3: loaders: llama-family w/ REAL SPM scores (Mistral) → `nlp.SPMFromGGUF` (! ⊥ UnigramFromGGUF, §B59); zero-score files (TheBloke TinyLlama) work w/ either; qwen2 → `nlp.BPEFromGGUF`. qwen2 rejected by `nlp.LlamaFromGGUF` → read config/weights straight from GGUF metadata/tensors (pattern: buildQwenFixed / rawGraphDecoder).

M4: ≥7B: `gguf.ReadRaw` + per-tensor Dequantize→requant→upload (RUN8). VRAM: Q8 TinyLlama ≈1376 MiB total (≈62.5 MiB/layer); Mistral-7B Q8 ≈7.9 GB / Q4 ≈3.9 GB resident — both fit 12 GB.

M5: validated e2e coverage: {TinyLlama, Qwen 0.5/1.5/3B, Mistral-7B} × {f32|Q8|Q4 where K%256 ✓} — see §PERF-SCALEBENCH-2 / §PERF-Q4-SWEEP for the numbers; Qwen-0.5B = Q8-only (dim 896 ⊥ K%256).
