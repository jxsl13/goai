# M2 CPU quantized-decode refresh — 2026-08-22

This campaign refreshes the stale dim-256 Q8_0-versus-float cell after the
ARM64 fused row-dot campaign and adds a production-size TinyLlama Q4_K_M CPU
measurement boundary. It does not claim cross-library leadership where cache
dtype and token streams differ.

## Result

| Cell | Result | Interpretation |
|---|---:|---|
| Float dim-256, 500 decode steps | 316.21 ms median | Same Go 1.26.6 binary and eight threads |
| Q8_0 dim-256, 500 decode steps | 251.77 ms median | **1.256x faster** than float; the historical 8.8x loss is closed |
| GoAI TinyLlama-1.1B Q4_K_M, 64 steps | 1.9874 s median, 32.203 tok/s | f32 KV, exact digest `ea3df5516f17df83` |
| llama.cpp TinyLlama-1.1B Q4_K_M, 64 steps | 0.5558–0.7259 s process medians, 88.2–115.2 tok/s | f16 KV; diagnostic only, not a matched claim |

The retained dim-256 campaign contains ten process samples after the first
sample of each benchmark was discarded. Float spans 312.81–355.90 ms. Q8_0
spans 243.72–414.24 ms, including the observed high outlier; no retained sample
was removed. Q8_0 uses about 83.8 MB/op versus float's 136.7 MB/op. Its
allocation count is higher (about 247k versus 203k), so allocation reduction is
still an independent optimization front even though wall time now wins.

The production GoAI harness times only 64 forward calls: model loading, cache
construction, and the process-local warmup are excluded. It holds prior caches
live so their collection cannot contaminate later samples, reports allocation
deltas, and checks one exact final-logit digest across every retained sample.

## Incumbent boundary

Both production runs use the identical GGUF SHA-256, eight CPU threads, no GPU
layers, no KV offload, FlashAttention off, and 64 forward steps. They are still
not semantically matched:

- GoAI stores the KV cache as f32; this llama-bench build accepts f16 but rejects
  f32 cache types.
- llama-bench selects its own token stream and starts after one prompt token;
  GoAI uses a fixed documented token sequence starting at position zero.
- llama.cpp reports f16 KV and its own generation boundary; GoAI reports its
  public `QuantLlama.DecodeStep` boundary.

The incumbent's 2.74–3.60x observed throughput advantage is therefore a
directional loss signal, not a leadership ratio. A publishable cross-library
claim requires a shared cache dtype and token stream.

## Rejected pool hypothesis

The base CPU profile showed `pthread_cond_wait`/`pthread_cond_signal` at
84.92%/9.44%. A bounded candidate preserved the existing work-derived fan-out
while replacing per-call goroutine creation with `internal/parallel` workers.
It preserved the exact production digest but measured 2.0712 s versus 2.0700 s
control in one pair and 2.5156 s versus 2.1564 s control in the reversed pair.
Its profile still showed 86.90% wait and 8.49% signal. The candidate was removed:
these samples primarily describe scheduler idleness and park/wake behavior, not
proof that goroutine construction is the bottleneck. The corrected general
lesson is recorded in perfscan issue #827, comment 5381844361.

## Protocol

- Hardware: Apple M2 Pro, 12 cores, 32 GiB; benchmark fan-out fixed to eight.
- OS: macOS 26.5.1 (25F80), darwin/arm64.
- GoAI base: `26e3e3caf8104029504214f09c1d130b37a1c97b`.
- Toolchain: Go 1.26.6; Spectackle 0.10.0.
- Model: `tinyllama-1.1b-q4km.gguf`, SHA-256
  `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0`.
- llama.cpp: v0.2.0 / commit
  `bb4caa7540188872173c44d161602d9271386413`, build 10566; ggml 0.21.0.
- External perfscan: v1.80.0, resolved with `GOPROXY=direct`.
- Tests were selected only on already compiled binaries via `-test.run`; no
  `go test -run` invocation was used.

