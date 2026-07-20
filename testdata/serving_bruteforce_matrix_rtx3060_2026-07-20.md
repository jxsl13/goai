# FRONT B serving — brute-force variant matrix vs incumbents (RTX 3060, TinyLlama-1.1B)

All numbers tok/s, measured 2026-07-20 on the SAME RTX 3060. GoAI = graph-captured batched
decode; vLLM 0.25.1 f16, graphs on, decode-marginal (prefill cancelled), measured this box
(`testdata/vllm_ctx_sweep.py`); llama.cpp single-stream (no continuous batching).

## Decode throughput by concurrency (ctx128)

| Variant (GoAI unless noted)                     | b64  | b256 | b512  | precision vs vLLM |
|-------------------------------------------------|-----:|-----:|------:|-------------------|
| **A1 f16-act / f16-acc** (GoAI best, layers)    | 6354 | 9459 | 10234 | LOWER (f16 accum)  |
| A1 f16-act / f32-acc (matched, layers)          | 4815 | 6972 |  7045 | MATCHED            |
| bgBuild f32-act / f16-acc (layers)              | 5765 | 8268 |  8965 | lower accum, hi act|
| bgBuild f32-act / f16-acc **full-step**         | 5503 | 8019 |  8675 | +logits            |
| bgBuild f32-act / f32-acc **full-step**         | 4571 | 6219 |  6643 | MATCHED +logits    |
| **vLLM** (incumbent, full step)                 | 5096 | 7466 |  8721 | —                 |
| llama.cpp (single-stream, no cont. batching)    | ~244 | ~244 |  ~244 | —                 |

## Decode throughput by context (b64)

| Variant                          | ctx128 | ctx256 | ctx512 |
|----------------------------------|-------:|-------:|-------:|
| GoAI A1 f16-act/f16-acc (layers) |   6354 |   5283 |   3965 |
| GoAI bgBuild f16-acc full-step   |   5503 |   4728 |   3621 |
| **vLLM** (incumbent)             |   5096 |   5020 |   4367 |

## Honest ratios (best GoAI full-step ≈ layers −4% for logits)

- **GoAI best (A1 f16-acc) vs vLLM**: ~1.20× @b64, ~1.22× @b256, ~1.13× @b512 (ctx128).
  Context: ~1.20× @ctx128, ~1.05× @ctx256, ~0.91× @ctx512 (vLLM's FlashDecoding wins long ctx).
- **At MATCHED f32-accumulate precision**: vLLM wins everywhere — GoAI 0.78–0.91×.
- The **f16-accumulate GeForce exploit** (validated token-identical on TinyLlama) is worth
  +24–31% and is the entire source of GoAI's edge; vLLM defaults to f32-accumulate.

## Variants tried and REJECTED (measured negative, this session)

| Lever                         | Result                          | Why |
|-------------------------------|---------------------------------|-----|
| int8 MMQ GEMM                 | 0.71× f16 (prefill)             | device-quant + scale-epilogue > int8-TC gain |
| f32-shared attn K/V tile      | +4% slower                      | doubles shared → halves occupancy |
| split-K attn (short ctx, b512)| slower                          | SM-saturated |
| split-K attn (long ctx, b64/256)| −12/−14%                      | SMs saturate even at b64 |
| WMMA tensor-core decode attn  | 1.47× slower                    | M=8 GQA group wastes 16×16 tile |
| int8/fp8 KV cache             | ≤ f16's 3–7% (won't help)       | attention compute-bound not bandwidth |
| gate+up GEMM fusion           | +0.9% (wash)                    | cuBLAS already efficient |

## Verdict

Continuous-batching CAPABILITY closed (paged KV + admit/evict, bit-identical to eager).
GoAI's **best config beats vLLM ~1.1–1.2× on this GeForce** via a validated f16-accumulate
optimization vLLM doesn't take; at strictly matched precision vLLM's decode kernels lead
~1.1–1.3% (FlashAttention + fused f16 forward). vLLM wins at long context regardless. The
open lever is a fused FlashAttention + fused-f16-forward (engine-maturity work), shared with
the prefill front.
