# M2 Metal exact quant residual epilogue — rejected

Date: 2026-08-18  
Hardware: Apple M2 Pro  
Baseline: `f3332361873d27bb93f16e6a5ffab2e03d7e2b36`  
Model: TinyLlama 1.1B Q4_K_M, f16 KV, grouped mixed-QKV prefill

## Hypothesis

Fuse the final f32 residual addition into the lane-zero epilogue of the established M=1
cooperative Q4_K/Q6_K Metal kernels. The prototype preserved quant reduction order, removed the
separate bounded add and scratch-buffer traffic, and left every other shape and format on the
existing fallback.

## Correctness and dispatch proof

The fused output was bit-identical to `QMatMulResident` followed by `BinaryN` for Q4_K and Q6_K,
including odd output tails, nonzero residuals, and TinyLlama production projection shapes. Across
each trained-model invocation, all 76 greedy tokens and all 32,000 checked logits were identical.
The profiled decode graph changed from 44 `binary.add` events to zero and emitted 44 quant
`.acc` events.

## Leaf measurements

Command:

```text
go test ./backend/metal -run '^$' -bench '^BenchmarkMetalQuantAccDecodeLeaf$' -benchtime=10x -count=5 -benchmem
```

Median M=1 boundary results:

| Shape | Split | Fused | Ratio |
|---|---:|---:|---:|
| Q4_K K=2048 N=2048 | 227.5 us | 222.4 us | 1.023x |
| Q6_K K=5632 N=2048 | 310.7 us | 306.8 us | 1.013x |

Both arms allocated 32 B/op with one allocation.

## Trained-model promotion gate

Command:

```text
GOAI_METAL_QUANT_ACC_REAL=1 \
GOAI_TINYLLAMA_GGUF=/Users/john/Desktop/goai/models/tinyllama-1.1b-q4km.gguf \
go test ./llamagpu -run '^TestMetalQuantAccRealModelQualityAndSpeed$' -count=3 -v -timeout=60m
```

Each invocation contained three fresh-decoder, interleaved candidate/control campaigns.

| Invocation | pp64 | pp512 | tg64 |
|---:|---:|---:|---:|
| 1 | 0.9876x | 1.0052x | 1.0235x |
| 2 | 1.0121x | 1.0120x | 1.0206x |
| 3 | 1.0104x | 1.0034x | 1.0140x |

Frozen gates were pp64/pp512 >=0.99x and tg64 >=1.02x in every invocation. Invocation 1 failed
pp64 and invocation 3 failed tg64, so all executable prototype changes were reverted.

## Learning

Removing 44 explicit add dispatches was real but close to the graph-level Amdahl ceiling. The
existing profile attributed 317.958 us of 8.357 ms explicit event time, or 3.8%, to `binary.add`,
while total command duration was 10.621 ms. A 1.013-1.023x leaf improvement therefore did not
reliably transfer to a 1.02x full-token improvement. General validation tooling should combine
profiled stage share, leaf distributions, and repeated end-to-end gates before promoting
dispatch-elimination fusions.

Generalized follow-up: <https://github.com/jxsl13/perfscan/issues/769>
