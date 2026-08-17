---
schema: v1
---

## T-01KYJREJ34FN9BEZDREAF6WYT8 Fix the broken BenchmarkGPTDecode — it blocks honest sizing of dispatch work
kind: task
state: draft
created: 2026-07-27

BLOCKER, not an optimization. Small, but it gates a whole line of work.

SITE: backend/metal/gpt_test.go:303 BenchmarkGPTDecode.

SYMPTOM: it fails on both sub-benchmarks with: nlp: DecodeStep position 8 does not continue the cache: the next token is at position 9 (cached rows: 9).

WHY IT MATTERS: this is the ONLY benchmark in the tree that measures per-op dispatch cost at decode shapes (seq=1). Prefill benchmarks cannot see it — the researching agent measured BenchmarkGPTForward/metal at 29 ms for roughly 93 ops, so about 200 ns/op of dispatch overhead is 0.06%, i.e. invisible. At decode shapes the same fixed cost is roughly 10% (BenchmarkGemmaDecode averages 406 ns per Execute at dim=16). Without this benchmark the Execute-memoization task cannot be honestly sized, and any claim about it would be an estimate dressed as a measurement.

FIX: diagnose the position/cache-row mismatch. The error says the loop advanced position to 8 while the cache already held 9 rows, so either the benchmark seeds the cache with a prompt and then restarts positions from a stale index, or it reuses one cache across b.N iterations without resetting. Establish which before changing anything — the fix is either resetting per iteration or deriving the position from the cache length rather than from a counter.

VALIDATION: the benchmark runs to completion on both sub-benchmarks and produces a stable tok/s figure across -count=3. Then confirm it actually exercises the intended regime by checking with GOAI_TIME_OPS=1 that per-token Execute counts match the model's layer count, so it is measuring decode and not accidentally re-running prefill.

SCOPE: fix the harness only. Do not optimize anything in the same change — the entire point is to obtain a trustworthy measuring instrument before optimizing, and a benchmark repaired in the same commit as the thing it measures cannot serve as the A/B baseline.

## P-01M08SN9PFECDAK5WMYZ7TSX7H Port half-input float-accumulate Q4_K simdgroup matmul to M2
kind: proposal
state: approved
created: 2026-08-17
grilled: 2026-08-17 open=0
targets: msl:qmatmul_q4k_mm, go:metal.SetQ4KMatrixUnit, objc:metal_bridge.mtl_recorder_qmatmul, backend/metal/q4k_mm_parity_test.go, backend/metal/q4k_mm_crossover_test.go

Context: Current matched M2 attribution leaves GoAI 1.1193x behind llama.cpp b10450 at TinyLlama Q4_K_M pp64. The retained GoAI Q4_K matrix-unit kernel stages dequantized weights and activations as float and uses float simdgroup inputs; after prior hoists it measured 1198.1 us at M64 K2048 N5632 versus 743.0 us for the then-current dequant+GEMM route and stayed disabled.

New incumbent evidence: pinned llama.cpp commit ece963f41b0b02d7a0d61436ae365762c073a4c8 uses half threadgroup staging and simdgroup_half8x8 operands with simdgroup_float8x8 accumulators in its M2 legacy kernel_mul_mm_q4_K path. The current winning GoAI cached-f16 MPS path already converts both the activation and dequantized weight to half, so adopting half matrix inputs changes the disabled candidate toward the production numerical boundary while cutting its 32x64 weight tile plus 32x64 activation tile from 16 KB to 8 KB and total threadgroup storage from roughly 20 KB to 12 KB.

Hypothesis: half-input/float-accumulate staging can remove the previous matrix-unit path dtype and occupancy handicap, potentially beating cached-f16 MPS for the thin M64 Q4_K projections without materializing the full expanded weight.

Prior rejection boundary: R-01M02431SFE9VS94RN2V14AM0P rejected only the float-staged matrix-unit kernel after it reached 1.198 ms; it did not test half simdgroup operands. R-01M022K6X1ENYBCN7YVGXRV53M requires explicit reachability and partial-tile mutation proof. The recently rejected R-01M08S3QNGFE3 rules out conversion-only FFN residency and makes matmul execution itself the target.

Promotion gates on Apple M2: (1) explicit candidate reachability and mutation proof; finite parity against the scalar/current cached-f16 paths across full and partial M/N tiles, Q4_K K multiples, and trained-model logits, with NaN/Inf rejected before aggregate errors; (2) ten alternating warm samples at M64 for K2048xN5632, K2048xN2048, and K5632xN2048 where applicable, requiring at least 1.10x on the Q4_K gate/up leaf and no sampled Q4_K production shape below 0.98x; (3) a rows sweep M32/64/128/256/512 to derive a shape gate rather than globally replacing MPS; (4) ten alternating fresh-decoder TinyLlama Q4_K_M production pairs requiring pp64 >=1.03x, identical greedy argmax, finite logits, unchanged steady-state allocations, controls pp128/pp512/tg64 each >=0.98x. Fully revert executable changes if any promotion gate fails.

## T-01M08SPZPNEG69NCZB79G4R386 Implement and gate M2 Q4_K half-input simdgroup matmul
kind: task
state: active
created: 2026-08-17
parent: P-01M08SN9PFECDAK5WMYZ7TSX7H
grilled: 2026-08-17 open=0
targets: msl:qmatmul_q4k_mm, go:metal.SetQ4KMatrixUnit, go:metal.TestQ4KMatrixUnitMatchesCooperative, go:metal.TestQ4KMatrixUnitHasNoCrossover

Change the retained Metal qmatmul_q4k_mm candidate from float threadgroup operands and simdgroup_float8x8 inputs to half threadgroup operands and simdgroup_half8x8 inputs while preserving float accumulators and f32 output. Keep the existing Q4_K dequant formula, 32x32 tile, bounds handling, and explicit selector for the first pass. Update parity/reachability tests and add an alternating M2 leaf matrix against the current cached-f16 MPS route.

Only after the leaf matrix passes, derive an M/N gate and place the matrix-unit selector ahead of cached-f16 MPS for winning Q4_K shapes. Then run ten alternating fresh-decoder TinyLlama Q4_K_M pairs at pp64 plus pp128/pp512/tg64 controls. Require finite outputs, identical greedy argmax, pp64 >=1.03x, each control >=0.98x, and unchanged steady-state allocations. Fully revert executable changes if any gate fails.

## R-01M08T14CKES181P11RG303W1H M2 Q4_K half-MMA improves the old prototype but is 0.4018x of cached-f16 MPS
kind: research
state: draft
created: 2026-08-17
targets: msl:qmatmul_q4k_mm, go:metal.SetQ4KMatrixUnit

The pinned llama.cpp b10450 evidence was tested as a bounded change to GoAI's retained Q4_K matrix-unit prototype and fully reverted.

Candidate: change 32x64 dequantized-weight and activation threadgroup tiles from float to half; feed simdgroup_half8x8 operands; retain simdgroup_float8x8 accumulators, f32 output, the existing 32x32 output tile, dequant formula, and bounds logic. This reduced live threadgroup storage from about 20 KB to 12 KB and matched the incumbent operand/accumulator split at commit ece963f41b0b02d7a0d61436ae365762c073a4c8.

Correctness: a finite trained-like Q4_K M64 K2048 N5632 fixture produced no NaN/Inf and differed from the current cached-f16 MPS output by max relative 2.947e-4. The existing selector/crossover guard turned red, proving the half-MMA path was live.

Performance: against uncached f32 dequant+GEMM, half-MMA won only at M32 (484.4 us versus 555.4 us, 1.15x), then lost at M48/M64/M96/M128/M256; at M64 it was 841.0 us versus 765.9 us. Against the actual production cached-f16 MPS comparator, a ten-sample alternating run measured 960.125 us half-MMA versus 385.750 us cached-f16, only 0.4018x. The sample ranges were high (83.64%/101.02%), but the 2.49x median loss is independently bounded by the M64 loss to slower f32 dequant+GEMM and by the established f16 MPS advantage at M64. It cannot approach the 1.10x leaf gate.

Learning: the half-input mechanism improves the old float prototype but does not port the incumbent performance. llama.cpp couples it to a different 64x32 output tile, K32 staging layout, dequant placement, and no persistent expanded-weight policy. A mechanism-only port compared the source reason for selecting direct quant matmul with a target that already trades memory for a tuned dense MPS kernel. Any later attempt must port and price the complete tile/layout or target memory footprint explicitly; half dtype alone is closed. Generalized as jxsl13/perfscan#759.
