---
schema: v1
prefix: CPU
---

## CPU-002
WHERE the f32-native GEMM path under the SIMD experiment build, the cpu backend SHALL be compared against the f64 reference at rel 2e-3 and abs 1e-4 rather than at tolerance 0.

Rationale: This path accumulates in f32, so it amends the general f64-accumulation rule; observed maximum deviation is about 1e-4 for K up to 128. The default build keeps scalar f64 accumulation and stays bit-exact, f64 is bit-exact in both builds, and convolution through the banded GEMM is untouched. Migrated from the worker spec Iw4 and ADR-01KYCZF2W8F3GS2X0410JSHPKZ.

## PERF-CACHE-GATE-IS-PER-DTYPE-001
IF a cache-sized threshold measured for one element type is reused for another, THEN the implementing agent SHALL re-sweep the threshold for the second type, since the f32 and f64 GEMM pack gates turned over an order of magnitude apart.

Rationale: Packing B turned over between 1MB and 4MB of B for f32 but between 128KB and 512KB for f64. Sharing one gate would have either left a 7.46% win unclaimed or packed f32 where it cost 2.78%.

## PERF-HOIST-PAYS-ONLY-WHERE-THE-STREAM-IS-TIGHT-001
IF a hoist that won inside a packed or blocked kernel is carried to the unblocked twin beside it, THEN the implementing agent SHALL re-measure it there, since the same A widening was worth minus 11 points packed and plus 11 points unpacked.

Rationale: Widening A to f64 once per row block removed real work in the packed f32 GEMM band because B arrives from a compact panel. In the unpacked band the tile already streams B with fourfold line waste, so the extra 4*k pass evicts what the loop is streaming and the hoist costs more than the conversions it removes: +11.71% at n=256 and +11.08% at n=512, both p<=0.002. Reverted.

## PERF-THRESHOLD-IS-STALE-WHEN-ITS-ARM-CHANGES-001
WHEN a kernel behind a size or cost threshold is made faster, the implementing agent SHALL re-sweep that threshold in the same change, because it was calibrated against the old arm and now excludes inputs the new one wins on.

Rationale: The f32 GEMM pack gate was set at 1<<19 elements when packing measured +2.78% at n=256. Hoisting two operand widenings made the packed band about 18% faster and moved the crossover two orders of magnitude, to 1<<12: the same n=256 went from +2.78% to -17.17%, and the stale gate was leaving 8 to 19 percent unclaimed at every size below 1024. Vision gained 8.99% geomean from the re-sweep alone.

## PERF-SWEEP-EVERY-AXIS-THE-GATE-READS-001
IF a gate reads more than one input and only one axis was swept to calibrate it, THEN the implementing agent SHALL sweep each axis the condition names before shipping, since a square sweep moves rows and columns together and can never exercise the row term.

Rationale: The GEMM pack gate reads both m and k*n. Its work term was calibrated on square matrices, where m was always in the hundreds, so the row term went unmeasured and shipped at 32. A row sweep at k=n=512 later showed packing costing 13.59% for f32 and 17.51% for f64 at exactly that value, on the few-rows-wide-B shape decode and attention matmuls use.

## TEST-GUARD-MUST-ASK-THE-KERNELS-QUESTION-001
IF a coverage guard checks a subset of the conditions the code under test actually branches on, THEN the implementing agent SHALL make the guard call the same predicate the code calls, since a partial guard reports coverage the test does not have.

Rationale: The portable GEMM golden guarded only the row gate while the kernel gated on rows AND a work threshold. Its geometries had k*n=72 against a 4096 threshold, so it had not reached the packed band since that threshold was introduced, and the guard reported it as covered throughout. Pointing the guard at the kernel predicate rejected the set immediately.
