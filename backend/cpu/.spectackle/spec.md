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
