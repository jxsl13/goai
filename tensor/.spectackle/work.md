---
schema: v1
---

## T-01KYJQ3R68EGDAQCYFT6HJ64MJ Generalize the cache-blocked gather past rank 2
kind: task
state: draft
created: 2026-07-27

SITE: tensor/tensor.go:430 — blk2d := nd == 2 && !rowRuns (in Contiguous) and tensor/tensor.go:347 — blk2d := nd == 2 && t.strides[nd-1] != 1 (in Cast). Both fall through to gatherCast at tensor/tensor.go:153.

WHY HOT: Contiguous() has 343 non-test call sites. Concrete rank-3 chain, once per forward pass per model: nlp/t5.go:76-82 — m.RelBias.Bias(ctx, seq, seq) returns [seq, seq, heads], then bias.Permute(2,0,1) gives [heads, seq, seq] with strides [1, seq*heads, heads], then bias.Contiguous(). Identical code at nlp/t5_decoder.go:73-79 and llamagpu/t5.go:115. nd == 3 and strides[nd-1] == heads != 1, so rowRuns is false AND blk2d is false — naive gatherCast.

DEFECT: gatherCast walks output row-major, advancing the source by strides[nd-1] per element. For the T5 shape that stride is heads (8-32 elements, 32-128 bytes), so consecutive reads land on a different cache line every 1-2 elements while the source plane spans seq*seq*heads*4 bytes — 8MiB at seq=512, heads=8. It sweeps the entire working set once per output row. This is exactly the pattern already fixed for rank 2 and MEASURED at 1.89x (Contiguous) / 1.50x (Cast) on this machine; the fix was simply never generalized past nd == 2.

FIX: generalize the gate to the innermost two axes of ANY rank — when nd >= 2 && strides[nd-1] != 1, loop the outer nd-2 axes with the existing incremental-offset walk and call gatherBlocked2D on each shape[nd-2] x shape[nd-1] plane, passing the plane's base offset and s0 = strides[nd-2], s1 = strides[nd-1]. gatherBlocked2D already takes exactly (rows, cols, s0, s1, off), so there is no signature change — just an outer driver. Apply symmetrically at both :347 and :430. While there, benchmark blk = 32 and blk = 8 against the current blk = 16 (at f32, blk=16 is a 64-byte dst row, exactly one cache line, which is tight and gives only 16 inner iterations per loop setup). ALSO DELETE the stale NOTE(rejected) block at tensor.go:194-198 claiming tiling landed within noise — it contradicts both the shipped gatherBlocked2D and its own commit history.

VALIDATION GATE (benchmark only): NONE of the existing benchmarks cover rank > 2 — all of perf_bench_test.go and core_bench_test.go use rank-2 or contiguous shapes. Write BenchmarkContiguousPermuted3D on New(F32, Shape{256,256,8}).Permute(2,0,1) (the T5 rel-bias shape), BenchmarkContiguousTransposed4D on New(F32, Shape{4,8,128,64}).Transpose(2,3) (the attention shape), BenchmarkCastPermuted3D, and an F64 variant of the 3D case since the tiling win is size-dependent. The EXISTING BenchmarkContiguousStrided / BenchmarkCastStrided must be run as regression guards — the rank-2 path must be untouched.

EXPECTED: 1.5-2.5x on rank-3/4 strided Contiguous/Cast. High confidence that the naive path is taken (the strides were traced by hand and the gate is a literal nd == 2); medium-high on magnitude, anchored on the measured 1.89x/1.50x for the structurally identical rank-2 change on this same machine.

BIT-IDENTITY BAR: none — pure reordering of independent element copies and conversions, the same (i,j) -> dst[i*cols+j] mapping, no accumulation involved. gatherBlocked2D's doc comment at tensor.go:130-134 already makes this argument for rank 2 and it carries over unchanged. Verify with TestContiguousPermuted3DMatchesGeneric asserting exact []float32 equality against gatherCast.

PERFSCAN RULE REQUIRED: a fast path gated on an exact rank or dimension literal where the underlying condition is rank-agnostic. AST shape: an assignment whose RHS is a BinaryExpr{&&} containing X == <int literal> where X is len(<field>) or a variable assigned from it, and where the guarded branch calls a helper ALSO reachable from a default/else branch handling the same dtypes. Report as "dimensionality-gated fast path — check whether the general case is reachable". Related sub-check worth adding at the same time: flag stale NOTE(... rejected) comments whose claim contradicts a currently-live code path.

## T-01M0H8C73BFWX820S798P27FYG Specialize rank-1/rank-2 Tensor AtF64 and SetF64
kind: task
state: active
created: 2026-08-21
parent: P-01M0H89XBWFF0RG24N9673Y3DX
targets: go:tensor.Tensor.AtF64, go:tensor.Tensor.SetF64, go:tensor.Tensor.flatOffset, go:tensor.Storage.atF64, go:tensor.Storage.setF64

Implement a direct rank-1/rank-2 scalar accessor path for Tensor.AtF64 and Tensor.SetF64 on top of the existing typed Storage layout.

First add F32, F64, F16, and BF16 rank-1/rank-2/rank-N correctness and benchmark controls. Preserve exact offsets, strides, conversions, half rounding, mismatch panics, and uninitialized-storage behavior.

Retain the implementation only if three independent count-seven Apple M2 Pro campaigns improve every F32 rank-1/rank-2 common cell by at least 1.15x, do not reduce any F64 common cell below 0.97x, and keep rank-N fallback within 0.97x to 1.03x. Attribute code size and inliner decisions explicitly.

File the compositional inlining-budget mechanism as a generalized perfscan issue, then run tensor tests, tensor race, pure-Go preflight, cgo short validation, and the full PR CI lifecycle.
