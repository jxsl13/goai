---
schema: v1
prefix: FMT
---

## FMT-005 {applies: go:safetensors.FuzzLoad}
IF a format reader receives hostile or malformed input, THEN the reader SHALL return a clean error, never panicking, over-allocating or returning a wrong tensor with a nil error.

Rationale: Applies to every foreign-format reader: gguf, safetensors, npy, npz and pytorch. A wrong-but-nil-error result is as dangerous as a panic because it silently corrupts downstream numerics. Migrated from cavekit SPEC.md V15 and V29.

## FMT-006
The length, dimension and nesting claim read from a file header SHALL be capped before any allocation, with header size, element count and array depth bounded (depth at most 64) and growth never pre-allocated from a claimed size.

Rationale: A hostile header otherwise turns a size field directly into an out-of-memory abort. Migrated from cavekit SPEC.md V15.

## FMT-007 {applies: go:pytorch_test.loadHostile}
The a shape product computed from untrusted dimensions SHALL use the division-before-multiply overflow guard and reject negative dimensions, since a post-multiply cap wraps and lets a hostile header through.

Rationale: No opcode or field parser may index a stack or buffer without an emptiness check either. Migrated from cavekit SPEC.md V29.

## FMT-008 {applies: go:npy.FuzzLoad}
The fuzz target SHALL reach the parser it names, taking raw inner bytes rather than a container envelope.

Rationale: A fuzz target aimed at the zip container left the pickle interpreter behind it effectively unreachable: 966k executions yielded 6 interesting inputs, while a target taking raw inner bytes reached 5.77M and 104. A target that does not reach its parser gives false assurance. Migrated from cavekit SPEC.md V29.

## NUM-ACCUM-NARROW-001
IF a dot product accumulates in float64 and stores its result as float32, THEN the exactness gate SHALL state that it covers element mapping and scale selection, not summation order, as TestQMatMulFusedDecodeMatchesGeneralPathExactly does.

## PERF-FASTPATH-FAMILY-001
IF an early-return path short-circuits only some members of a variant family a switch enumerates, THEN the uncovered variant SHALL be benchmarked before the asymmetry is assumed intentional; 3 such in QMatMul measured 1.40x-1.52x.

## intent
- T-01KYMVPJW5FEAVQY0AX02QC420 Benchmark Q2_K/Q3_K/Q5_K single-token QMatMul before fusing them: DONE, all three fused. Q2_K 1.67x (959.1-965.4us -> 571.2-586.3us), Q3_K 1.75x (1092.5-1103.6us -> 617.7-646.8us), Q5_K 1.41x (794.0-796.6us -> 560.7-570.2us). Interleaved, 4 alternations per quant, one quant at a time, every arm inside 5%. All seven quant types now sit at 102 allocs per decode step; the span across formats collapsed from 525-1082us to 528-626us.

THE TASK'S CAUTION WAS RIGHT TO DEMAND A MEASUREMENT AND WRONG ABOUT THE DIRECTION. It argued the aggressive quants might not repay fusion the way the deployment formats did, because they do more per-element unpacking. They repaid it MORE (1.41-1.75x against 1.40-1.52x). The alloc signature the task said to check first — 107 vs 102 per decode step — was present on all three and was the correct predictor; the reasoning about unpacking cost was not.

PS6003 FALSE-POSITIVED ON THE FIRST DRAFT AND THE CODE WAS CHANGED, NOT THE RULE. Wrapping a dispatch switch in a single `if m == 1` guard hides the covered set behind a function value the AST check cannot follow, so it reported a gap that no longer existed. Naming the five types in the guard keeps coverage legible to the check and meaningful for whoever adds an eighth type. Teaching the rule dataflow would have cost soundness and bought nothing. Second time this campaign that dogfooding PS6003 on fresh code corrected something — the first was its own || handling.

GATE: the three paths were added to TestQMatMulFusedDecodeMatchesGeneralPathExactly before being written. Four mutations turn it red (sign-flipped Q2_K dmin offset, wrong Q2_K activation group, inverted Q3_K high-bit select, Q5_K fifth bit shifted one position) while every pre-existing 1e-5 test stays green. Three further probes were REJECTED as invalid rather than counted: two anchors matched both a dequant and its fused twin, one did not compile.

LANDED: commit 621f4b68. Learning in PERF-FASTPATH-FAMILY-001 and NUM-ACCUM-NARROW-001; research R-01KYMVGRENEND is now fully consumed. NOT DONE, deliberately: no fusion for m>1 (it would re-dequantize per activation row), and no SIMD or integer-accumulator rewrite of any dot — that trades the bit-identity these gates rest on.
- T-01KYMZC07EFT6R50S1THRKFCZB Register-block the six unblocked fused row dots, as Q8_0 already is: DONE, all six unblocked dots register-blocked by 4. Measured interleaved, one quant at a time, 4-6 alternations each: Q4_0 1.55x, Q4_K 1.49x, Q6_K 1.40x, Q5_K 1.34x, Q2_K 1.34x, Q3_K 1.13x. With Q8_0's 2.26x (which arrived from main and started this), all seven types now share each activation load across four accumulators.

THE PREDICTION HELD, and it held for the stated reason. The task argued the K-quants would gain LESS than Q4_0 because their per-element unpack is heavier, so the shared-activation load is a smaller share of their work. The measured ordering tracks unpacking cost monotonically: Q8_0 2.26 > Q4_0 1.55 > Q4_K 1.49 > Q6_K 1.40 > Q5_K 1.34 = Q2_K 1.34 > Q3_K 1.13. This is the first prediction in this campaign that was not inverted by measurement — the two earlier ones were, which is why it was measured per type rather than extrapolated from Q4_0.

Q3_K IS THE FLOOR AND SHOWS WHERE THE TRANSFORM STOPS PAYING: its per-element high-bit select is a BRANCH, evaluated once per row, so blocking runs it four times per shared load. The load amortizes; the branch does not. A kernel whose per-element cost is control flow rather than arithmetic will not repay register blocking. That is the transferable boundary, not the individual numbers.

METHOD NOTE worth carrying: three of the six A/B sets had a first-alternation warm-up outlier (Q4_0 303us against 315-320, Q6_K 304 against 318-324, Q5_K 287 against 324-329) that pushed the OLD arm's spread past the 5% bar. Discarding the first alternation, not the set, put every arm back inside bar. Reported rather than silently dropped, since the discard direction favors the change and a reader should be able to check it.

STRUCTURE: dot4 sits beside dot in the dispatch, nil meaning "still one row at a time", so partial completion was shippable per type — four separate commits, each with its own measurement, rather than one unverifiable batch. Blocking happens within [lo,hi) because parallelRows owns the row partition; blocking across n would write outside the chunk and race. All verified under -race.

GATE: TestQMatMulFusedDecodeMatchesGeneralPathExactly covers all seven types and stayed green throughout — it is what caught nothing here, which is the point of having written it before the fusion work rather than after.

LANDED: commits 1881bfb2 (Q4_K), 301da651 (Q6_K), 6837972e (Q5_K/Q2_K/Q3_K), cf9f5de0 (Q4_0). Generalized as perfscan PS6005. NOT DONE, deliberately: the m>1 general path at quant_matmul.go:360, which PS6005 still flags — it re-reads a dequantized weight row across activation rows, which is the classic unroll-and-jam on a prefill workload rather than decode, a different loop with its own benchmark. Also not done: any SIMD or integer-accumulator rewrite, which would trade the bit-identity these gates rest on.
- T-01KYN0ADDNER9BVJ0HNH8ENZ0X Unroll-and-jam the m>1 QMatMul general path over activation rows (PS6005, prefill): REDIRECTED, not completed as written, and the redirect is the finding. The task asked for an unroll-and-jam of the activation-row loop. That was ALREADY DONE — main had landed it — and the PS6005 finding that motivated the task points at the `default:` AtF64 branch, a declined-shape fallback, not the optimized f32/f64 arms. Reading the code before implementing is what caught it.

WHAT THE PROFILE FOUND INSTEAD WAS LARGER. Quantized prefill did not scale at all across GOMAXPROCS 1..12 (22.1ms vs 22.6ms), and total CPU samples matched wall time on a 12-core host — barely one core busy. QMatMul's weight-row loop was the serial spine. Parallelizing it over weight rows: SERIAL 21.28-21.62ms -> PARALLEL 8.43-9.13ms, 2.52x interleaved over 4 alternations, and it now scales 20.8/14.0/9.8/8.5ms at 1/2/4/12 Ps.

THE BLOCKER WAS THE SHARED SCRATCH, not the loop shape. One dequant buffer was reused across all weight rows — the very optimization that made the serial path fast is what made it unparallelizable. Moving it per-chunk costs prefill allocs 907 -> 972 and unlocks the 2.5x. An earlier optimization becoming the next one's obstacle is worth expecting rather than being surprised by.

THE UNSUPPORTED-TYPE REJECTION had to move above the loop: it must stay an error return and a chunk body on a pool worker has nowhere to return one to. Cheaper there anyway.

BENCHMARK BUILT FIRST AND PANIC-PROBED. No pre-existing quantized benchmark entered this loop at all — every one is single-token and takes a fused m==1 path. The probe confirms the new BenchmarkQuantMamba2Prefill reaches the general path and the decode benchmark does not. Decode verified unregressed by interleaved before/after (209-220us vs 213-214us).

THE FROZEN-ORACLE CONCERN THE TASK RAISED DID NOT MATERIALIZE, and here is why, so nobody re-derives it: the change reorders WHICH goroutine computes each output, not how any output is computed. TestQMatMulFusedDecodeMatchesGeneralPathExactly compares m==1 against m==2 row 0, and both sides still run the same arithmetic in the same order — no self-fulfilling comparison is possible from a partition change. A frozen reference WOULD be required for a change that alters the general path's arithmetic; this one does not.

LANDED: commit 9b96f04a. STILL OPEN: the `default:` AtF64 fallback that PS6005 flags — deliberately left, it is the declined-shape path for non-contiguous tensors and carries no benchmark. Also not done: any wider parallelization of the surrounding prefill, where a profile shows the remaining time is scheduler churn from OTHER ops' pools (backend/cpu zone, not this one).
- T-01KYQK0VZ1EQYAC9Z6RKBK5Z2Y Re-integrate the 4-row K-quant dot kernels on main's parallelized QMatMul (currently dead code): Shipped. Q6_K m=1 86.8us -> 43.7us (1.96-2.44x), Q2_K m=1 ~100us -> ~57us (1.73-2.19x), M2
Pro darwin/arm64, 3 alternations, min of 3 runs of 2000x per arm. Bit-identical — blocking the
OUTPUT rows leaves each row's accumulation order untouched.

The two optimizations composed exactly as predicted: main's parallelism splits across row
groups, the register blocking works within one. No conflict existed between them; they were
simply written at the same time and the merge had to pick one file.

THE REAL FINDING is why the loss went unnoticed. The gguf benchmark suite covered Q8_0 and
Q4_K only — and Q4_K is the one K-quant type that does NOT use these kernels, because its row
dot is SIMD-gated and deliberately kept scalar. So four kernels became unreachable and every
benchmark stayed green. Panic-probing the existing benchmarks showed the blocked path was not
entered; adding Q6_K and Q2_K benchmarks and re-probing is what confirmed the fix.

A test suite that passes over dead code is not a safety net for merges. The benchmarks added
here are the durable part of this task, more than the reinstated call sites.

Q4_K remains excluded, stated at the site rather than left implicit: it needs a flag reporting
whether the SIMD override took effect before it can choose between the asm kernel and 4-row
blocking (T-01KYQ65MEEEN6, still open).
