---
schema: v1
---

## T-01KYJREGCMF3Q8H8D9SNSNT9ZE Fix reduceKernel — closure call, odometer and F32 pre-widening on every element
kind: task
state: draft
created: 2026-07-27
targets: backend/ref/reduce.go, docs/benchmarking.md

PROGRESS: sub-steps (a) and (b) are LANDED and measured. Sub-steps (c) and (d) are NOT DONE and are why this item stays open.

LANDED (a), reduce-all fast path. Every element lands in acc[0], so the output offset is invariably 0 and the odometer computes of += 0 once per element; a local accumulator also removes the acc[0] load/store and its bounds check. MEASURED 226,448 -> 132,089 ns on BenchmarkSumF64_64K, 1.71x.

LANDED (b), axis case run-length strip-mining. The innermost axis has a constant effective stride across its run, so the run is either one accumulator fed repeatedly (stride 0, reduced innermost axis) or a straight walk down consecutive accumulators (stride 1, axis survives into the output). Hoisting it ticks the odometer once per run. MEASURED 226,176 -> 129,071 ns on BenchmarkSumAxisF64_256x256, 1.75x.

Each change used the other benchmark as its CONTROL and neither moved the other path: when (b) landed, reduce-all held at 131,981 vs 132,259 ns. Both A/Bs were interleaved three-round file-copy toggles on the same host in one session, medians reported. Together the two put both reduce paths at roughly 1.7x off original.

BIT-IDENTITY: exact, verified rather than argued, because ref is the numeric truth every accelerated kernel is validated against. Raw output bits compared byte-for-byte before and after: 5 ops x 3 axis configurations x 2 dtypes (7,760 bytes) for (a), and a wider matrix for (b) — 5 ops x 4 axis configurations x 3 shapes including a rank-3 and a non-power-of-two x 2 dtypes, 17,040 bytes. Every accumulator sees the same values in the same ascending order, so its combine chain is unchanged; only index bookkeeping moved. Cross-backend parity suites in backend/, backend/ref/ and backend/cpu/ pass.

STILL TO DO:
(c) Devirtualize combine/finalize with the zero-size functor generic pattern. This is now the largest remaining term: the per-element indirect CALL is untouched by (a) and (b). GATE IT on -gcflags=-S showing no FMADDD in the reduce core — inlining a+x cannot fuse, but OpProds a*x adjacent to a store is the case to check, and a new contraction would silently change the reference. The per-op tolerance must NOT be widened to absorb a drift.
(d) F32 branch reading []float32 and widening per element instead of f64Data pre-widening the whole input. BenchmarkSumAxisF32_256x256 shows 529,697 B/op — a 512 KB garbage buffer per call — and +19 percent over the F64 path.

The originally predicted 3.5-5x covers all four sub-steps; the two landed account for about 1.7x, so (c) and (d) carry the rest. Do not read the shortfall as the prediction being wrong until they are attempted.

METHOD NOTE: capture the golden output bits before touching anything and diff afterwards. On this kernel it is cheap, and the argument that a restructuring preserves order is easy to make and easy to get wrong.

## T-01KYJREGVQEX696JPW87A887YA Make broadcastKernel copy contiguous runs instead of walking an odometer per element
kind: task
state: done
created: 2026-07-27
targets: backend/ref/broadcast.go, backend/ref/broadcast_runs_test.go, docs/benchmarking.md

LOWEST-RISK ITEM IN THE BACKEND SET and a clean 3x — take it first if you want a safe calibration of the harness.

SITE: backend/ref/broadcast.go:14 broadcastKernel; loops at :48-59 (F64) and :64-75 (F32); registered at :96-97.

WHY HOT: OpBroadcast is registered ONLY by ref — no cpu, no metal, nothing below it. It is the VJP of every reduction and of AddBias, so it runs once per broadcast-shaped gradient per training step, per element of the OUTPUT (the larger tensor). In autograd-heavy code that output is the full activation tensor.

MEASURED: BenchmarkBroadcastF64_256to256x256 159,650-161,650 ns = 2.44 ns/element, against BenchmarkReshapeF64_64K allocating and copying the identical 512 KB at 0.657 ns/element — a 3.7x gap on pure data movement.

DEFECT: the op is a verbatim same-dtype copy, yet every element pays the odometer at :50-58. For the benchmarked [256] -> [256,256] case eff = [0, 1]: the innermost axis advances ioff by exactly 1 for 255 of every 256 elements. That is a fully contiguous 2 KB run being copied one float64 at a time with an odometer and bounds checks around each store.

FIX: before the loop, compute the trailing contiguous run length. Innermost axes with eff[d] == 1 and matching extents give L = product -> emit copy(dst[pos:pos+L], src[ioff:ioff+L]). Innermost axes with eff[d] == 0 (the true broadcast axes) mean the source value is constant over the run -> fill dst[pos:pos+L], or use a doubling copy from the first written element. Tick the odometer once per run rather than once per element. Both branches are dtype-generic over refFloat so F32 and F64 share one core.

VALIDATION GATE (benchmark only): BenchmarkBroadcastF64_256to256x256 (backend/ref/perf_regress_test.go:197) covers the eff=[0,1] shape. ADD TWO CASES for the other regimes, since the run-length computation is where a bug would hide: [256,1] -> [256,256] (innermost eff==0, the replicate path) and [1,256,1] -> [8,256,16] (mixed). Land the run-length change ALONE, nothing else, and A/B against all three.

EXPECTED: 2.44 -> about 0.7 ns/element, i.e. roughly 3-3.5x (160 us -> 45-55 us); the replicate path should go further since it avoids re-reading src. High confidence — the target rate is measured in the same package on the same host.

BIT-IDENTITY BAR: LOWEST OF THE SET, though still a ref kernel. The op performs NO ARITHMETIC WHATSOEVER — the kernel's own comment at broadcast.go:30 calls it a verbatim copy. Replacing element-at-a-time stores with copy() over the same source and destination pairs cannot change any bit: no accumulation order, no widening, no rounding. The only failure mode is an INDEXING bug (wrong run length), which is a wrong-value error rather than a rounding error and is caught outright by the existing exact-equality broadcast tests. No tolerance change, and no cross-backend implication since no other backend implements the op.

PERFSCAN RULE REQUIRED: scalar copy loop with a contiguous inner run. AST shape: a ForStmt whose body's only data statement is an assignment dst[i] = src[j] between two slices of the same element type, where j advances by a loop-invariant stride slice and the loop contains no arithmetic on the copied value. Detector: AssignStmt with IndexExpr on both sides, both base idents having identical types.Slice element types, no BinaryExpr on the RHS value.

## T-01KYJREH8QF2HTHB9KC6S8NBJ0 Remove the per-element indirect call and bounds checks from the elementwise kernels
kind: task
state: draft
created: 2026-07-27

SITE: backend/ref/elementwise.go:17 unaryKernel (loops :35-38, :42-45) and :57 binaryKernel (loops :78-81, :86-89).

WHY HOT: per element, per elementwise op. cpu covers OpAdd/Sub/Mul/Div and the unaries, so on a full build ref is not the production path for those — BUT it is the production path for OpMaximum and OpMinimum, for every broadcasting binary (elementwise.go:96-112), for F64 where cpu's coverage is thinner, and for every CGO_ENABLED=0 or ref-only build. It is also the kernel every accelerated elementwise kernel is cross-validated against, so it runs once per validation case in the parity suites.

DEFECT: -gcflags=-S at elementwise.go:81 shows FMOVD, FMOVD, then CALL (R0) — an indirect call per element to compute one a+b. -d=ssa/check_bce/debug=1 reports IsInBounds at 37:7, 37:17, 44:7, 44:33, 81:8, 81:19, 81:26, 89:8, 89:35, 89:51 — THREE bounds checks per element in the binary loops, because the trip count is n := a.Numel() (:71), which the compiler cannot relate to len(as)/len(bs)/len(os). Measured 2.11-2.40 ns/element on BenchmarkAddF64_4K for a working set that fits in L2, against roughly 0.3-0.5 ns/element for a memory-bound f64 add.

FIX, two steps, land and A/B separately: (1) re-slice to the trip count once before the loop (as = as[:n]; bs = bs[:n]; os = os[:n]) and range over os — this alone should clear all three IsInBounds; re-run the BCE dump to confirm zero hits at :81 and :89. (2) devirtualize with the zero-size functor generic pattern — one generic core elemBinary[T refFloat, O binOp](as, bs, os []T) instantiated per operation, so -gcflags=-m shows the operator inlined and -S shows no CALL in the loop body. refFloat already exists at backend/ref/devirt.go:18.

VALIDATION GATE (benchmark only): BenchmarkAddF64_4K and BenchmarkAddF32_4K (backend/ref/bench_test.go:28,34) cover the binary path. THE UNARY PATH HAS NO BENCHMARK — add BenchmarkTanhF64_64K and BenchmarkReLUF64_64K. That pair is deliberate: ReLU is where call overhead most dwarfs the work, Tanh is where it does not, so together they BOUND the win rather than flattering it.

EXPECTED: bounds-check removal alone about 15-25%; devirtualization takes Add from 2.11-2.40 to roughly 0.5-0.8 ns/element, about 3x on BenchmarkAddF64_4K. On Tanh and GELU the win is a few percent because the transcendental dominates — EXPECT AND ACCEPT THAT rather than reading it as failure. High confidence on the mechanism, medium on the multiplier for the unary family.

BIT-IDENTITY BAR, HIGH TIER — ref is the numeric truth and every accelerated elementwise kernel is validated against these functions, so a drift here moves the goalposts for cpu, metal, cuda and vulkan simultaneously. Re-slicing is provably neutral: it removes a panic path and nothing else. Devirtualization is neutral PROVIDED THE INLINED BODY IS NOT FUSED — the F32 loops at :86-89 are os[i] = float32(op(float64(as[i]), float64(bs[i]))), a widen/compute/narrow chain that must survive verbatim, and while inlining a*b adjacent to a store is safe, any combine of the form a*b + c could contract to FMADDD on arm64 and silently change the reference. GATE the change on -gcflags=-S showing no FMADD in the generated elementwise cores, plus exact-equality comparison against the pre-change output on backend/ref/testdata. The documented per-op tolerance must not be widened to absorb a drift.

PERFSCAN RULE REQUIRED, in addition to the widened PS1002 covered by the reduce task: trip count divorced from slice length. AST shape: a loop whose bound is an int variable derived from a METHOD CALL (.Numel(), .Len(), .Size()) while the body indexes slices whose len() is never compared to that bound in the enclosing block. Detector: the bound ident's definition is a CallExpr with a non-slice receiver, and the body contains at least one IndexExpr on a slice-typed ident not re-sliced by that bound. Instances: ref/elementwise.go:36,43,79,87 and ref/reduce.go:110.
