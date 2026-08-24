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
