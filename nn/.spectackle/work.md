---
schema: v1
---

## T-01KYJR5WRXF5CSDZC316FB10T5 Replace Muon's private naive GEMM — it runs at 3.3 GFLOP/s beside the repo's own 61 GFLOP/s kernel
kind: task
state: draft
created: 2026-07-27

LARGEST SINGLE GAP FOUND IN THE SWEEP: a 19x ratio against a tuned kernel that already exists in this repository.

MEASURED on this host (M2 Pro, darwin/arm64, go1.26): BenchmarkMuonStepOnly 409,706,044 ns/op, 26,738,729 B/op. pprof over that benchmark: nn.matmulABt 4.61s (59.95%), nn.matmulFlat inlined 2.72s (35.37%), nn.newtonSchulz5 7.52s cumulative (97.79%). One Step is 1.343 GFLOP in 409.7 ms = 3.28 GFLOP/s. backend/cpu.gemmF64Band measures 61.4 GFLOP/s at f64-512 (BenchmarkGemmDirF64_512).

SITE: nn/muon.go:190 matmulABt (inner dot at :198-199) and nn/muon.go:169 matmulFlat (inner axpy at :182), both called from newtonSchulz5 (:151, :152, :157). Reached via (*Muon).Step (:91) -> newtonSchulz5 (:116): 3 matmuls x NSSteps(5) x every 2-D parameter, every step.

DEFECT, four parts, all confirmed in emitted code:
(a) SERIAL FP-ACCUMULATOR CHAIN. -gcflags=-S at muon.go:199 emits FMOVD then FMADDD F1, F0, F2, F0 — F0 is both operand and destination, so each FMADD waits on the previous one's ~4-cycle latency. Measured 0.92 ns/MAC (about 3.2 cycles) against matmulFlat's independent-accumulator axpy at 0.32 ns/MAC — a 2.9x per-MAC gap even though matmulABt does LESS work (251M vs 419M MACs/step).
(b) UN-ELIMINATED BOUNDS CHECK. -gcflags=-d=ssa/check_bce/debug=1 reports nn/muon.go:199:20 Found IsInBounds, and the loop contains CMP/BHI plus a CALL to runtime.panicBounds. The comment at :198 claims "equal-length slice dot -> auto-vectorizes" and BOTH HALVES ARE FALSE: ai and bj are distinct slices so the compiler cannot prove the length relation, and gc does not auto-vectorize on arm64 — the emitted code is scalar FMADDD. muon.go:182:21 has the same un-eliminated check.
(c) SERIAL, while a tuned parallel kernel exists in-repo and nn already imports ops (see nn/pissa.go:7).
(d) bm (:153) and the matmul return buffers are re-allocated every NS iteration — 26.7 MB/op — although shapes are fixed per parameter and Muon already carries a buf field (:32).

FIX, in increasing payoff, LAND SEPARATELY: (1) hoist bm/A/A2/bx into per-parameter scratch on the Muon struct beside buf; (2) reslice for bounds-check elimination — bj = bj[:len(ai)] at :196, bp = bp[:len(ci)] at :180; (3) rewrite matmulABt in matmulFlat's ikj/axpy form (transpose the k-dim operand once — 131k elements, negligible against 33.5M MACs) AND exploit that matmulABt(X, X, r, cc) at :151 passes THE SAME SLICE TWICE, so C = X*X^T is symmetric: compute j <= i and mirror, a further 2x; (4) best, route all three products through ops.MatMul / backend.OpMatMul to get the parallel gemmF64Band.

VALIDATION GATE (benchmark only): BenchmarkMuonStepOnly (nn/train_bench_test.go:159) isolates this exactly — run the A/B at -benchtime 50x. Add BenchmarkNewtonSchulz5_256x512 calling newtonSchulz5(x, 256, 512, 5) on an internal test file to separate steps 2-3 from step 4.

EXPECTED: steps 1-3 alone 409.7 ms -> about 200 ms (2.0x), high confidence since the arithmetic follows directly from the two measured per-MAC rates. With step 4: -> about 40 ms (10x), medium-high confidence (the GEMM shapes are smaller than the 512-cubed benchmark and there are roughly 30 fork/joins per step).

BIT-IDENTITY BAR: steps 1 and 2 bit-identical trivially. Step 3 IS BIT-IDENTICAL — for fixed (i,j) the ikj form accumulates over p in the same ascending order with the same FMADDD fusion, which is precisely the argument backend/cpu/gemm.go:9-14 makes for the tolerance-0 cross-reference gate; the symmetry mirror is exact because IEEE multiplication is commutative. Step 4 is EXPECTED bit-identical for the same reason but MUST be validated against the existing tolerance-0 tests before merge rather than assumed. No RNG anywhere.

PERFSCAN RULE REQUIRED: latency-bound scalar dot-product matmul. AST shape: a 3-level loop nest whose innermost body is a single AssignStmt acc += A[..] * B[..] (token.ADD_ASSIGN) where acc is a function-local float32/float64 declared in the enclosing loop, both operands are IndexExpr over DISTINCT slice idents, and the accumulated variable is stored to an IndexExpr after the inner loop. Use the bounds-check-elimination oracle as a secondary signal. High precision — this shape is always either an ikj rewrite or a call to the backend GEMM.
