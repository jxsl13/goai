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

## T-01KYJR5XB6FB2SGHXSSER77MAG Drop the interface-dispatched RNG from Dropout's per-element mask loop
kind: task
state: draft
created: 2026-07-27

THE DEFECT WAS ESTABLISHED BY A CONTROL EXPERIMENT, not by inference — read that part before changing anything, because the obvious suspects are verified innocent.

MEASURED on this host: BenchmarkDropoutForward 10,432,895 ns/op, 12,583,603 B/op, 14 allocs. BenchmarkDropPathForward at the IDENTICAL shape 813,237 ns/op, 12,583,614 B/op, 15 allocs.

THE CONTROL: DropPath (nn/droppath.go:71-76) uses the identical tensor.New mask allocation, the identical full mask write, and the identical backend.Execute(OpMul) — differing ONLY in that it draws 16 random numbers instead of 1,572,864. So allocation, memclr, mask stores and the multiply together account for 813 us, and the remaining 9.62 ms (92%) is 1.57M RNG draws at 6.1 ns each — about 21 cycles for a PCG step that should cost about 2. The tensor allocation and mask materialization, which look like the obvious suspects, are VERIFIED NOT TO BE THE PROBLEM.

SITE: nn/dropout.go:72 and :79 — d.rng.Float64() inside the mask loop; d.rng is declared *rand.Rand at nn/dropout.go:33, built at :42.

WHY HOT: (*Dropout).Forward runs per dropout layer, per forward pass, per step — 2-3 times per transformer block. At the benchmark's 16x128x768 activation that is 10.4 ms per call, so a 12-layer block stack pays roughly 250 ms/step.

MECHANISM, confirmed by -gcflags='-m -m': Rand.Float64 and Rand.Uint64 both inline, but Rand.src is a rand.Source INTERFACE FIELD, so src.Uint64() is an indirect call that is neither inlined nor devirtualized. pprof corroborates: (*PCG).Uint64 0.80s flat, (*PCG).next 0.25s, and 1.74s of call-site stall attributed to the loop line — against internal/simd.MulF32 at 0.03s (0.9%).

FIX: store the concrete *rand.PCG (or an inlined 2-word PCG state) on the Dropout struct instead of *rand.Rand, so next() inlines. Then derive the Bernoulli from the raw draw with a precomputed integer threshold: thr := uint64(math.Ceil(d.Rate * (1<<53))), test u<<11>>11 >= thr. This is EXACTLY equivalent to float64(u<<11>>11)/(1<<53) >= d.Rate because u53/2^53 is exact in binary64 — the same comparison result for every draw. Optionally make the store branchless.

VALIDATION GATE (benchmark only): BenchmarkDropoutForward (nn/dropout_fastpath_test.go:143) isolates it. KEEP BenchmarkDropPathForward as the invariant control — it must not move; if it does, the change touched something it should not have.

EXPECTED: 10.43 ms -> about 2.5-3.5 ms (3-4x). High confidence — the control experiment bounds the non-RNG floor at 813 us and the remaining cost is a single known indirect call.

BIT-IDENTITY BAR: BIT-IDENTICAL AND RNG-SAFE. The draw count is unchanged (Rand.Float64 consumes exactly one Uint64), the PCG stream is unchanged, and the integer threshold reproduces the float comparison exactly. The mask, and therefore the output, is bit-for-bit the same. Because this class of change is RNG-adjacent, the existing seeded-determinism tests must be run and named explicitly in the commit rather than assumed to pass.

PERFSCAN RULE REQUIRED, and it has wide reach here: interface-sourced RNG in a per-element loop. AST shape: a SelectorExpr call X.Float64() / .Uint64() / .IntN() / .NormFloat64() where X resolves to *math/rand.Rand or *math/rand/v2.Rand, inside a loop whose bound is a slice length or Numel(). Recommend the concrete source type. THIS IS NOT CONFINED TO DROPOUT: 12 non-test files in nn hold a *rand.Rand field with 32 per-element draw sites — neftune.go, mixup.go, cutmix.go, rso.go, droppath.go, psgd.go, apollo.go, qgalore.go, aqlm.go among them. Run the finished detector and report every site.

## T-01KYJR5XXZFJ1AFGN2HR674XNT Flatten the SOAP/Shampoo basis rotation and eliminate its bounds checks
kind: task
state: draft
created: 2026-07-27

SITE: nn/soap.go:229 rotateForward (inner loops :236, :246) and nn/soap.go:255 rotateBack (:261, :271); scratch from zeroMat at nn/soap.go:198.

MEASURED: BenchmarkSOAPStepOnly 11,452,655 ns/op with 758,290 B/op and 976 allocs/op; BenchmarkShampooStepOnly 9,163,610 ns/op, 392 allocs. Profile share: rotateForward 9.05%, rotateBack 5.60% — about 34% of (*SOAP).Step's own cumulative time. Called per matrix parameter, per step.

DEFECT, four parts: serial FP-accumulator dot loops (acc += ...), latency-bound exactly as in the Muon case; THREE UN-ELIMINATED BOUNDS CHECKS PER ELEMENT, confirmed at soap.go:236:17, :236:24, :236:27, :246:13, :246:16, :246:27 all reporting Found IsInBounds; jagged [][]float64 storage, so every element access is a pointer load then an indexed load; and rotateForward:236 reads ql[i][k] with i INNERMOST, striding down a column across separately allocated rows — one cache line touched per element. Separately, zeroMat allocates 1+r slices per call and there are 4 calls per rotate pair per parameter per step, which is where the 976 allocs/op come from.

FIX: change the SOAP/Shampoo state matrices to flat []float64 with an explicit stride (they are already dense and rectangular), reorder both products to ikj so the inner loop is a contiguous axpy, and hoist the t/out scratch onto the optimizer struct. For rotateForward's first product Q_L^T*G the ikj form is: for i { for k { av := ql[i*m+k]; for j { t[k*n+j] += av * g[i*n+j] } } } — note this PRESERVES the i accumulation order per (k,j), which is what keeps it bit-identical.

VALIDATION GATE (benchmark only): BenchmarkSOAPStepOnly (nn/train_bench_test.go:189) and BenchmarkShampooStepOnly (:200) cover it end to end, with BenchmarkSOAPStepOnlyVec (:230) and BenchmarkShampooStepOnlyVec (:241) as CONTROLS THAT MUST NOT MOVE. For isolation add BenchmarkRotateForward64x128 on an internal test file, building ql[64][64], g[64][128], qr[128][128] in f64 and calling rotateForward in the loop.

EXPECTED: rotate pair 3-5x, taking BenchmarkSOAPStepOnly 11.45 ms -> about 8.5 ms (1.35x), and allocs 976 -> about 100. Medium-high confidence on the rotate speedup, medium on the end-to-end number since SymEig at 34% of the profile is the remaining floor.

BIT-IDENTITY BAR: bit-identical — same per-output ascending accumulation order, same FMA contraction; flat-versus-jagged storage changes no value. No RNG.

VERIFIED NOT A DEFECT, recorded so it is not re-investigated: SOAP does NOT recompute its eigenbasis every step. nn/soap.go:133 gates on s.t == 1 || s.t%s.Freq == 0 with Freq defaulting to 10 (:80). The SymEig cost visible at 34% of the profile is already amortized and is not a candidate.
