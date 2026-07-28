---
schema: v1
---

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
