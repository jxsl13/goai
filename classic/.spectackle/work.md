---
schema: v1
---

## T-01KYJPYC1JEKQT3X1F1A7A26VY Carry values and labels in lockstep with the presorted columns in the split sweep
kind: task
state: draft
created: 2026-07-27

Measured baseline on this host: BenchmarkTreeFit 11.80ms (4000x20, 3-class, unlimited depth), BenchmarkForestFit 184.3ms, BenchmarkSVCFit/n4000_rbf 7.25ms.

SITES: classic/tree.go:552 (*cartBuilder).sweep — the value gather at :558-561, regression target gathers at :566-568 and :571, classification label gathers at :595-597, :617, :641. Threshold re-read at tree.go:537-538 in bestSplit. Mirror defect in the GBM exact grower at classic/gbm.go:329-331 and :334.

WHY HOT: Fit -> fitWithSeed -> build -> bestSplit -> sweep, once PER FEATURE PER NODE. Per node of m samples the sweep does d*m value gathers plus d*m (classification) or 2*d*m (regression) target gathers; over a tree that is d*n*depth — for the benchmark shape 20 x 4000 x about 15 levels, roughly 1.2M value gathers plus as many label gathers. This is the dominant memory traffic of a single-tree fit and of every GBM weak learner, since GBM's DEFAULT grower is the exact gbmBuilder, not the opt-in histogram one.

DEFECT: order[k] walks samples in sorted-by-feature order, a pseudo-random permutation of row order. b.x is a slice-of-slices, so each element costs TWO DEPENDENT random loads — the 24-byte slice header at b.x[id], then row[f] inside a separately allocated 160-byte row. At n=4000, d=20 the row data is about 640KB, over L1, so every gather is an L2 round trip with no prefetch possible and the second load cannot issue until the first retires. The same permutation is then re-walked to gather the labels. Comments at tree.go:555-557 show a previous pass halved the COUNT of these gathers; the gathers themselves were never removed. This is precisely where sklearn's C splitter has its structural edge: it holds X as one contiguous Fortran-ordered buffer, so its Xf fill is a strided-but-linear read rather than a pointer chase — and decision tree is one of the two estimators this library currently LOSES on (about 1.3x).

FIX: carry the sorted PAYLOAD alongside the sorted INDEX. In initColumns (tree.go:282) allocate from single backing arrays colVal [][]float64 (d x n) and, classification only, colLab [][]int32 (regression carries colY instead), filled at presort time so colVal[f][p] == b.x[cols[f][p]][f]. In partition (tree.go:324-350) move the value/label element in lockstep with cf[w]=s — one extra sequential read and write per element per feature, introducing NO new gather because the payload travels with the index already being moved. sweep then takes vals := b.colVal[f][start:end] directly and drops the fill loop entirely; class/target reads become labs[p-1]. bestSplit's threshold reads become vals[cut-1], vals[cut]. Apply the same to gbmBuilder (gbm.go:238/:266/:314/:358); GBM's fit filter loop at :279-284 already streams the master column so carrying a value twin there is free. Memory cost about +960KB at the benchmark shape — guard it behind an n*d budget with a fallback to today's gather path, which is result-identical either way.

VALIDATION GATE (benchmark only): BenchmarkTreeFit (classic/treefit_bench_test.go:41) and BenchmarkGBMFit (:64) cover this today. Add BenchmarkTreeFitShapes sweeping {2000,10,3}, {4000,20,3}, {8000,50,3}, {4000,20,8} — THE WIN MUST GROW WITH n*d as the row data leaves L1/L2, and that gradient is the proof the fix is the gather rather than something else. Add the DecisionTreeRegressor variant (it carries 2x the target gathers so it should improve most) and a WithCriterion(Entropy) case.

EXPECTED: 1.4-1.9x on BenchmarkTreeFit (11.8ms -> about 6.5-8.5ms), larger on the regressor and at wider d. High confidence that it is a real win, medium on the exact factor since L2 hit latency is partly hidden by the sweep's arithmetic.

BIT-IDENTITY BAR: bit-identical — a pure data-layout change. Every arithmetic expression, its operand order, and the candidate-split acceptance tests are untouched, so thresholds and chosen splits are unchanged. Pin the invariant colVal[f][p] == x[cols[f][p]][f] across partition with a debug assertion. Does NOT touch any predict-family width guard (tree.go:861-863, :926-928) — those live in Predict, which never sees the builder. Existing goldens classic/testdata/trees.json and gbm.json are the regression net.

PERFSCAN RULE REQUIRED: indexed-payload gather in a hot permutation walk. AST shape: a for loop containing an index expression A[B[i]][c] or A[B[i]] where B is a []int that is itself the loop's iteration source or a slice of one, A is a struct-level [][]T or []T not written in the loop, and the enclosing function is called from a loop nest at least 2 deep. Report when the gathered result is written into a dense scratch slice (scratch[k] = A[B[k]][c]) — that write is the tell that the value is being materialized in permuted order, which means it could have been maintained in that order instead. Highest-confidence sub-case is A being [][]T (double indirection). Suppress when B is provably the identity or ascending.

## T-01KYJQ78WJF57RA3W2CYJSE2DP Flatten the SVC kernel matrix and specialize the kernel column per kernel type
kind: task
state: draft
created: 2026-07-27

RBF-SVC is one of the two estimators this library currently LOSES on against scikit-learn (about 2.0x). Measured here: BenchmarkSVCFit/n4000_rbf 7.247ms, /n1000_rbf 1.543ms — 4x the samples for only 4.7x the time, so SMO STEP COUNT IS NEARLY FLAT IN n and the per-step O(n) work is the whole cost.

SITE: classic/svm.go:326 (*kernelCache).column, inner loop :332-334; callee (*SVC).kernel at :169-185 (RBF branch :173-179). Same pattern at :307-309 (diagonal precompute) and :543-549 (*SVC).decision on the predict side.

WHY HOT: Fit -> smo -> kc.column(i) at :402 and kc.column(j) at :469, up to twice per SMO step, each computing n kernel evaluations — order 1e5 to 1e6 evaluations for the n=4000 fit. Confirmed by go build -gcflags=-m: neither kernel nor sweep is inlinable.

DEFECT, three compounding: (1) PER-ELEMENT DISPATCH — kc.m.kernel(xi, kc.x[t]) is a method call INSIDE the t loop; kernel is not inlinable (it contains loops), so every one of the n elements pays a call plus a 4-way switch on a LOOP-INVARIANT value plus a reload of m.gammaVal through the pointer. (2) SLICE-OF-SLICES ROW CHASE — kc.x[t] is a slice header load followed by a load of a separately allocated row, striding over the whole 640KB training matrix once per column with no prefetch. (3) SERIAL FP DEPENDENCY CHAIN — s += d*d at :177 is a single accumulator: d=20 dependent FADDs at about 3-cycle latency, roughly 60 cycles of pure latency per evaluation against about 20 cycles of issue work.

FIX, staged; the first two stages are BIT-EXACT and must land before the third is even judged.
STAGE A (bit-exact): give kernelCache a flat contiguous copy of X built once in Fit — xf []float64 with row t at xf[t*d:(t+1)*d]. Specialize column per kernel type: hoist the switch ABOVE the t loop into columnRBF / columnLinear / columnPoly, each with the kernel body inlined and gamma, d, coef0 hoisted into locals. Accumulation order unchanged, so every kernel value is bit-identical; only dispatch and one level of indirection disappear. Apply the same to decision (:543-549) — store support vectors flat (m.sv is currently [][]float64 built by row-copy at :276) and hoist the kernel switch out of the SV loop; that is a straight Predict/DecisionFunction win with no fit-time behavior change.
STAGE B (bit-exact): block the column computation, processing 4 rows per iteration against a single xi held in registers, so xi's d values load once per block instead of once per row and four independent accumulator chains interleave WITHOUT changing any single chain's order.
STAGE C (NOT bit-exact — DO NOT BUNDLE): the libsvm/sklearn gram identity ||xi-xt||^2 = sq[i] + sq[t] - 2*xi.xt with sq precomputed once. This turns the column into a gemv and halves the arithmetic — it is what sklearn actually does — but it changes rounding in the last ulps, which perturbs the SMO trajectory and therefore the fitted alpha, b and support-vector set. Land A+B first, measure, then judge C on its own evidence.

VALIDATION GATE (benchmark only): BenchmarkSVCFit (classic/svm_bench_test.go:37) is the right harness. Extend svmBenchData's sweep to d in {5, 20, 80} at fixed n=4000 — STAGE A/B GAINS MUST SCALE WITH d, which is what distinguishes them from any step-count change. Add a predict-side benchmark, which no current file covers: BenchmarkSVCDecisionFunction fitting once at n=4000/d=20 RBF then timing DecisionFunction over 1000 rows.

EXPECTED: Stage A alone 1.15-1.3x; A+B 1.3-1.6x (7.25ms -> about 4.5-5.5ms), which would close most of the 2.0x gap when combined with the SMO membership work. Medium-high confidence for A+B; the irreducible math.Exp per element sets the floor.

BIT-IDENTITY BAR: Stages A and B change NOTHING about fitted-model output — same operations, same order, same values. Stage C WILL change fitted models in the last ulps of alpha/b and possibly by one support vector; the golden test TestSVMGoldenParity (classic/svm_test.go:107) allows 2e-3 on the decision function and +/-1 on SV count so it would likely still pass, but that makes it a BEHAVIOR CHANGE that must be declared as such, not an optimization. Neither stage touches the width guard, which lives solely in DecisionFunction (:559-562) with Predict deliberately routed through it (:572-577) — the flat-SV change must keep m.nFeat as the compared value and must NOT start deriving the width from the flat buffer's stride.

NOTE ON PS4002: math.Exp at :179 is a scalar transcendental in a hot loop and a hand-vectorized sibling does exist (backend/cpu/vexp_arm64.s), but PS4002 will not fire — it is gated on the file already calling a SIMD kernel, and classic/svm.go calls none. Reporting the instance rather than proposing a rule change: the vexp path is f32-only and unexported so it is not directly reusable for []float64, and the exp count is irreducible, which is exactly why Stages A/B target everything AROUND the exp.

## R-01KYN1JXW6EPFBR4PWD04HNE6P Scaling sweep found 4 serial spines; GBM histogram fixed at 1.57x, three still open with numbers
kind: research
state: draft
created: 2026-07-28

A new diagnostic and its first harvest. Running each benchmark at GOMAXPROCS=1 and at full width and dividing turns "is this parallel?" from a code-reading question into a measurement. Shipped as internal/perfscan/tools/scaling_sweep.sh.

FIRST SWEEP, on this host (M2 Pro, 12 P):
  BenchmarkGBMHist_hist_80k   334ms  1.01x   FIXED, now 1.57x
  BenchmarkGMMFitFull          77ms  1.00x   OPEN
  BenchmarkMLAVJPSeq256        20ms  0.99x   OPEN
  BenchmarkCholeskyVJP_128    4.7ms  0.88x   OPEN, and SLOWER with more cores
  BenchmarkMamba2Prefill_512   57ms  2.84x   already parallel, no action
  BenchmarkOLSFit_512x64      1.2ms  2.12x   already parallel, no action

GBM: two per-feature loops, binning in newHistBuilder (1.26x) and accumulation in
buildHist (1.29x on top), combined 337.0 -> 213.9ms interleaved. Scales 329.8 / 244.3 /
211.7ms at 1 / 4 / 12 Ps.

THE TRANSFERABLE TRAP, and the reason this is worth a record rather than a commit message:
the FASTER SERIAL LOOP ORDER WAS THE UNPARTITIONABLE ONE. buildHist was sample-major,
which reads the bin table contiguously; feature-major is what gives each feature exclusive
ownership of its bins and makes splitting safe, and it costs 22% on a single core (336 ->
410ms). The first version shipped that regression. Both orders are now kept and chosen by
whether the work will actually be split. Expect this shape wherever a reduction is
parallelized: the axis that makes writes disjoint is rarely the axis with the best
locality.

GATE: TestGBMHistogramDeterministic could not serve — it fits twice with the SAME code, so
it proves reproducibility, not preservation, and a deterministic-but-wrong change passes
it. A frozen bit-level golden was added and mutation-probed: a true one-ulp bump of the
histogram sums and a one-ulp shift of a binning boundary each turn it red while every
pre-existing histogram test stays green. One probe was rejected as VACUOUS rather than
counted — multiplying by (1 + 1e-16) is a no-op because that rounds to exactly 1.0 in
float64.

NOT GENERALIZED into a perfscan rule, deliberately and for the third time in this
campaign: proving loop-iteration independence is dataflow, perfscan is AST-only, and a
rule that guessed would advise races. The fusion and register-blocking wins DID become
rules (PS6003, PS6005) because they are structural. Parallelization gets a measurement
tool instead.

CholeskyVJP_128 at 0.88x deserves its own look: something there is not merely serial but
actively penalized by more cores, which usually means false sharing or a barrier in a hot
loop, and that is a different defect from an unparallelized one.

## R-01KYN2CWFFFA5T1EGCHVEYPK56 GMM E-step 1.93x; the shared receiver scratch was both the race and the reason the first attempt only measured 1.16x
kind: research
state: draft
created: 2026-07-28

Second serial spine from the scaling sweep (R-01KYN1JXW6EPF). BenchmarkGMMFitFull ran 77ms at one core and 77 at twelve. The E-step is 49% of the profile and every sample's responsibilities are independent.

MEASURED INTERLEAVED, 4 alternations: 75.4-77.5ms -> 39.3-40.5ms, 1.93x. Scales 77.9 / 43.2 / 39.4ms at 1 / 4 / 12 Ps.

THREE THINGS WENT WRONG IN ORDER, and each is the useful part of this record.

1. AN EXACT NULL FROM A THRESHOLD, not from the transform. The first measurement was 1.00x because parallelSamples was given k as the per-sample cost. A full-covariance Mahalanobis is a triangular solve, O(d^2) per component, so the estimate was low by a factor of d and a 2000x24 fit took the serial path. A parallelization measuring as a perfect null is more often a threshold that never fired than a transform that does not pay — check the guard before concluding anything about the code.

2. A DATA RACE THE CODE HAD ALREADY DOCUMENTED. logGaussian's triangular-solve buffer was a receiver field carrying the comment "logGaussian runs serially". The precondition was known and written down, and parallelizing violated it silently. -race caught it. A comment is not a guard: the requirement now lives in the signature as a parameter, and the field is deleted rather than left available.

3. THE RACE WAS ALSO A PERFORMANCE BUG. The racy version measured 1.16x; fixing the sharing took the identical parallelization to 1.93x. Twelve cores were writing one cache line. A shared scratch under contention costs more than the allocation it saves — which inverts the usual reason such buffers are hoisted onto a receiver in the first place.

GATE: TestGMMDeterminism could not serve — it fits twice with the SAME code, so it proves reproducibility, not preservation, and EM is iterative enough that a one-ulp shift can land on a different iterate. A frozen bit-level golden was added and probed: red under a one-ulp mStep mean bump, a reversed mStep sample order, and a one-ulp responsibility bump, while every pre-existing GMM test stays green.

HONEST LIMIT OF THAT GATE, recorded so it is not over-trusted: it cannot see the log-likelihood total, because that only drives a convergence check against tol=1e-3, some thirteen orders of magnitude coarser than an ulp. The total is preserved exactly anyway (per-sample contributions summed in sample order rather than accumulated per chunk) because it costs one n-word array and removes the need to reason about how unlikely a flipped stopping decision is.

GENERALIZED as perfscan PS6006 and cast as CONC-SCRATCH-FIELD-001. NOT DONE: mStep is still serial, but it parallelizes only k ways (6 here), so it is a smaller and shallower win than the E-step was.

## R-01KYN4ARFEEA3VKX8RVRMV649P GMM fit 4.09x complete (76.5 -> 18.7ms); mStep was three quarters of what the E-step fix left
kind: research
state: draft
created: 2026-07-28

Closes the GMM line from the scaling sweep. Two changes, measured separately, each interleaved with min of 3 runs per arm.

  baseline            76.5ms
  + E-step parallel   39.4ms   1.93x  (over samples, plus the logGaussian race fix)
  + M-step parallel   18.7ms   2.03x  (over components)
  cumulative                   4.09x
Scales 76.1 / 26.4 / 18.7ms at 1 / 4 / 12 Ps.

THE SECOND NUMBER IS THE INFORMATIVE ONE. The M-step splits only k ways and k is 6 here, yet it returned 2.03x — which measures how completely the full-covariance accumulation dominated what the E-step fix left behind. A 6-way-parallel section returning 2x was roughly three quarters of the remaining work. Attribution shifted that much because fixing the larger half re-weighted everything; the profile taken BEFORE the E-step change put mStep at 36.5%, and acting on that number alone would have understated this by half.

SAME BLOCKER BOTH TIMES, and the second was avoided rather than discovered: a scratch buffer shared across the units being parallelized. In the E-step it was logGaussian's triangular-solve buffer, a receiver field carrying a comment that said the method runs serially; -race caught it and it cost a 1.16x measurement before the sharing was fixed and the real 1.93x appeared. In the M-step it was the centered-row scratch, moved per-chunk by construction. PS6006 was written from the first; the second is the check being applied.

ERROR PATHS both needed a captured first-error under a mutex, since a chunk body on a pool worker has nowhere to return one to. Worth expecting whenever a loop with an error return is parallelized — it is not incidental.

GATE: a frozen bit-level golden of the full-covariance scores, mutation-probed red under a one-ulp mStep mean bump, a reversed mStep sample order and a one-ulp responsibility bump, while every pre-existing GMM test stays green. TestGMMDeterminism could not have served — it fits twice with the SAME code and so proves reproducibility, not preservation.

NOT DONE: the diagonal-covariance mStep path is still serial. It is the cheap covariance mode (O(n*k*d) rather than O(n*k*d^2)) and no benchmark in the suite exercises it at scale, so there is nothing to validate against — measure first if it is ever wanted.

## R-01KYN5DVW6F849JR3T85G3FVXG Exact GBM split search 1.72x (saves 780ms/fit); the prediction golden could not see the tie-break, so it needed its own test
kind: research
state: draft
created: 2026-07-28

BenchmarkGBMHist_exact_80k, the presort grower, ran 1.86 SECONDS fully serial — the largest absolute saving found in this campaign. gbmBuilder.bestSplit is 59% flat, partition another 28%.

MEASURED INTERLEAVED, 3 alternations, min of 3 per arm: 1836-1883ms -> 1080-1092ms, 1.72x. Scales 1821 / 1110 / 1089ms at 1 / 4 / 12 Ps, flattening after 4 because partition stays serial. Amdahl caps this at about 2.2x, so 1.72x is most of what is reachable without touching that loop.

A SECOND FALSE "SLOWER WITH MORE CORES": the sweep read 0.93x, and re-measuring at every core count gave 1880-1933ms flat — a 3% noise band. Same shape as the CholeskyVJP 0.88x that reached three records before correction. The re-verification rule caught this one before it became a claim. A sub-1.0 ratio should be treated as unmeasured until re-run per core count.

TWO BLOCKERS, both recurring shapes: b.vals was a shared receiver scratch overwritten per feature (the PS6006 shape — and its name heuristic does NOT catch "vals", a real limitation of that rule found on the very next case), and the best-gain tracking is an argmax reduction, now per-feature candidates combined in ascending order with strict >.

THE GATE GAP IS THE PART WORTH CARRYING. TestGBMHistogramBitIdenticalToGolden constructs with WithGBMHistogram and never enters this function — the exact and histogram growers are different code, and parallelizing one under the other's gate would have been unguarded. A separate frozen golden was captured.

AND THAT GOLDEN IS WEAK IN A SPECIFIC WAY, established by probing: red when a feature is dropped from the search, but GREEN under a one-ulp left-sum bump, under >= instead of > in the combine, and under a descending combine order. The reason is structural, not fixable by a better fixture: tree growth is decided by COMPARISONS, so ulp-level noise does not move a split, and random data never produces two bit-equal gains, so no prediction fixture can exercise a tie at all.

So the tie-break needed a constructed fixture: two features inducing the IDENTICAL partition have bit-equal gains while their thresholds differ, which makes the choice observable. That test is red under both mutations the goldens miss. Cast as NUM-ARGMAX-TIEBREAK-001.

GENERAL LESSON: a prediction-level golden gates VALUES, not DECISIONS. Where an algorithm's output is chosen by comparison rather than computed by arithmetic, exactness gates go quiet exactly where the parallelization is riskiest, and the tie must be constructed deliberately.

NOT DONE: partition (28% flat) is still serial. It permutes every feature column in place for the chosen split, so it is a scatter rather than an independent-row loop and needs its own analysis.
