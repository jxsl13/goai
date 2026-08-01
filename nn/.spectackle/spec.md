---
schema: v1
prefix: MOE
---

## MOE-002
WHERE auxiliary-loss-free MoE routing is enabled, the router SHALL use the paper affinity exactly: s=sigmoid(routerLogit), TopK ranks s+b, and combine weights derive from s with b absent, renormalizing only after selection.

Rationale: The unoptioned Mixtral route stays softmax. The bias is control state outside Params and autograd and updates only through the explicit per-batch sign rule. Required gates include a decisive case where sigmoid(logit)+b and logit+b select different experts, plus a router finite-difference gradient check. Migrated from cavekit SPEC.md V32.

## LOSS-002
The EagleLoss feature regression SHALL compute mean SmoothL1 with beta=1 against the next feature, then add 0.1 times token cross-entropy, never substituting an MSE proxy.

Rationale: Required decisive gates: the abs(d)>1 linear branch, a full-head finite-difference gradient check, and lossless generation at an arbitrary head. Migrated from cavekit SPEC.md V33.

## OPT-003
WHERE Q-GaLore is selected, the optimizer SHALL apply all four deltas: block-wise INT8 moments, packed affine INT4 projection, adaptive SVD gap at 0.4/2/5, and INT8 weights re-quantized by unbiased stochastic rounding.

Rationale: QuantBits 0 or 64 disables all four and must stay bit-identical to GaLore across both projection sides and full rank. No end-to-end weight-memory claim is permitted while the public Tensor keeps a float execution mirror. Migrated from cavekit SPEC.md V34.

## OPT-004
WHERE APOLLO is selected, the optimizer SHALL persist only a two-word projection seed, low-rank moments and a limiter scalar, never retaining the O(r*min(m,n)) Gaussian projection matrix.

Rationale: P regenerates bit-identically from the seed on every use and changes only when the gap replaces that seed; reseeding preserves the moment buffers. Required gates cover both projection sides, seed regeneration equality, gap reseed difference, state shape, convergence and trajectory determinism. Migrated from cavekit SPEC.md V35.

## intent
- T-01KYJR5WRXF5CSDZC316FB10T5 Replace Muon's private naive GEMM — it runs at 3.3 GFLOP/s beside the repo's own 61 GFLOP/s kernel: LANDED steps 2+3 of the four-part fix, benchmark-validated on M2 Pro darwin/arm64 go1.26.5: matmulABt rewritten from dot-per-output to ikj/axpy with a one-shot k-dim transpose, plus the symmetry halving available because newtonSchulz5 calls it as matmulABt(X,X,...). BenchmarkMuonStepOnly 418.3 -> 200.0 ms median (2.09x), 3 reps of 30x each, interleaved with BenchmarkAdamStepOnly as an unaffected control. This matches the brief predicted ~200 ms for steps 1-3 exactly. Bytes/op 26.7 -> 28.3 MB (+6%): the transpose scratch is hoisted to once per newtonSchulz5 run rather than per iteration, which recovered most of the 34.6 MB the naive placement cost.

BIT-IDENTITY PROVEN, NOT ASSUMED, and the proof needed strengthening: the first cross-reference gate caught a reversed accumulation order but NOT the insertion of matmulFlat zero-skip, because random fixtures contain no exact zero. That skip is not order-preserving (it drops 0*(+-Inf) NaNs), so an explicit zero/Inf fixture was added and the rewrite deliberately omits the skip. Both mutations now turn the gate red.

GENERALIZED as perfscan PS4008 serial-dot-matmul with 5 fixtures and a mutation-probed detector: 26 candidates tree-wide, and it independently rediscovered all 8 sites of the hand-found SOAP/Shampoo basis-rotation task. Silent on the ikj/axpy form itself, on same-base reductions with no indexed store, and on compound inner bodies.

NOT DONE, deliberately, both split out as follow-ups: step 1 (hoist bm/A/A2/bx onto the Muon struct, which is what would take bytes/op BELOW baseline rather than 6% above) and step 4 (route the three products through ops.MatMul to reach the parallel gemmF64Band, the predicted further ~5x). Step 4 remains unvalidated here because it changes the kernel rather than the loop order, so its bit-identity must be measured against the tolerance-0 tests rather than argued.

Pre-existing unrelated red in the package: TestEMAUpdateBitIdenticalToSlowPath fails identically before and after.
- T-01KYJR5XXZFJ1AFGN2HR674XNT Flatten the SOAP/Shampoo basis rotation and eliminate its bounds checks: LANDED the ikj/axpy half, benchmark-validated on M2 Pro darwin/arm64 go1.26.5 (3 reps of -benchtime 20x, medians): BenchmarkSOAPStepOnly 11.94 -> 9.94 ms (1.20x, brief predicted 1.35x) and BenchmarkShampooStepOnly 7.03 -> 5.60 ms (1.26x). Both Vec controls held flat. Three of the four rotation products convert; the fourth is A.Bt with both operands already walking l contiguously, so it is left as a dot and marked //perfscan:ignore PS4008 rather than paying an n*n transpose allocation per call on a pooled path.\n\nTWO PREMISES IN THIS BRIEF WERE STALE — recorded so they are not re-derived. (1) The allocation defect no longer exists: it cited 758,290 B/op and 976 allocs/op from zeroMat, but the pooled rotateForwardInto/rotateBackInto variants already carry the hot path and the measured baseline was 261,683 B/op / 256 allocs. The 976 -> 100 alloc target was therefore not pursued and is not available. (2) BenchmarkShampooStepOnly does NOT cover the rotations — they are SOAP-only. Shampoo did not move at all until a SEPARATE site was rewritten, the L^-1/4.G.R^-1/4 preconditioner in shampoo.go, which PS4008 had flagged independently.\n\nThe flat-storage half was DELIBERATELY NOT DONE. The brief proposed converting the state matrices from [][]float64 to flat []float64 with an explicit stride. That is a large refactor of the optimizer state, and the sibling evidence is against it: PS4006 row-slice-to-flat already LOST on cholSolve at 0.93x. The ikj reorder is the part with the proven mechanism, and it delivers without touching the struct layout. Anyone revisiting flat storage should A/B it alone, on top of this.\n\nBIT-IDENTITY: a path was found BLIND. Deliberately reversing the accumulation order in Shampoo preconditioner turned NO existing test in the package red. It is now extracted as shampooPrecondInto and gated tolerance-0 against a frozen pre-rewrite oracle, as the SOAP rotations are. All four gates were mutation-probed with a reversed order AND a missing scratch clear; the gates feed deliberately dirty buffers because the ikj form accumulates with += where the dot form overwrote with =. The allocating rotateForward/rotateBack now delegate to their pooled twins rather than duplicating the loops.\n\nPS4008 tree-wide 26 -> 18, soap.go 8 -> 1, shampoo.go 1 -> 0. Remaining PS4008 candidates for later: nn/galore.go (4), backend/ref/mla.go (3), nn/kda.go (1), classic/models.go (1).\n\nPre-existing unrelated red in the package: TestEMAUpdateBitIdenticalToSlowPath, identical before and after.
- T-01KYMCQ31GEB0TW27W6ZN2AR3P Route Muon newtonSchulz5 products through ops.MatMul and hoist its scratch onto the struct: PART B SUPERSEDED AND DELIVERED BY A DIFFERENT ROUTE; PART A STILL OPEN.

Part B asked to route newtonSchulz5's three products through backend/cpu's parallel gemmF64Band, expecting ~40ms (a further ~5x) and warning that a different kernel may block or reassociate, so exactness would have to be re-established rather than argued.

Parallelizing the EXISTING ikj kernel over output rows reaches the same target — 41.8ms, 4.63x, measured interleaved over 3 alternations with BenchmarkAdamStepOnly flat as control — and is bit-identical BY CONSTRUCTION: row i writes only c[i*n:(i+1)*n], both operands are read-only, and the accumulation order within a row is untouched. No kernel swap, so no exactness question to reopen. Scales 193.6 / 68.5 / 43.7ms at 1 / 4 / 12 Ps. Landed as 2d0293de.

THE LESSON: the task assumed reaching the parallel GEMM was the way to get parallelism, when the loop already had an independent axis. Check whether the existing kernel has a partitionable axis before importing another one — the in-place answer kept a tolerance-0 guarantee the swap would have put at risk.

FOUND BY THE SCALING SWEEP, not by re-reading this task: BenchmarkMuonStepOnly measured 0.99x across GOMAXPROCS 1..12, which is what identified it as the package's largest serial spine.

PART A (hoist bm, the transpose buffer and the matmul return buffers onto the Muon struct) IS NOT DONE and is now more valuable than it was: the pool's per-call barrier took allocs/op from 47 to 111 across roughly thirty matmuls per Step. Bytes/op is unchanged at 28.3MB. Re-draft it as its own item if the allocation axis is wanted.

Symmetric-path caveat for anyone tuning further: rows compute only j <= i, so work is triangular and contiguous chunks are unbalanced. A strided partition would balance it and destroy the sequential locality the ikj rewrite exists to exploit; the 4.63x is what that imbalance still leaves.
- T-01KYN31VARFTH9KX8RHYMYFYQ3 Parallelize the SOAP and Shampoo rotation matmuls — two loops need an interchange first: DONE. SOAP 8.076-8.089ms -> 6.092-6.137ms (1.32x), Shampoo 5.872-5.900ms -> 5.318-5.353ms (1.10x), interleaved over 3 alternations with min of 3 runs per arm and BenchmarkAdamStepOnly flat as control. Single core unchanged (8.31 vs 8.36ms). Landed as ce91c955.

THE TASK OVERSTATED THE WORK: it claimed two loops need a loop interchange; only ONE does. Five of the six products are already output-row-outer, each row writing its own slice of the destination, so they parallelize directly and bit-identically with no restructuring. Reading the actual loop bounds before implementing is what caught it — the task was written from a two-sample glance at rotateForwardInto and generalized wrongly to its siblings.

The genuine exception is rotateForwardInto's first product, which loops over the reduction index while every iteration accumulates into every row of tmp. Interchanged to k-outer it is still bit-identical (for fixed (k,j) the sum over i runs ascending in both), worth 1.03x, and kept behind the dual-order guard the GBM histogram established.

METHOD FAILURE WORTH CARRYING, and it is mine: two measurements were initially misread because each arm was a SINGLE run. One 1P sample showed the interchange costing 7% — min of 3 shows it flat. Three single-sample 12P alternations had it winning, losing, then winning — min of 3 shows a consistent 1.03x. The sweep tool was hardened against exactly this one iteration earlier, after a single sample turned a 1.10x into a reported 0.88x, and I did not carry the lesson to hand-run A/Bs until they contradicted themselves. Cast as PROC-BENCH-MINOFN-001.

NOT DONE: rotateBackInto's second product remains a dot rather than an ikj/axpy form. That is declined for ALLOCATION, not speed — the ikj rewrite needs a transposed copy of qr, an n*n allocation per call on a path whose entire purpose is pooling. It is now parallel, which was the available win without touching that trade.

Shampoo gains least because its products are the smallest here and more of its step lies outside them — an Amdahl ceiling, not a weaker transform.
- T-01KYNBK6PAFA5SCPX7W1SP3BW7 Benchmark sinkhorn, kda and nsa — PS6005 flags them but nothing can validate a change: All three modules now have benchmarks, and every PS6010 site is resolved or explained.

Sinkhorn 2.80x (34.83ms -> 12.50ms at 512x512, allocs 521 -> 9): register-blocked both
half-iterations over the output index, then flattened the kernel matrix. An axpy rewrite of
the transposed half was measured and REJECTED (14.60ms vs 13.34ms) — cast as
PERF-ACCUM-RESIDENCY-001.

KDA 1.75x (5.614ms -> 3.15ms): the flagged S·k loop was real but secondary. The decay loop
next to it scaled a COLUMN of row-major S at stride d_k, once per key channel per timestep;
interchanging it is bit-neutral because the decay is a pure elementwise scale. Both output
loops then blocked 4 ways.

NSA 2.70x and allocs 14611 -> 24 (62.1ms -> 23.0ms): two separate findings. Everything
allocated per (head, query) was loop-invariant in size — blockW, the importance slice,
sort.Slice's reflect Swapper, a selected-set map, and a closure per attendMask call.
Replacing the keep callback with a precomputed []bool removed those closures AND an indirect
call per key, which is the honest answer to the PS6010 finding at old nsa.go:130: the
shared-operand reload was second order behind a call that cannot inline. attendMask's score
dot and P·V were then blocked 4 ways for 2.40x on top; P·V had the same column-walk shape as
Sinkhorn and KDA. Both selection sorts became total orders, since ties decide which blocks
and keys get attended and were previously left to the sort's whim. DSA shares attendMask and
gets the same win.

Panic-probed all five flagged sites before trusting any number. Bit-identity gates added for
each module: transcribed references for Sinkhorn and KDA, FNV checksums over raw output bits
for NSA, plus a determinism test on deliberately tied importances.

The one PS6010 site NOT taken is the AtF64 fallback arm of attendMask, which is dead whenever
the F64 fast path applies.

Incidental but blocking: the tree was red and could not be pushed. Root cause was
internal/cichange's tests inheriting GIT_DIR under the pre-push hook and running git against
the real repository, writing core.bare=true and breaking every worktree. Fixed. Five fused
recurrences (EMA, GLA, DeltaNet, GatedDeltaNet, RGLRU, HGRN) also failed their own
bit-exactness claims to FMA contraction on arm64 — fixed and cast as NUM-FUSED-PATH-FMA-001.
go test -short ./... is now clean tree-wide.
- T-01KYQ9RRGEE2CBVF8TM09P19DX Partially fuse NeuralMemory.Scan: backend matmuls, fused elementwise (per ADR-01KYQ9PHNPEFC): Shipped. seq128 10.46ms -> 4.02ms (2.60-2.96x), allocs 24,525 -> 3,305, bytes 39.7MB -> 5.0MB;
seq256 17.32ms -> 5.38ms (3.13-3.27x), allocs 48,845 -> 6,378. Deep variant untouched by design
and measures unchanged. Bit-exact against the dispatch path across dims below/at/above 4 and
sequences through 33; -race clean.

The scoping decision (ADR-01KYQ9PHNPEFC) was necessary but not sufficient. Keeping the three
matmuls on the backend removed the dots as a suspect, which is what finally made the remaining
divergence findable — but the actual defect was in the fused elementwise chain all along, and
it would have broken a fully fused version too.

ROOT CAUSE, after three failed attempts: one unpinned product. `inc := gs[i] * th` assigned to
a named local and then used in `s = float64(s*et) - inc` is still inlined by the compiler and
contracted to fma(-gs[i], th, ...) — one rounding where the dispatch path does two. NAMING A
SUBEXPRESSION DOES NOT PIN IT. This is why t=0 always matched (it computes `s = -inc`, a
negation with nothing to fuse into) while every t>0 diverged by one ulp; that asymmetry was
the visible symptom for three attempts and was misread each time as a momentum-branch bug.
NUM-FUSED-PATH-FMA-001 has been strengthened to say every product, including named locals.

METHOD LESSON: an earlier probe compared the same elementwise chain on SYNTHETIC inputs,
passed, and was taken as proof the chain was correct. It was not — the divergence is
input-dependent. The bug was found only by re-running the diff on the REAL projected and
L2-normalized inputs. A green probe on convenient data is not evidence about the data that
actually fails.

The eliminated hypotheses from R-01KYQ9CQ3XE1D (input handling, matmul accumulation order,
matmul loop shape, vectorized broadcast) were all correct eliminations — none of them was the
cause, and none needs revisiting.
- T-01KYQCWQ5GF928BSDN480N34B5 Replace WandaPrune's per-column sort with a quickselect; extend the panel transpose to F32: Quickselect shipped: 282ms -> 55ms (5.11-5.15x). Combined with the panel transpose,
WandaPrune is 348ms -> 55ms, 6.31x cumulative, at unchanged allocations. M2 Pro darwin/arm64,
interleaved over 3 alternations, min of 3 runs of 2x per arm.

The full sort was answering a membership question. Only `for r := 0; r < k; r++ {
d[idx[r]] = true }` consumed it, so the order beyond the prefix was never observed — 46M
comparisons per call to decide which half of each column to drop.

Bit-identity rests on the comparator being TOTAL (score, then input index, indices unique).
No two elements compare equal, so the k-smallest SET is uniquely determined and the arbitrary
order the selection leaves inside the prefix is unobservable. The same property is why Lomuto
partitioning cannot degrade on a column of identical scores: index tie-breaking means there
are no duplicate keys. Median-of-three pivoting was still required — score columns are
|w|·‖x‖ products and are frequently near-sorted, the exact shape that takes a first-element
pivot quadratic.

METHOD NOTE worth carrying forward: the property tests (selected set == sorted prefix, across
random/constant/sorted/reverse/few-valued/boundary-tied columns, every k, n from 1 to 64) were
verified non-vacuous by inverting the tie-break and confirming red. That check earned its
keep immediately — the first draft of the column generator had `return c` inside its fill
loop, so every column was near-constant and the suite proved almost nothing. It passed
anyway. A green property test on a generator you have not mutated is not evidence.

NOT DONE, deliberately: the F32 branch and the generic AtF64 fallback keep the full sort.
The transform and its correctness argument carry over unchanged, but only the F64 path has a
benchmark and this campaign does not ship unmeasured changes. The F32 half of this task
therefore remains open; it needs an F32 benchmark first, and the panel transpose from the
previous commit is in the same position.
- T-01KYQEBM14EPQT7N7PBMHS1C5N Restore DSA's total-order comparator and slices.SortFunc after main's concurrent rewrite: Both items shipped. DSAAttention seq512 19.40ms -> 18.77ms (1.033-1.036x), allocations
1,547 -> 13, bytes 584KB -> 538KB; seq1024 3,083 -> 13 allocations. M2 Pro darwin/arm64,
interleaved over 3 alternations, min of 3 runs of 5x per arm.

The wall-clock win is small because the sort is not the dominant term — the O(seq^2*idxDim)
indexer scoring is. The allocation collapse is the result worth having: sort.Slice reaches
its swap through reflectlite.Swapper and allocates once per query.

THE TIE-ORDER ITEM CHANGED NOTHING OBSERVABLE, and that is the honest finding rather than a
disappointment. Hashing the output over a fully tied input — every indexer row identical —
gives the same bits before and after: Go's pdqsort already happened to leave tied elements in
index order. For the same reason a repeated-run determinism test passes on the OLD code,
because pdqsort is deterministic; the order was stable, merely unspecified. So this is
hardening against a Go version or sort implementation changing that incidental behavior, not
the repair of an observed defect. The task body claimed it would be a behavior change; that
prediction was wrong and is corrected here.

METHOD NOTE: the determinism test was written expecting it to fail on the old code. It
passed. Rather than delete it or claim it proved something, the check was repeated as a
direct before/after hash on tied input, which is what actually answered the question. A test
that passes on the code it was written to indict has not validated that code — it has failed
to discriminate, and the difference matters.

A benchmark for DSAAttention now exists (seq512 and seq1024); there was none, which is why
these two changes could not simply be re-applied after the merge took main's version.

## PROC-BENCH-MINOFN-001
IF an A/B arm is measured from a single benchmark run, THEN the result SHALL be re-measured as the minimum of at least 3 runs per arm before it is reported; single samples inverted 2 verdicts in one session.

## PROC-SPLIT-SEARCH-FOLD-001
IF an expensive independent search feeds a cheap order-dependent reduction in one loop, THEN the loop SHALL be split into a parallel search pass writing an array and a sequential fold over it, since chunked partials reassociate; used in AQLM k-means and the GMM E-step.

## PERF-MEASURE-THE-COMPOSITION-001
WHEN two bit-identical optimizations of one loop were each measured as a win separately, the implementing agent SHALL produce a 3-arm benchstat report and keep the composition only if it wins both columns.

Rationale: Both outcomes have now been measured, and the rule exists because neither is predictable. LOST: merging two branches that had each optimized the MoBA and DSA P·V loops gave register-blocked sparse iteration 48.89ms/46.07ms on DSAAttention_seq1024/MoBAAttention, the composition with a contiguous v-row axpy 53.14ms/57.20ms (+8.7%/+24.2%), and dense axpy 79.01ms/68.40ms. WON: merging two branches that had each optimized WandaPrune gave panel transpose 57.64ms, per-column fan-out 20.46ms (-64.5%), and the composition 15.55ms (-73.0%), beating main own arm by a further 24%. The discriminator is whether the two fixes attack the SAME cost or different ones. MoBA both targeted memory traffic in opposite directions, so the axpy reintroduced the output-row traffic the register accumulators existed to remove. Wanda targeted orthogonal costs — cache lines per gather versus idle cores — so they added. Bit-identity constrains arithmetic, not traffic or scheduling, so it never predicts which case applies. Note the composition can also carry its own new cost: Wanda fan-out multiplied the panel scratch by the worker count, +23.6% B/op, halved to +11.3% by pooling.

## TEST-BITDIFF-MISSES-SHARED-SCRATCH-001
IF a two-arm bit-identity test is the only guard on a fan-out with per-worker scratch, THEN the implementing agent SHALL also run go test -race over it, since a bit-diff missed a shared-scratch race in 12 of 12 tests.

Rationale: Measured on two fan-outs in nn/wanda.go, with opposite outcomes, which is why this cannot be reasoned about and has to be checked per site. WandaPrune panel fan-out: hoisting the worker scratch out of the closure so all workers share it reddened the two-arm GOMAXPROCS gate. WandaPruneNM fan-out: the same mutation left that gate AND all eleven other Wanda tests green, because a racing write can still land on the value the serial arm produced, so the bit comparison sees nothing. The race detector reported it immediately in both cases. The two guards cover different failure classes — the bit-diff catches partition-DEPENDENT logic such as an off-by-one chunk bound, which -race cannot see, and -race catches unsynchronized sharing, which the bit-diff may not. Neither substitutes for the other. CI runs the cgo+race lane and make race exists locally.

## PERF-PER-WORKER-ALLOCS-ARE-BOUNDED-001
IF a fan-out multiplies a fixed set of scratch buffers by the worker count, THEN the implementing agent SHALL accept the allocation growth when it is O(GOMAXPROCS) and report both numbers, not just the speedup.

Rationale: Two rules in this spec pull in opposite directions on allocation count and the distinction is the scaling, not the direction. PERF-SLAB-IS-RESOURCE-NOT-SPEED-001 removes O(n) allocations — one per data row — and measured no latency change at two sites, so it is purely a resource win. Fanning out duplicates a FIXED set of buffers per worker: MoBAAttention went from 12 to 81 allocations and DSAAttention from 16 to 136.5, bytes up 7.2% and 34.9%, in exchange for 84.7% and 76.5% less wall clock. That increase is bounded by GOMAXPROCS regardless of input size, so it does not grow with the workload the way the slab pattern does. Accept it, but report both numbers: a commit that shows only the speedup hides a real peak-memory change, and one that treats any allocation growth as a regression would block a 6.5x win. The test to apply is whether the growth scales with the data or with the core count.

## PERF-MERGE-PREFER-THE-PASSING-ARM-001
IF two branches implement the same fused kernel and one fails a bit-exactness assertion, THEN the implementing agent SHALL keep the arm whose parity test passes, cite NUM-FUSED-PATH-FMA-001, and list the coverage given up.

Rationale: Eight times this session both branches built the same optimization independently, and until now the merge question was only which composition measured faster. The Titans fused inference path is the first case where one side is numerically WRONG: main implementation plus main own parity test fail in the merged tree at seq=5 dim=16 by one ulp, dispatch -0.037425319772220855 against fused -0.03742531977222085. That cannot be blamed on the other branch, because the test computes both arms in one process from the same inputs, so any shared dependency moves both equally and cannot produce a dispatch-versus-fused divergence. The cause is visible in the diff: the passing version rounds every intermediate product explicitly, three casts, as NUM-FUSED-PATH-FMA-001 requires because the dispatch path rounds between every backend op; the failing one has a single cast. Choosing the passing arm cost real coverage — the failing version also fuses the DEEP memory variant, which the kept one leaves on the dispatch path — and that loss must be named in the merge record rather than omitted, together with the fact that the deep fusion had no bit-exactness test at all. A slower correct path beats a faster one that is an ulp wrong, and the shortfall becomes a task rather than a silent regression.

## NUM-CLAIM-MUST-MATCH-ASSERTION-001
IF a commit or comment claims bit-exactness while its test asserts a tolerance, THEN the implementing agent SHALL treat the claim as untested and measure it with math.Float64bits before relying on it.

Rationale: The fused deep-memory Titans path shipped describing itself as bit-exact on the default build, and its parity test asserts a tolerance whose own comment explains why (the amd64 SIMD sigmoid differs by about one ulp). The tolerance is therefore the only tested property, and the stronger claim went unchecked. Measured on darwin/arm64 with no SIMD sigmoid in play, the path differs from dispatch at every geometry tried, including two the commit ships with: 22 of 30 values at seq=6 dim=5 hid=7 (maxRel 1.27e-14), 49 of 80 at seq=5 dim=16 hid=8, 732 of 792 at seq=33 dim=24 hid=40 (maxRel 2.65e-10). The error grows with sequence length because the memory is a recurrence, so a per-step rounding difference compounds across the scan and a short fixture understates it. Two consequences. A tolerance test cannot support a bit-exactness claim, so the claim must either be asserted with Float64bits or restated as the tolerance that actually holds. And a recurrence needs a LONG-sequence case: the seq=1 geometry in the same suite differed in only 2 of 8 values and would have looked like noise.

## NN-EXEC-CLONES-BYPASS-THE-POOLS-001
IF an nn block type is given its own exec dispatch wrapper, THEN the implementing agent SHALL route it through nnIns1Pool..nnIns3Pool instead, since 43 clone wrappers already bypass them across 450 call sites.

## AN-EXPLICIT-CONVERSION-SUPPRESSES-FMA-A-VARIABLE-DOES-NOT-001
IF a fused path is pinned bit-exact against a path that does not contract the same way, THEN the implementing agent SHALL wrap each product in float64(...); assigning it to a local left all 32 FMADDD in place and arm64 still diverged by 1 ulp.
