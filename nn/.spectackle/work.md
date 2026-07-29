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

## T-01KYMDP9EMFTBT3952B5NMZXTN Assess the last PS4008 site, nn/kda.go, and the three in backend/ref/mla.go
kind: task
state: draft
created: 2026-07-28

REMAINING PS4008 candidates after three landed rounds (Muon 2.09x, SOAP 1.20x / Shampoo 1.26x, GaLore 1.75x): nn/kda.go (1), backend/ref/mla.go (3), classic/models.go (1). Tree-wide PS4008 is 12, down from 26 when the rule was minted.

The transform is now routine and its bit-identity argument is settled: transpose or hoist so the inner loop is an axpy over independent accumulators, keeping the summation index ascending, which preserves every rounding. What is NOT routine is whether each site is HOT enough to be worth it — all three landed wins were on optimizer Step paths called once per parameter per step, and a cold or rarely-reached site does not qualify under the standing mandate.

SO THE FIRST QUESTION IS HOTNESS, NOT CORRECTNESS. Before touching any of these, establish a benchmark that actually reaches the site and shows it in the profile; GaLore had NO benchmark at all until one was added for it, and a site with no gate cannot be validated. backend/ref/mla.go is a REFERENCE backend, which historically means correctness-first and not on any hot path, so it likely fails the hotness bar outright — check its callers before writing a benchmark for it. classic/models.go is outside the nn/nlp/autograd lane.

METHOD, established over three rounds: (1) benchmark the site end to end, with an unaffected control benchmark interleaved in the same session; (2) write a tolerance-0 cross-reference test against a FROZEN copy of the pre-rewrite loop before changing anything; (3) mutation-probe that gate by reversing the accumulation order and, if the rewrite accumulates into reused scratch, by removing the zeroing pass — feed deliberately dirty buffers, since clean ones hide a missing clear; (4) A/B, 3 reps, medians.

TRAPS, all three hit in practice: do NOT carry over a zero-skip (if av == 0 continue), which drops 0*(+-Inf) NaNs and is not order-preserving; destinations that are freshly make()d are already zero and need no clearing, but POOLED scratch does; and if a site is deliberately left as a dot, verify the //perfscan:ignore actually suppresses it by re-running the scan — a directive that does not apply is silently inert.

A site may legitimately be DECLINED as A.Bt: when both operands already walk the summation index contiguously, the ikj form buys nothing but costs a transposed copy per call. Two sites were declined on exactly that ground (soap.go rotateBackInto second product, galore.go projectDown right). Record a decline with its reason rather than leaving the finding unexplained.

## R-01KYN72D18E63B15N12HE4KAFX AQLM encode 2.06x; PS6006's rework predicted the blocker instead of -race finding it
kind: research
state: draft
created: 2026-07-28

BenchmarkEncodeAQLM_256x256 ran 986ms at 1.00x across GOMAXPROCS 1..12 — the largest serial spine left in nn. icmEncodeAQLM is 61.6% cumulative and refines each group independently.

MEASURED INTERLEAVED, 3 alternations, min of 3 per arm: 983-995ms -> 478.8-481.6ms, 2.06x. Scales 1000 / 551 / 481ms at 1 / 4 / 12 Ps. kmeansAQLM (18.7%) and the codebook refit remain serial; Amdahl caps this at 2.30x.

THE POINT OF THIS RECORD IS THE FOURTH SHARED SCRATCH. target was a shared residual buffer, after GaussianMixture.logGaussian's solve buffer, gbmBuilder.vals and gbmBuilder.part. The first was found by -race AFTER shipping a racy version that measured a misleading 1.16x; this one was EXPECTED, because PS6006 was reworked from a name heuristic to a structural test one commit earlier. The rework paid for itself immediately: four instances in four unrelated files is not coincidence, it is what a per-call buffer hoisted onto a receiver looks like everywhere.

GATE: TestAQLMDeterministic could not serve — encodes twice with the SAME code, proving reproducibility rather than preservation. ICM is a coordinate-descent search whose every step is an argmin, so a reordering lands on different codes and passes unchanged. A frozen rolling hash over ALL 768 codes plus the refit codebook row replaced it.

HASHING EVERY CODE RATHER THAN A PREFIX IS LOAD-BEARING: groups refine independently, so a partitioning bug corrupts a MIDDLE chunk that a prefix check never reads. Probed — dropping the last group of each chunk turns the golden red while every pre-existing AQLM test stays green.

GATE LIMITS, stated: it does not exercise the argmin tie-break, because float distances never tie exactly — the same structural blindness found on the GBM split search, where the fix was a deliberately constructed tie. Here no constructed tie was added: unlike GBM, the tie-break was not touched by this change (the comparison sequence within a group is untouched by a partition), so the property at risk is coverage of the partition, which IS gated. Recording the reasoning so the absence is not read as an oversight.

It is also unmoved by cutting an ICM sweep from three to two, which means the search has converged by the third on this fixture. Worth knowing before anyone tunes that constant expecting the gate to notice.

## R-01KYN7C0EKFA5T7QVQ2P6GQ27W AQLM encode 3.56x complete; the split-search-then-fold pattern, now used twice
kind: research
state: draft
created: 2026-07-28

Closes the AQLM line. Two changes, measured separately, interleaved with min of 3 runs per arm.

  baseline                 990ms
  + ICM parallel           481ms   2.06x  (over groups)
  + nearestAQLM parallel   278ms   1.73x  (two call sites, different treatments)
  cumulative                       3.56x
Scales 1004 / 375 / 280ms at 1 / 4 / 12 Ps.

THE PATTERN WORTH NAMING is how the second change handled k-means. Its assignment loop computes an argmin per point (expensive, independent) and then accumulates into shared sums/cnt indexed by the chosen cluster (cheap, order-dependent). Parallelizing the whole loop with per-chunk partial sums would have been the obvious move, would have been faster to write, and would have been silently NON-bit-identical — chunked partials reassociate the float sums. It would also have passed every existing AQLM test, because TestAQLMDeterministic only proves reproducibility.

Instead: two passes. The argmin runs in parallel into an assignment array; the fold into sums/cnt runs sequentially in the original point order. The expensive half parallelizes and the order-dependent half does not move. Cast as PROC-SPLIT-SEARCH-FOLD-001.

Second use of this shape in the campaign. The GMM E-step needed it for the log-likelihood total, where per-sample contributions are stored and summed afterward in sample order rather than accumulated per chunk. Both times the reduction was a small fraction of the work, so keeping it serial cost nothing measurable and bought exactness outright.

THE OTHER CALL SITE needed nothing special — the residual encode touches only its own group's row and partitions directly. Two call sites of the SAME function, two different correct treatments; deciding per site rather than per function is the actual discipline.

STILL SERIAL, with reasons: k-means centroid update (a reduction like the above but not the bottleneck) and solveLinearAQLM, now the largest remaining share. Gauss-Jordan elimination carries a genuine loop-dependency down the pivots; only the per-row updates within one elimination step are independent, which is a narrower win and would need its own measurement to justify.

GATE: the frozen rolling hash over all 768 codes plus the refit codebook row, added with the ICM change. Red on a partition bug, green throughout here, verified under -race.

## R-01KYN8AXQVF96TY283NATPXTPE AQLM: the worker's 4-way unroll and this branch's group parallelization compose to 4.44x — and each independently validated the other
kind: research
state: draft
created: 2026-07-28

A merge-resolution record, worth keeping because the naive resolution would have silently dropped half the work.

BOTH SIDES OPTIMIZED icmEncodeAQLM, on different axes and without knowing about each other. Main (PR #467) unroll-and-jammed the inner entry scan by 4, amortizing each target[t] load across four squared-distance accumulators WITHIN one group's scan (-41.5%). This branch parallelized the outer group loop (2.06x at the time). Different axes, so they compose rather than conflict.

COMPOSED RESULT on M2 Pro: 617 / 275 / 223ms at 1 / 4 / 12 Ps. Against the ~990ms serial starting point, 4.44x. Main's unroll accounts for the 1-P improvement (1004 -> 617ms); the parallelization accounts for 617 -> 223ms.

THE REBASE PRESENTED IT AS A CONFLICT AND THE OBVIOUS RESOLUTIONS WERE BOTH WRONG. Taking either side whole discards the other's optimization, and a naive marker-strip produced a file with the OLD scalar scan and the unrolled scan both present, the second outside its enclosing loop — it compiled far enough to look plausible and referenced an undefined variable. The correct resolution was to restore main's function VERBATIM as a clean base and re-apply the parallel wrapper on top of it, rather than to edit the conflicted text.

EACH SIDE'S GATE VALIDATED THE OTHER'S CLAIM, which is the useful part. This branch's frozen golden — a rolling hash over all 768 codes plus the refit codebook row, captured BEFORE either change — still passes on the composed code. That independently confirms the worker's "bit-exact" claim for the unroll, which their commit asserted from the ascending-j argmin fold but could not check against a pre-change reference, since none existed when they wrote it. The gate was built for one change and paid off on another.

LESSON for parallel lines of work on the same function: when a rebase conflicts inside a hot loop, check whether the two changes are on DIFFERENT AXES before choosing a side. Unrolling and partitioning almost always compose; two rewrites of the same loop order almost never do.

## R-01KYNB3SA4FKER2VAB0G2C60H6 PS6009 triage complete: 3 converted, the rest declined ON MEASUREMENT — and two self-inflicted reporting errors
kind: research
state: draft
created: 2026-07-28

Closes the reflect-swapper sweep in this branch's lane. Every remaining site was measured rather than argued about.

DECLINED, with the measurement that decided it:
- nn routing sorts (moe, moba, nsa, dsa, mod, lossfree, peer — 9 sites). They run per forward pass, which sounded high-frequency, but no swapper or sort frame appears in the allocation profile of ANY existing benchmark. BenchmarkMoEDecodeQwenSparse totals 539 allocs and MixtralSparse 153; the sorts are over a handful of experts and the allocation is dominated by SwiGLU. Per-forward is not the same as hot.
- nlp tokenizer sorts (tiktoken, special, packing, grammar, guided). No swapper anywhere in the BPE encode allocation profile.
- classic/gbm.go:195/:275, linalg/svd.go:100, format/safetensors — once per call.

CONVERTED EARLIER, for contrast: CART radixByFeature (per node per feature) 3.11x, and three KNN sites (per node / per query) 1.50x. The dividing line is call frequency, and it is sharp: nothing between per-call and per-node showed up at all.

TWO REPORTING ERRORS OF MY OWN, both caught before they reached a conclusion and both worth recording as method.

1. FIELD-INDEXED BENCHMARK PARSING IS UNSAFE. A benchmark calling SetBytes emits an extra MB/s column after ns/op, shifting everything after it. A naive awk read reported BenchmarkGPT2Encode at 10,565,506 allocs/op; the true figure is 37, and the 10.5M was its B/op. Caught only because allocations cannot exceed bytes and the accompanying B/op read as zero. Match trailing unit labels, not positions.

2. A CORRECTION THAT DID NOT REACH THE INSTRUMENT. An earlier commit corrected CholeskyVJP from a reported 0.88x to 1.09x and hardened scaling_sweep.sh so the misreading could not recur — but the script's own HEADER went on citing 0.88x as an example of what the tool finds. A stale figure inside the instrument that produced it is worse than one in prose: it is the first thing the next reader sees, and it recommends chasing a defect that does not exist. Fixed. When correcting a measurement, grep for the number, not just the document.

## R-01KYNBK7Z7FF0TMW19APKKQQ1R PS6005 triage: 14 sites in this lane, all low-leverage or unmeasurable — and a single-run scare that was not real
kind: research
state: draft
created: 2026-07-28

Triage of the register-blocking rule's findings outside format/gguf, where it was worth 2.26x (Q8_0) and 1.55x (Q4_0).

SITES AND VERDICTS:
- nn/sinkhorn.go (2), nn/kda.go (1), nn/nsa.go (2) — NO BENCHMARK EXISTS for any of these modules. Not declinable on leverage, because leverage is precisely what cannot be measured. Booked as a benchmark-building task; the benchmark is the deliverable there, not the optimization.
- nn/soap.go (2), nn/shampoo.go (3) — 8.0ms and 6.2ms benchmarks, already parallelized (1.31x / 1.08x) and Amdahl-limited; a further register-block would be a small fraction of a small number.
- nn/galore.go (3) — 1.66ms benchmark.
- classic/models.go (1) — no hot benchmark reaches it.

Nothing here approaches the gguf case, where the flagged loop WAS the decode step.

A SINGLE-RUN SCARE, recorded because the reflex it tests is the point. BenchmarkSOAPStepOnly read 8.06ms in a one-shot run at -benchtime 20x, against 6.1ms published after parallelizing it — which looks exactly like a change lost to one of the many rebases this branch has taken. It was not: min of 3 at 60x gives 7.83ms at one core and 5.99ms at twelve, a 1.31x that matches the published 1.32x, and the parallel call sites are all still present. The 8.06 was an un-warmed single sample.

That is the third time in this campaign a single run has produced a misleading number (after CholeskyVJP's 0.88x and GBM exact's 0.93x), and the second time it nearly became a claim. PROC-BENCH-MINOFN-001 exists for this; the useful habit is applying it to alarming readings as readily as to favorable ones, since an apparent REGRESSION is exactly the reading one is least inclined to re-measure before reporting.

## R-01KYPCPWZ9EDYRNWF6J9JHF78Y Sinkhorn register-blocked 2.80x; axpy variant measured and rejected; KDA/NSA baselined
kind: research
state: draft
created: 2026-07-29

Sinkhorn, KDA and NSA had no benchmarks at all, so their PS6010 findings
(output-invariant-operand-reload) could be neither acted on nor declined. Benchmarks now
exist in nn/sparse_bench_test.go, and all five flagged sites were panic-probed to confirm
the benchmarks actually enter the flagged loops before any number was trusted.

SHIPPED -- Sinkhorn, 2.80x total, allocs 521 -> 9 at 512x512:
Host M2 Pro darwin/arm64 go1.26.5 GOMAXPROCS=12; interleaved, 3 alternations, min of 3 runs
per arm, within-arm spread 0.3%.
  256x256 / 50 iters   9.05ms -> 3.86ms   2.28-2.36x
  512x512 / 50 iters  34.83ms -> 12.50ms  2.78-2.83x
Two changes. (1) Register-block both half-iterations 4 ways over the OUTPUT index. The
Kt-u half gained more than the K-v half because k is row-major and accumulating a column
touches one cache line per row while using one of its eight doubles; four adjacent output
columns per pass reuse the line already paid for. (2) Flatten k from [][]float64 to a
single [m*n] buffer (PS4006), worth a further 1.06x and collapsing m row allocations into
one. Bit-identity holds because blocking the OUTPUT index leaves every accumulator walking
its reduction axis in ascending order; nn/sinkhorn_golden_test.go pins this against an
in-test transcription of the unblocked form compared on raw float64 bits, at sizes covering
every remainder class of a 4-way unroll on both axes.

REJECTED -- axpy rewrite of the transposed half:
Replacing the column walk with a scatter into an acc[] vector touches k ONCE in memory
order instead of once per output block. It is bit-identical (acc[j] still sums over
ascending i) and beats the original 2.39x, but it LOSES to register blocking: 14.60ms vs
13.34ms at 512, all three alternations, within-arm spread 1.2%. The premise was wrong.
512^2 f64 is 2MB and resident in this machine's L2, so the repeated passes cost no DRAM
traffic, while acc[j] forces a store-to-load round trip per FMA where four register
accumulators do not. Not taken.

STILL OPEN: the three remaining PS6010 sites. kda.go:78 is a clean matvec shape over S and
kt and should block straightforwardly. nsa.go:72 additionally carries a max reduction that
must be folded across the four accumulators. nsa.go:130 is the weakest candidate -- its
inner loop calls k.AtF64 per element and takes a keep(j) callback with a continue, so
per-element dispatch and the indirect call dominate the shared-operand reload the rule is
pointing at; it should be triaged as PS1001 rather than blocked.

Baselines for the remaining two: KDA seq512/dk64/dv64 5.23ms, 10 allocs; NSABranches
seq512/dm128/heads4/block32 188ms, 14611 allocs.

## R-01KYQ75SC2FZJB9JFV58X8YXVM PS6011 strided-inner-walk: rule built, replay-validated, 3 wins shipped and 1 rejected
kind: research
state: draft
created: 2026-07-29

PS6011 (strided-inner-walk) was added because the same pathology paid three times and no
rule could find it: an inner loop that indexes a flat row-major buffer with the INNER loop
variable multiplied by a stride, so consecutive iterations jump a whole row. PS4006 covers
only the [][]T spelling.

DETECTOR: the discriminator needs no type information. In a column walk the inner variable
is scaled and the outer appears additively (S[r*dk+c] over r, vs[j*dm+off+d] over j);
correct row-major traversal is the mirror image. Two false-positive classes are excluded by
construction — a transpose (out[j*r+i] = x[i*c+j]) strides on one side whichever way it
runs, so interchange only relocates the problem, and a nest whose loop variables never meet
in one index has no interchangeable axes. Suppressing transposes removed 19 of 139 raw
findings; 120 remain tree-wide, advisory.

VALIDATION BY REPLAY MATTERED AGAIN. The first draft searched only the outer loop body's
direct statements and missed its own motivating case, because NSA's P·V loop sits inside an
`if sum > 0`. That is the third rule in this scanner to have missed the case that motivated
it and the second caught only by replaying against pre-fix sources. The regression test for
it was confirmed to fail when the discovery is narrowed back.

VALIDATED AND SHIPPED from the rule's own findings, all bit-identical, all interleaved over
3 alternations with min of 3 runs per arm on M2 Pro darwin/arm64:
- MoBA P·V 1.85x (129.8ms -> 70.9ms). Same shape as the NSA fix. The loop also divided by
  sum on every element, d_k divides per key where one suffices; folding it out is the
  identical arithmetic because `scores[j] / sum * v` associates left.
- RetentionRecurrent 1.54x (16.11ms -> 10.52ms). Had no benchmark at all — only the
  chunkwise form did — so its finding was neither actionable nor declinable until one was
  added and panic-probed.
- RetentionChunkwise 1.70x (4.77ms -> 2.80ms). Carries the walk twice per output channel,
  over R and over V.

REJECTED after measurement: NSA's cmp-branch poolV walk (nsa.go:108) is the same shape but
measured 0.997-1.010x, inside noise. nPast is only i/blockSize, so the loop is negligible
beside attendMask's O(seq) work. This is the rule's advisory NOTE earning its place — a
correct pattern match is not a hot loop, and hotness (SC3) still has to be established
per site.

Gate quality note: RetentionChunkwise's existing duality test compares against the parallel
form at 1e-10 and cannot see a reassociation, so checksum gates over raw output bits were
captured from pre-change sources for both retention functions and for MoBA.

STILL OPEN: 120 candidates tree-wide, unswept outside nn. The AtF64 fallback arms
(retention.go:238, nsa.go attendMask's) are deliberately left — dead whenever the F64 fast
path applies.

## R-01KYQ9CQ3XE1DRXF1FRXBEBE8W Titans fused path attempt 2: recurrence proven correct, divergence localized to broadcast elementwise rounding
kind: research
state: draft
created: 2026-07-29

Second attempt at the Titans fused path (T-01KYQ8ZZQ4EAH). Not shipped, but the fault is
now localized far more sharply and two hypotheses are eliminated. Recording so the next
attempt starts from here instead of repeating the search.

METHOD: built the diff harness the task asked for — an internal test that runs the linear
recurrence BOTH ways for N timesteps and compares S, MEM and OUT element by element after
every step, with backend.Reference() forced on both sides.

ELIMINATED:
1. The recurrence arithmetic is CORRECT. At seq=2 d=3 the harness matches bit-for-bit on S,
   MEM and OUT at every step. The transcription of predict, e2, the outer product, the
   momentum branch and the forget-write is right.
2. It is NOT input handling. Driving Scan directly with controlled q/k/v/eta/theta/alpha —
   bypassing Forward's projections and the q/k L2 normalization — still diverges. The
   earlier suspicion that flatF64 on a normalized tensor was returning the wrong layout is
   dead; seq=2 d=3 passes through Scan directly.
3. It is NOT the matmul's accumulation order or loop shape. ref registers the same plain
   matmulKernel for F64 and F32. Pinning the dots against FMA changes WHICH elements
   diverge rather than fixing it, and rewriting the dot to mirror matmulKernel's exact
   source form (`for p, av := range arow { acc += av * brow[p] }`) reproduces the original
   failure exactly.

THE REMAINING CLUE, and it is a good one: the divergence is PARTIAL WITHIN A ROW. At seq=5
d=8 the first failure is t=1, and only S[14], S[15] and S[31] differ — that is (i=1,j=6),
(i=1,j=7) and (i=3,j=7). A wrong pred or e2 would corrupt every element of row i, since e2[i]
multiplies the whole row. It does not. So the error is per-element in the
grad -> inc -> S chain, and it clusters at the TAIL of a length-8 row.

That pattern points away from the matmul and toward the BROADCAST ELEMENTWISE ops —
Mul(grad, theta) and Mul(S, eta) multiply a [D,D] tensor by a [1,1] tensor, and Sub/Add
combine [D,D] with [D,D]. A tail-clustered one-ulp difference is what a 4-wide vectorized
elementwise kernel with a scalar remainder loop produces if the two paths round differently.
NEXT STEP: read backend/ref's broadcast elementwise implementation for Mul/Sub/Add and check
whether it processes [D,D] against [1,1] in vectorized blocks with a differently-rounded
remainder, then reproduce that shape rather than assuming a plain scalar loop.

WIDER LESSON, worth weighing before the next attempt: a fused path that must reproduce the
bits of a CHAIN of backend ops inherits the rounding behavior of every one of them, not just
the arithmetic. The existing fused paths in nn (GLA, DeltaNet, GatedDeltaNet, RGLRU, HGRN)
each reproduce two or three elementwise ops and got away with it. This one reproduces about
twenty, including two matmuls, an outer product and three broadcast multiplies. If the next
attempt also fails, the honest options are to call the backend for the ops whose rounding
cannot be reproduced (keeping the ~15 slice/transpose dispatches fused, which is where most
of the 24.5k allocations are anyway) or to accept a tolerance-based parity gate for this one
path with an ADR recording why bit-exactness was given up.

The benchmark (nn/titans_bench_test.go, committed) stands: 11.4ms, 39.7MB and 24,527
allocations for one seq=128 d=64 linear forward, about 191 allocations per timestep.
