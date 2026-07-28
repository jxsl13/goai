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
