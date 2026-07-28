---
schema: v1
---

## T-01KYJQ3PXEFMW9RAY47HDFTM18 Give gemmF64Band a NEON kernel and move its accumulators out of memory
kind: task
state: draft
created: 2026-07-27

SITE: backend/cpu/gemm_nosimd.go:22 gemmF64Band and its f32-into-f64 twin :56 gemmF32Band, build tag !(amd64 && goexperiment.simd) — so arm64 with the experiment gets the SCALAR body. The vectorized replacement exists only for amd64 at backend/cpu/gemm_simd.go:31.

WHY HOT, four consumers and two are dtype-agnostic: OpMatMul F64 (gemm.go:130 serial small path, :133 parallel band); Conv2D FORWARD for BOTH f64 and f32 (conv.go:100 and :108 — both dtype arms call gemmF64Band; f32 conv escapes to NEON only via the f32NativeKernels gate at conv.go:58, ahead of this, and f64 has no such escape); Conv2D BACKWARD for both dtypes (conv_backward.go:70 and :79 — there is NO f32NativeKernels gate anywhere in conv_backward.go, so f32 vision TRAINING on this host runs its dW/dX GEMMs through this scalar f64 kernel). BENCHMARKS.md:356 puts the f64 kernel at about 68 GFLOP/s at 1024 cubed against the same box's 758 (NEON f32) and about 2100 (AMX). f64 peak is half f32 peak, so the honest ceiling is roughly 2x above where it sits.

DEFECT, two compounding: (1) no .2D kernel at all — scalar-only path where a vectorized sibling exists; (2) accumulators in MEMORY. The scalar body is ikj with c0[j] += a0*bv for four rows inside the j loop, i.e. per 4 MACs it issues 4 loads of C, 4 FMADDD and 4 stores of C on EVERY p iteration. M2 retires about 2 stores/cycle, so the store port — not the 4 FP pipes — sets the rate at about 2 MAC/cycle against a 4 MAC/cycle ceiling. The amd64 sibling already fixed exactly this (gemm_simd.go:78-107 loads eight accumulators BEFORE the p-loop and stores after, giving 8 independent chains); that restructuring is architecture-neutral and was simply never applied to the shared file.

FIX, two independent steps — LAND SEPARATELY so the A/B attributes:
(2a) PURE GO, NO ASM: restructure gemmF64Band to the gemm_simd.go:57 shape — outer j blocks, inner p, accumulators in locals, hoisted ar0..ar3 := A[i*k:(i+1)*k] slices to kill bounds checks, running bo += n to strength-reduce p*n+j. For fixed (i,j) the p order is unchanged (0..k-1, one chain), so this is bit-exact by construction.
(2b) PLAN9 NEON: add gemmF64Tile4x8Neon in a new gemm_f64_neon_arm64.s modelled on gemmF32Tile4x16Neon (gemm_neon_arm64.s:26) — eight V registers of 2xf64 as a 4x4 C tile, VLD1 the C tile before the k-loop and store once after (preserving the += contract conv.go's im2col scatter depends on), k-loop unrolled by 2, and BY-ELEMENT FMLA Vd.2D, Vn.2D, Vm.D[e] so the A scalar is read from a lane and no VDUP competes for the FP pipes — the same trick that took the f32 tile from 87 to 104.6 GFLOP/s. Go's assembler has no by-element vector FMLA mnemonic; WORD-encode as that file already does throughout. Reuse the existing l2Target column blocking and packBPanels unchanged.

VALIDATION GATE (benchmark only), all existing and all isolating: gemm_grind_bench_test.go:73-75 BenchmarkGemmDirF64_512 / _1024 / _512x2048x2048 (direct kernel entry, no op dispatch — the right level to attribute 2a against 2b); gflops_bench_test.go:40-41; gemm_test.go:99-102 for the op-level view; conv_test.go:78 BenchmarkConv2D_cpu and conv_backward_test.go:83 BenchmarkConv2DBackward_cpu for the conv consumers. Add BenchmarkGemmDirF64_511x513x515 mirroring the existing f32 ragged case to keep the n%8 / m%4 tails honest.

EXPECTED: 2a alone 1.4-1.8x (removing the store-port bottleneck); 2a+2b 2.0-2.6x at 512/1024 cubed, more on the >L2 shapes where column blocking also lands; Conv2D backward should track it closely. Medium-high confidence for 2a (the amd64 measurement records f32 +88% / f64 +60% from column blocking alone, and the store-port argument is structural), medium for 2b (f64 NEON is only 2 lanes, so the win is halved loads/stores and 8 register chains, not lane width).

BIT-IDENTITY BAR: BIT-EXACT, and it must be — TestGemmCrossReferenceExact (gemm_test.go:17) and TestConvCrossReferenceExact (conv_test.go:19) gate this kernel at tolerance 0. Three conditions: (i) vectorize j only, p stays scalar-ordered ascending; (ii) USE FMLA, NOT FMUL+FADD — this is the arm64 inversion now recorded as SIMD-007: the scalar twin at gemm_nosimd.go:41 compiles to FMADDD, so separate mul and add would introduce a SECOND rounding and break exactness, the precise opposite of the amd64 rule in SIMD-006; (iii) load C before the p-loop and store after, preserving +=. The tail must obey the same rule. Add the f64 arm64 case to gemm_simd_test.go:19, which already exists to probe residues at the body/tail boundary.

## T-01KYJQ3QB2EMCR0T09D7XA4TEX Vectorize rowMaxF32 and scaleRowF32 on arm64 — two of every softmax's three passes are scalar
kind: task
state: draft
created: 2026-07-27

BEST EFFORT-TO-PAYOFF RATIO IN THE SIMD UNIT.

SITE: backend/cpu/softmax_avx_other.go:10 rowMaxF32, :20 axpbRowF32, :26 scaleRowF32, tag !(goexperiment.simd && amd64). The file's own comment at :7 says the quiet part out loud — "arm64's SIMD build included — vectorizing these is an amd64-only change". Vectorized siblings exist at softmax_avx_amd64.go:18/:69/:51.

WHY HOT: these bracket the NEON exp on EVERY f32 softmax row in the tree. vexp.go:480 m := rowMaxF32(xr) and :483 scaleRowF32(or, 1/sum) in softmaxVexpF32; :500 and :523 in the wide (1x32000 logit) form; :421-423 in mhaSoftmaxBandVexpF32, the MHA band softmax, on every head of every layer. The comments at vexp.go:420-421 and :480 even annotate them "AVX2 ... / scalar elsewhere". Net effect on this host: pass 1 (max) scalar, pass 2 (exp) 4-wide NEON, pass 3 (scale) scalar. Since the NEON exp costs about 1 cycle/element amortized while a scalar compare-select or multiply costs 1-2, the two scalar passes plausibly dominate the vectorized one.

FIX: three small Plan9 NEON kernels in vexp_arm64.s (which already has the constant block and quad-loop scaffolding), with //go:noescape declarations in vexp_arm64.go next to vexpQuadsNeonF32. rowMaxQuadsNeonF32: FMAXNM Vacc.4S accumulate, horizontal reduce, scalar -Inf-start tail, two accumulators unrolled by 2. scaleRowQuadsNeonF32: VDUP plus FMUL Vd.4S in place. axpbRowQuadsNeonF32: FMLA Vd.4S. Follow the established driver shape (nv := len(x) &^ 3, NEON body, scalar tail through the identical scalar expression) so a value gets the same result regardless of where it lands — the contract vexpRowF32 (vexp.go:446) already documents. Add softmax_neon_arm64.go and narrow the other tag; delete nothing, keep one definition per build.

VALIDATION GATE (benchmark only), existing and well-targeted: softmax_bench_test.go:11 BenchmarkSoftmaxF32_512x512_cpu and :14 _2048sq_cpu; elementwise_grind_bench_test.go:68 BenchmarkSoftmaxF32_1x32000_cpu (hits softmaxWideVexpF32 where all three passes are separately parallelized — the cleanest isolation) and :71 _4x32000_cpu; :62 _32x2048_cpu for attention-row scale; backend/cpu/normattn_bench_test.go for the MHA level. To separate the max pass from the scale pass add direct microbenchmarks BenchmarkRowMaxF32_2048 and BenchmarkScaleRowF32_2048 with b.SetBytes.

EXPECTED: 1.5-2.2x on the 32000-wide and 2048-wide row cases; 1.15-1.3x on MHA forward (softmax is one of three stages there, the two GEMMs already being NEON/AMX). High confidence — the change is small, the ops are trivially vectorizable, and the pass structure is fully visible in vexp.go:470-486.

BIT-IDENTITY BAR, mixed and cleanly separable: scaleRowF32 is BIT-EXACT (a single multiply; FMUL.4S is the identical IEEE op per lane, no reassociation). rowMaxF32 is BIT-EXACT ON ALL FINITE INPUTS, which is the entire domain here — max is associative and commutative over finite floats. The one divergence is NaN: FMAX propagates NaN where the serial if v > m scan skips it. USE FMAXNM, whose IEEE-754-2008 maxNum semantics IS NaN-skipping, making it bit-exact including NaN — it costs nothing and closes the caveat the amd64 sibling explicitly left open at softmax_avx_amd64.go:16. axpbRowF32 is the interesting one: its scalar twin x[j] = x[j]*a + b contracts to FMADDS on arm64, so FMLA.4S is BIT-IDENTICAL here — whereas the amd64 sibling had to accept one rounding versus two. Same code, exact on arm64, tolerant on amd64: a clean illustration of why SIMD-006 and SIMD-007 had to be split by architecture.

## T-01KYJQ7822FVBBKJV7F9T08ZAK Enable the f32 norm fast path on arm64 — RMSNorm/LayerNorm currently normalize through a per-element f64 round trip
kind: task
state: draft
created: 2026-07-27

SITE: backend/cpu/norm_avx_other.go:10 const normF32Fast = false, tag !(goexperiment.simd && amd64), with the exclusion stated deliberately at :8. Consumed at norm.go:143 (rmsNormFwd) and :232 (layerNormFwd); the scalar bodies that actually run here are norm.go:169 and :265. Vectorized siblings at norm_avx_amd64.go:12/:19/:44.

WHY HOT: RMSNorm runs twice per transformer block per token on every Llama/Qwen/Mistral, LayerNorm likewise on GPT. For a 32-layer model that is 64 full passes over [tokens, d_model] f32 per forward. The amd64 file records the normalize pass as the profiled ~45% scalar-f64 hotspot (norm.go:222, norm_avx_amd64.go:7).

DEFECT: not merely unvectorized but actively pessimal. norm.go:169 is or[j] = T(float64(xr[j]) * inv * float64(g)) — per element one f32->f64 convert on x, one on gamma, TWO f64 multiplies, one f64->f32 narrow. norm.go:265 is worse (an extra f64 subtract and a third convert on beta). Meanwhile inv is already an f64 scalar hoisted out of the loop, so the widening buys nothing the row statistics did not already secure: it is the REDUCTIONS that must stay f64, not the normalize. The amd64 side reached exactly this conclusion (norm.go:220-224). Secondarily sumF32/sumSqF32/varSumF32 (norm_avx_other.go:25/:48/:74) are 4-way-unrolled scalar here versus f64-widened AVX2 there.

FIX: add norm_neon_arm64.go setting normF32Fast = true and narrow norm_avx_other.go's tag to exclude arm64+simd. Two 4-wide kernels: rmsNormNormalizeF32 (VDUP inv, then FMUL Vx.4S by Vinv and by Vgamma per quad — direct transcription of norm_avx_amd64.go:57); layerNormNormalizeF32 (FSUB mean, FMUL inv, FMLA with gamma/beta). KEEP THE SUB-THEN-MUL ORDERING, not the algebraically equal x*inv - mean*inv — norm_avx_amd64.go:17 documents why (cancellation when x is near mean). The reductions are a separate lower-priority follow-on: NEON has no convenient f64-widening horizontal accumulate, so FCVTL/FCVTL2 to .2D plus four FADD.2D accumulators, mirroring the four-partial grouping the scalar version already uses so the parity argument carries over unchanged.

VALIDATION GATE (benchmark only), existing and exact: norm_bench_test.go:52 BenchmarkLayerNormFwdF32_512x1024 and :53 BenchmarkRMSNormFwdF32_512x1024; norm_kernel_bench_test.go:31 BenchmarkNormCore isolates the kernel body below op dispatch. Add BenchmarkRMSNormFwdF32_512x4096 — d=1024 may sit L2-resident where d=4096 does not, and the two will show different speedups. norm_bench_test.go:31/:34 cover the backward path this change does NOT touch, so they serve as a negative control.

EXPECTED: 2.5-4x on the normalize pass in isolation (removing 3 converts and 2 f64 muls per element in favor of 2 f32 vector ops per 4 elements); 1.5-1.8x on the 512x1024 benchmark end to end with the reductions still scalar, rising toward 2.2x if the reductions are vectorized too. Bandwidth (12 bytes/element touched) caps it below the compute ratio. Medium-high confidence — the amd64 45% figure is measured and the arm64 scalar body is strictly more expensive than amd64's was.

BIT-IDENTITY BAR: TOLERANCE-EQUAL ONLY, and the budget is already established and already accepted for this transform on the other architecture. Two sources: inv (and mean) become f32-rounded once per row rather than kept f64 through the per-element chain; and FMLA in the LayerNorm form fuses t*gamma+beta to one rounding. norm.go:18-25 states the governing standard explicitly — these kernels are NOT bit-exact against ref by design, and the parity test asserts 4 ulps f64 / 1e-6 f32, with norm_test at rtol 5e-5 for the f32 fast path. The f64 arms are structurally unreachable from this change, since the gate is isF32 && normF32Fast. This is the one item in the SIMD set that cannot be made bit-exact, and it is acceptable precisely because the surrounding kernel never promised bit-exactness.

## T-01KYJR5YCJF4M9BC6F960CGG9Z Investigate the worker-pool park/wake cost that dominates small-model training steps
kind: task
state: draft
created: 2026-07-27

FLAGGED UPWARD from the nn sweep — the finding is in backend/cpu but was found while profiling nn, and if it holds it is worth more than every nn optimization combined for small-model training.

OBSERVATION: pprof over BenchmarkTrainStepAdamF64 (193,592 ns/op, 296,828 B/op, 168 allocs) is dominated by worker-pool park and wake, NOT by any kernel and not by nn at all — runtime.usleep 36.2%, pthread_cond_wait 14.9%, kevent 13.1%, cpu.poolWorker 48.9% cumulative, with total samples at 274% of wall-clock (i.e. cores are spinning, not computing). The actual kernels are gemmF32Band at 2.2% and gemmF64Band at 1.4%.

SITE: backend/cpu/cpu.go:245 parallelWork and the pool's spin/park policy.

WHY THIS MATTERS: at small model sizes the per-dispatch fork/join cost exceeds the work dispatched. Every op in a small training step pays it, and it compounds with any change that increases dispatch count. It also interacts with the Muon task in this same round: routing Newton-Schulz through the parallel GEMM adds roughly 30 fork/joins per step, so if the park/wake cost is as large as this profile suggests, that task's step-4 estimate is optimistic and the two must be measured together.

SCOPE THIS AS AN INVESTIGATION, NOT A PRESCRIBED FIX. The profile is strong evidence that something is wrong, but the shape of the fix (spin-then-park thresholds, a serial cutoff below some work size, batching dispatches, or a persistent worker handoff) depends on measurements this sweep did not make. Concretely: (1) determine the minimum work size at which parallel dispatch beats serial on this host, per dtype; (2) check whether a serial fast path already exists and at what threshold (gemm.go:130 has a serial small path — establish whether its cutoff is calibrated or arbitrary); (3) measure fork/join latency in isolation; (4) only then propose a change.

VALIDATION GATE (benchmark only): BenchmarkTrainStepAdamF64 (nn/train_bench_test.go) as the end-to-end signal, plus a new microbenchmark that dispatches a trivially small parallel op in a loop to measure fork/join cost directly, and the existing gemm_grind_bench_test.go direct-kernel benchmarks at SMALL shapes (the current set starts at 512, which is far above where this effect lives). Sweep shapes down to 32 and 64 — the crossover is what matters, not the large-shape numbers.

EXPECTED: unknown, deliberately. The profile suggests a large fraction of a small training step, but a profile that shows spinning does not by itself prove the spinning is removable — some of it may be unavoidable synchronization that would simply move. State the measured crossover before claiming a win.

BIT-IDENTITY BAR: any change to parallel decomposition must preserve per-output reduction order. The band kernels currently guarantee each C element accumulates its k products in ascending order in one chain; a change that alters banding or work-splitting could break the tolerance-0 cross-reference gate (TestGemmCrossReferenceExact, TestConvCrossReferenceExact). A change that only alters WHEN workers park, not how work is split, is bit-identical by construction — prefer that class.

COORDINATION NOTE: a separate agent was researching the backend package concurrently in this round. Check its findings before starting, to avoid duplicate or conflicting work on the same file.
