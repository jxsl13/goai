---
schema: v1
---

## T-01KYJQ3PXEFMW9RAY47HDFTM18 Give gemmF64Band a NEON kernel and move its accumulators out of memory
kind: task
state: active
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
