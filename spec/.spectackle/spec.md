---
schema: v1
prefix: PERF
---

## PERF-RESEEDED-LOOP-001
WHEN perfscan PS3072 reports a loop reseeding its carried object from the current item, the optimizing agent SHALL band that loop, allocating 1 copy of the reseeded object and of every reused scratch inside each parallelChunks band.

Rationale: The carried state is dead: it is overwritten before it is read, so each iteration is a pure function of its own item and the loop bands despite reading as a chain. A running count still crosses iterations, but integer addition is order-free, so it does not block the transform. The win depends on the body leaving its scratch as it found it, which the per-worker copy guarantees. Measured on nlp watermark Detect, whose per-token partial Fisher-Yates reseeds a PCG from key and the previous token: BenchmarkWatermarkDetect 39.17 to 7.17 ms, -81.7 percent, bit-identical, race clean, control benchmark flat, from a parallel-scaling ratio of 0.99x that made it look already-parallel. Automated as perfscan PS3072 serial-loop-reseeded-from-the-item, which reports the pre-change loop and is silent on the fixed tree.

## PERF-RESEEDED-LOOP-002
WHEN such a loop is banded, the optimizing agent SHALL accumulate each count or sum per worker, fold after the fan-out, and prove bit-identity at 3 or more sizes straddling the fan-out gate.

Rationale: Integer addition is order-free, so a running count crossing iterations does not block the transform, but a per-worker scratch that is not restored does. Measured on nlp watermark Detect: 39.17 to 7.17 ms, -81.7 percent, bit-identical, race clean. Automated as perfscan PS3072.

## PERF-INVARIANT-OPERAND-001
WHEN perfscan PS3073 reports a loop calling a reduction helper with a slice operand that does not vary with the loop, the optimizing agent SHALL jam 4 iterations into 1 pass over the shared operand, each result keeping the helper own accumulator count and combination.

## PERF-INVARIANT-OPERAND-002
WHEN a jam is applied to a factorization or kernel with more than one dtype arm, the optimizing agent SHALL apply it to every arm and cover each arm with its own frozen digest, not 1 digest over the default dtype.

Rationale: The sharing that pays is the SINGLE pass over the shared operand, not the unrolling: the same 4 rows written as two calls to a 2-row helper reloaded the operand twice and measured nothing. Measured on linalg cholFactor, whose row update called a 4-accumulator dot with the pivot row shared: Cholesky512 8.99 to 6.25 ms -30.5 percent, Cholesky256 -24.7 percent, CholSolve256x128 -13.7 percent, bit-identical in both dtype arms, SVD flat as a control. A mutation that read the wrong row in the second arm was invisible until an F32 case was added. PS6010 is the same transform for an inline accumulator loop; a reduction behind a call is invisible to it.

## PERF-PARALLELCOLS-FLOOR-001
IF a per-worker work floor is proposed for the linalg parallelCols helper, THEN the optimizing agent SHALL leave the helper unchanged, since floors swept from 2^12 to 2^16 were flat at best and cost Factor512 up to 26 percent.

Rationale: Measured on the M2 Pro host: baseline Factor512 16.30 ms, LUSolve_512x512 6.03, CholSolve256x128 2.77, Lstsq_256x64x128 1.21. At floors of 2^12, 2^13, 2^14, 2^15 and 2^16 the Factor512 cell read 16.81, 16.28, 16.54, 17.27 and 20.55 ms, with the other three cells inside noise throughout. This matches the history already recorded in perfscan PS3061: the transform pays only in the decode regime, where thousands of near-gate calls occur per generation, and is neutral or harmful elsewhere. Neutral is not a reason to add a knob. What the profile actually says about these cells is a barrier per column with shrinking work; the lever there is a blocked algorithm, which is not bit-identical.

## PERF-FLAT-RESULT-EXECUTION-001
IF a transform measures flat and the benchmarks chosen for it were picked by name or by shape, THEN the optimizing agent SHALL count executions at the site with an atomic counter read from a TestMain AFTER m.Run before recording the flat result, since a plain test reads 0 (tests run before benchmarks).

Rationale: Measured: jamming the F32 arm of the cpu attention forward read flat across five F32 attention cells, all of which route to a GEMM kernel and never reach that arm. The counter said 12288 and pointed at MHA512/fwd/cpu, the one cell missing from the sweep, where the same change measured 8.42 to 6.73 ms, -20.1 percent. A flat result from cells that never execute the code is not evidence about the transform.

## PERF-CODEGEN-NEIGHBOR-001
IF an inline optimization added to one arm of a function moves a bit-exact parity result in another arm that was not edited, THEN the optimizing agent SHALL move the new code into its own function, returning that arm to 0 differing elements.

Rationale: Measured on the cpu attention forward: jamming the F32 scoring branch inline moved 1 ALiBi element of the F64 parity comparison by a single ULP, with no arithmetic of the F64 arm touched. The new helper alone was green; only the enlarged mhaFwd body reproduced it. Extracting the jam into scoreRowF32 restored that arm exactly and kept the whole win, so the cause is the function size and not the edit.

## PERF-SHARED-RANGE-SUBJECT-001
WHEN perfscan PS3074 reports an item loop whose inner loop ranges over an operand that does not vary with the item, the optimizing agent SHALL jam the item loop 4 items per pass over 1 traversal of the shared operand, each item keeping its own accumulators.

Rationale: The operand is re-streamed once per item while the body writes per-item output, so one traversal serves four items. A shared accumulator stays bit-identical by being held in a local across the four additions in the same ascending item order. Measured on the cpu attention backward, whose two key loops re-streamed the gradient row and the query row: BenchmarkMHA512/bwd/cpu 24.73 to 13.40 ms, -45.8 percent (1.85x), forward and masked cells flat as controls, F64 parity bit-exact. This shape is invisible to PS6010, which requires the shared operand to appear as an index expression; a range subject never does, which is why the line holding 39 percent of that benchmark was reported by nothing.

## PERF-SHARED-ACCUMULATOR-001
WHEN perfscan PS3075 reports an item loop whose inner loop accumulates into a buffer that does not vary with the item, the optimizing agent SHALL jam the item loop 4 items per pass, holding the accumulator element in a local across their 4 additions and storing once.

Rationale: Every item otherwise makes a full load-store round trip through the shared buffer for one addition each. Bit-identical when the four additions keep the same ascending item order: the accumulator sees the same sequence of adds, in a register instead of memory. Measured on the cpu selective-attention kernel, where this weighted-sum loop was 26 percent of the profile and a score loop in the same function 46 percent: MHASelectF32CPU_1024x1024x64x16 191.3 to 100.9 ms, -47.3 percent (1.90x), the 512 cell -42.7 percent, the F64 arm -40.1 percent, masked cell flat as a control. This is the mirror of PS3074 — there the inner loop subject is shared and the outputs per item, here the outputs are shared and the subject per item — and neither is reachable from PS6010.

## PERF-BITIDENTITY-GATE-001
WHEN a transform claims bit identity and an existing parity test is used as its gate, the optimizing agent SHALL confirm the gate reddens when 2 of the reordered additions are swapped, before trusting it.

Rationale: A parity test proves agreement, not bit identity, and a jam preserves the second only. On the selective-attention kernel the parity test was byte-exact against ref but exercised no masked selector and only 1 of the 2 dtype arms, so the new fall-back path and the whole F32 instantiation were ungated until both were added. A swap of two additions to the shared accumulator changes no value materially and must still turn the gate red.

## PERF-ACCUMULATOR-INDEX-FORM-001
IF a check recognizes an accumulator only by a bare loop variable index, THEN the optimizing agent SHALL widen it to accept the inner variable as a term of a SUM, which found 11 further sites in this tree.

Rationale: A kernel that addresses its output as os[ob+d] rather than slicing a row first holds the same shared accumulator. Measured consequence: the MLA kernel had two near-duplicate arms and the check reported only the one using a scratch slice, leaving the arm beside it unreported. A product is not the same thing — an index that multiplies the inner variable addresses a different element per item and must stay excluded.

## PERF-UNROLL-SWEEP-001
WHEN perfscan PS3076 reports a register-blocked loop taking 2 steps per pass, the optimizing agent SHALL sweep the factor over 3, 4, 6 and 8 and keep the measured winner, rather than reasoning about register pressure.

Rationale: Both gemm band kernels carried a comment arguing that four would spill, and the swept optimum was six with eight back at the baseline — the spill boundary was real but placed two steps too early. Measured: MTAForward_ch16 277.2 to 250.3 ms (-9.7 percent) and ch8 -9.5 percent on the f32 band, GemmDirF64_1024 19.10 to 15.74 ms (-17.6 percent) and the 512x2048x2048 cell -11.7 percent on the f64 band. The sweep is cheap because every arm is bit-identical: each accumulator takes one rounding per step in ascending order, never a summed pair, so one bit-exact oracle gates all of them and each arm costs a single benchmark run. A register-pressure argument cannot see the scheduler.

## PERF-GEMM-CROW-BLOCK-001
IF a wider C-row block is proposed for the gemm band kernels, THEN the optimizing agent SHALL leave it at 4 rows, since a sweep of 2, 6 and 8 found no factor that wins on more than 1 shape.

Rationale: Swept on the f32 band at 6 B rows per pass: MTAForward_ch16 read 251.0 ms at 4 rows, 268.8 at 2, 266.8 at 6 and 285.1 at 8. Pairing 6 C rows with 4 B rows did win that cell, 251.0 to 236.8 or -5.7 percent, but simultaneously cost MTAForward_ch8 3.0 percent (39.4 to 40.6) and left TPAForward_1024 flat, so it trades one shape for another rather than improving the kernel. The B-row factor is the axis that pays and is already at its swept optimum of 6 (PERF-UNROLL-SWEEP-001). Splitting the blocking by shape would need a regime variable identified by measurement, not guessed.

## PERF-UNROLL-SWEEP-002
WHEN perfscan PS3076 reports a jam whose accumulators are scalar locals taken 4 at a time, the optimizing agent SHALL sweep the factor over 6, 8 and 10 and keep the measured winner, expecting a non-monotone curve.

Rationale: Four is simply what a first jam produces and is rarely measured afterwards. On the memory-retrieval tile, swept at 6, 8 and 10 the cell read 103.1, 85.9 and 101.9 ms against 94.7 at four: eight wins by 9.3 percent while six and ten are both worse than four. No register-pressure argument predicts that shape. Gate the sweep with an oracle whose input lengths include one at exactly factor minus 1 modulo the factor, or an off-by-one in the jam bound reads past the last item and stays green.

## PERF-WEIGHTEDSUM-JAM-001
IF a wider jam is proposed for the shared attention weighted sum mhaWeightedSum, THEN the optimizing agent SHALL leave it at 4 keys, since 6 and 8 measured inside the noise band on every cell.

Rationale: Swept on the masked kernel: MHAMaskedF32CPU_1024x1024x64x16 read 80.93 ms at four keys, 78.47 at six and 79.97 at eight, against a run-to-run spread of 78.5 to 84.7 on the same arm, and MHAMasked_cpu_512 read 21.52, 21.53 and 21.29. Nothing separates them. This loop is bound by streaming the value rows rather than by the accumulator round trip that the jam to four already removed, so a wider factor has nothing left to amortize. The SCORE jam in the same kernel is a different story and did pay at eight (PERF-UNROLL-SWEEP-002).

## PERF-TOLERANCE-ORACLE-001
IF the only gate for a bit-identical transform compares 2 independent implementations, THEN the optimizing agent SHALL freeze a digest of the changed path against ITSELF instead, since 2 implementations agreeing to 1e-9 cannot witness bit identity.

Rationale: Measured on the Titans deep scan: its fused path and its dispatch oracle already differed by 1 ulp on this host before any change, so raising that comparison to a bit-exact assertion fails on unmodified code and cannot be used. A digest of the fused path frozen on the pre-change build is what witnesses the jam. The same test also had every dim and hid divisible by four, so neither jam by-one tail ran; add shapes with remainders when the transform introduces one.

## PERF-SELF-REPORTING-CHECK-001
WHEN a check recommends a jam and its first converted site is still reported afterwards, the optimizing agent SHALL suppress operands that a wide-stride loop in the same function already amortizes, which took PS6010 from 52 to 49 findings.

Rationale: Applying a jam leaves a by-one tail whose body is the reported shape exactly, so without the suppression the check reports its own fix forever and every later round re-triages the same sites. PS3074 and PS3075 were built with this suppression from the start; PS6010 predated it and reported the Titans scan twice after that scan was converted.

## PERF-QMATMUL-POOL-001
IF a resident worker pool is proposed for the gguf quantized matmul fan-out, THEN the optimizing agent SHALL leave the per-call goroutines, since a pool measured flat on wall clock and saved only 2 percent of context switches.

Rationale: A 500-token quantized Llama profile put 91 percent of samples in pthread_cond_signal and cond_wait, and runtime.newproc waking a P was 770 ms of the 1.75 s signal cost, which suggested goroutine creation was the cost. A resident pool with non-blocking submission measured QuantLlamaGenerate500 at 508.7 ms against 511.9 for the spawning form, inside a 508 to 538 spread, with user CPU 4.64 versus 4.62 s, sys 0.59 versus 0.57 and context switches 164337 versus 161116. The samples are the wakeup, not the creation. Decode width is not reachable by a constant either: the whole process runs FASTER at GOMAXPROCS=2 than at 12 (480.7 versus 506.2 ms), and CLA decode is flat from 2 workers upward, so the remaining lever is fewer and larger operations.

## PERF-NARROWED-OUTPUT-GATE-001
IF a digest gates a bit-identity claim on a kernel that narrows its output to float32, THEN the optimizing agent SHALL state in the test what the digest cannot see, after checking that a 1e-10 perturbation moves no digest while a 1.5x one does.

Rationale: The gguf quantized matmul accumulates in float64 and stores float32 on BOTH typed paths, so a ulp-scale reassociation is invisible through its API — measured, not assumed. A digest there gates the failure modes a jam actually has (a bound reading past the last row, a dropped tail row, a row wired to the wrong accumulator), each of which moves a whole value, and nothing finer. Recording the limit is what stops a later round from reading a green digest as proof of bit identity.

## PERF-BENCHMARK-PINS-REF-001
IF a profile attributes time to a backend/ref kernel while an optimized kernel for that op exists, THEN the optimizing agent SHALL read the benchmark context before treating it as a defect, since a harness may pin backend.Reference() deliberately.

Rationale: The logdet backward profile put 12.65 percent in ref choleskyKernel although backend/cpu has registered an optimized OpCholesky since T1127. The harness selects the reference backend itself, so the library resolves the op correctly and there is nothing to fix in it. What that does mean is that the cell measures a kernel no caller uses, which dilutes any change to the VJP body around it; size a fixture on the backend the callers actually take.

## PERF-MINMAX-CLAMP-001
WHEN perfscan PS3077 reports a math.Min wrapped around a math.Max inside a loop, the optimizing agent SHALL replace it with a comparison chain using <= on the low bound, and gate it with both a bit-for-bit table and a caller digest.

Rationale: The two calls carry the whole NaN and signed-zero contract at every iteration. Measured on the HQQ quantizer, 29 percent of whose profile was archMin and archMax against 35 percent of its own arithmetic: BenchmarkHQQuantize 77.14 to 37.79 ms, -51.0 percent (2.04x), an optimizer cell flat as a control. The chain must use <= on the low bound, because < lets a negative zero through where math.Max(0, -0) returns +0, and NaN must fall through both bounds untouched, which is what math.Min and math.Max also do. Both gates are needed: ordinary data never produces those two cases, and the caller digest stayed green under a < for <= mutation that the table caught.

## PERF-PROFILE-LINE-ATTRIBUTION-001
IF a profile puts a large share on a trivial line at the end of a scope that contains an inlined closure, THEN the optimizing agent SHALL test the hypothesis with a cheap edit before building on it, since the line may be absorbing the inlined body.

Rationale: The HQQ quantizer profile put 770 ms of 1270 ms flat on a plain float-to-int store, which read as a bounds-check problem. Writing through a subslice whose length the compiler knows, which removes the check, measured flat: 36.96 versus 37.66 ms min of 3. The cost was the round() closure inlined into the same scope, already visible separately as HQQuantize.func1.1. A line number is not a cause.

## PERF-RADIX-UNIFORM-SKIP-001
WHEN perfscan PS3078 reports a byte-wise radix building its histogram inside each pass, the optimizing agent SHALL build all 8 histograms in 1 traversal, skip every pass whose bucket holds all n keys, and copy home when the surviving pass count is odd.

Rationale: A counting pass whose keys all land in one bucket emits them in read order, so skipping it leaves the same permutation. The fixed 8-pass form always ended on the caller slice because 8 is even; with passes skipped the count can be odd and the result needs copying back, which is the one new failure mode. Measured on the CART builder, where the per-feature radix was 24 percent of the profile: BenchmarkForestFit 88.6 to 79.4 ms, -10.4 percent, winning every paired round, with GBMFit and the SVC cells flat. The win comes from the DATA: float64 bit-keys of one feature column barely move in their sign and exponent bytes within a node. Full-entropy keys skip nothing, so measure the caller instead of assuming the shape pays.

## PERF-RADIX-CALL-COUNT-001
IF the uniform-pass skip is proposed for a radix that runs a few times per fit, THEN the optimizing agent SHALL leave it alone and count the passes first, since the GBM presort measured flat at 640 passes against the CART radix 3515136.

Rationale: Measured: applying the transform to the GBM per-feature presort gave GBMHist_exact_20k 129.2 against 130.0 ms and GBMFit 65.1 against 65.7, both inside noise, while the same change to the CART builder radix took ForestFit 88.6 to 79.4. An instrumented counter explains it — the CART radix runs 3515136 passes across a fit and skips 12.5 percent of them, the GBM one runs 640 and skips 5.6 percent. The two halves of the transform were also separated: 88.6 to 83.14 ms from the single-traversal histogram and 83.14 to 79.49 from the skip, so both are real and the skip is worth more than its pass share suggests because a skipped pass drops a random scatter and a buffer swap. Neither half reaches a sort called a few times.

## PERF-PER-JOB-GATHER-001
WHEN perfscan PS3079 reports a fan-out body calling a function that allocates and returns slices sized by its input, the optimizing agent SHALL recycle the buffers through a sync.Pool taken at the top of the job and returned at its end, and report bytes rather than time.

Rationale: Measured on the random forest, where each tree materialized its own row-pointer slice and label copy: BenchmarkForestFit 34.14 to 21.03 MB per operation, -38.4 percent, and 1905 to 1699 allocations, with the wall clock flat at 79.55 versus 78.07 ms. The safety question is retention and it is answerable by reading what the callee stores: the tree fitter keeps only its fitted root, the class set and the feature count, and the builder holding the rows dies with the call. The buffers must also be fully overwritten before being read. If either is false the pool is a correctness bug rather than an optimization.

## PERF-OP-OUTPUT-ALLOCATION-001
IF an allocation sweep of nn or backend/cpu proposes pooling to cut bytes per operation, THEN the optimizing agent SHALL stop at the op API, since 85 to 99 percent of those bytes are the output tensor each kernel returns.

Rationale: Measured: MTAForward_ch16 is 500 MB per operation of which 85.6 percent is tensor allocF32 under tensor NewOn, and CoPE_512x256_h4 is 179 MB of which 99.6 percent is allocF64 with 78.5 percent under backend Execute. Three alternative explanations were ruled out by measurement rather than argument. GC clearing the scratch pool: raising GOGC from 100 to 1600 moved MTA only 498 to 461 MB and did not improve the clock. A leaking or thrashing pool: getF64 instrumented over an MHA, Conv and MatMul set recorded 105 gets against 105 puts with 12 misses of which only 2 were growth, an 88 percent hit rate whose profile bytes are one-time per-worker fill. Per-job gather churn: real elsewhere and fixed in T1167, absent here. Recycling an output needs ownership semantics the op API lacks, because nothing tells a kernel whether its result is a graph temporary or a value the caller keeps.

## PERF-RANK1-ACCESSOR-WALK-001
WHEN perfscan PS3080 reports a loop making 3 or more AtF64 or SetF64 calls per element indexed by the loop variable alone, the optimizing agent SHALL take the typed slice once when every operand is already the right dtype and keep the accessor arm for the rest.

Rationale: PS1005 requires 2 or more index arguments, so a rank-1 walk is invisible to it — that gap hid the largest win of this round. Measured on the PPO clipped-surrogate backward, which made 4 such calls per element and had NO benchmark of any kind until one was written for it: BenchmarkPPOVJP_65536 2000 to 680 microseconds and the 4096 cell 124 to 42, both -66 percent. The fallback is not optional because the output tensor takes the dtype of its input. The 2 arms cannot be gated as equal bits: the accessor arm stores float32 where the typed arm stores float64, so the assertion that holds is that the accessor result equals the typed one rounded once.

## PERF-SHARED-TYPED-WALKER-001
WHEN 3 or more rules in a package share the same per-element accessor shape that PS3080 reports, the optimizing agent SHALL give them 1 shared walker that hands each rule CONTIGUOUS slices, not 1 typed arm per rule and not a per-element callback.

Rationale: A per-element callback was written first and cost half the win: CPO measured 1888.6 to 1208.8 microseconds that way against 1888.6 to 817.2 with the rule writing its own loop over slices handed to it. The slice form also lets the accessor fallback materialize scratch, run the SAME step and scatter back, so the two arms share one implementation and cannot drift, and one arms-agreement test covers every rule. Measured across the preference-optimization family at 65536 elements: CPO 1892 to 790, DPO 2444 to 983, IPO 954 to 128, KTO 1746 to 877, SimPO 1883 to 778, GRPO 4426 to 1492 microseconds, between -49.8 and -86.6 percent, with the already-converted PPO flat as a control.

## PERF-CHECK-FOLLOWS-HELPERS-001
IF a check suppresses on a code shape that a package helper can provide one call away, THEN the optimizing agent SHALL register the helpers in a pre-scan and follow the call, since testing for the inline form alone reported 5 already-converted reference kernels.

Rationale: PS3080 suppresses when a function already has a typed fast path. Testing only for a literal Storage F64 or F32 read missed the reference kernels whose fast path goes through f64Data, so their explicit else-branch fallback loops were reported as work to do. Following the helper removed 5 of 12 findings while still reporting the pre-conversion PPO site the check was built from, which is the regression that proves the widening did not blind it. The fixture for the new behavior first read as a false positive in the check when it was actually a missing setup step in the test: a registry populated by the pre-scan must be primed from the fixture, or the suppression can never trigger.

## PERF-WEIGHT-INIT-FLOOR-001
IF the per-element generator call in fillGen is proposed as a cost to remove, THEN the optimizing agent SHALL leave it, since an inlined typed loop measured 5.84 against 5.89 ms on a 1048576-element F64 fill.

Rationale: A profile of five nn benchmarks put 21.75 percent of them cumulatively in fillGen, with 98 percent of its flat time on the line that calls the generator — which reads as a closure call per element. Giving fillUniform its own typed loop with the PCG draw written inline measured no difference on either dtype: 5.84 versus 5.89 ms on F64 and 5.77 versus 5.77 on F32. The generator inlines into the loop and that line is simply where its body is attributed, which is the second instance of PERF-PROFILE-LINE-ATTRIBUTION-001 in this session. The remaining 5.5 nanoseconds per element is the PCG draw, the 53-bit float conversion and the store, and none of those can move without changing the value stream. BenchmarkFillUniformF64, BenchmarkFillUniformF32 and BenchmarkKaimingNormalF32 were added so this is one command to re-check.

## PERF-MASKED-LOOP-BLINDSPOT-001
WHEN a hot loop is skipped by PS6010 because its body calls a boolean mask predicate, the optimizing agent SHALL read it by hand and jam it anyway, since a guard call is 1 branch per item and not per-element work.

Rationale: PS6010 declines any body containing a call the compiler will not fold, a predicate calibrated at 95.7 percent precision and 97.8 percent recall. The NSA masked-attention score loop is its second known genuine casualty: it calls keep(j) to skip masked keys and is otherwise exactly the shape the check exists for. Jamming it four keys per pass, with the P and V accumulation PS3075 reported beside it, took BenchmarkNSABranches from 29.49 to 14.82 ms, -49.7 percent, with a CoPE cell flat as a control. Widening the predicate was deliberately not attempted in the same round, because a late change to a calibrated filter would ship unmeasured.

## PERF-GUARD-CALL-EXEMPT-001
WHEN a check excludes a loop because its body contains a call, and the call is reached only through a branch that skips the item, the optimizing agent SHALL exempt that branch and re-count the whole population, keeping the else arm, a computing branch and a calling sentinel store all counted.

Rationale: PS6010 declines any body with a call the compiler will not fold, calibrated at 95.7 percent precision and 97.8 percent recall, and both of its known genuine casualties were mask guards. Exempting a skip-only branch moved the tree from 47 to 49 findings — exactly the two backend/ref/mha_masked_backward sites the calibration note names, with nothing lost — and applied to the NSA score loop as it stood before T1173 it reports that too. Narrowness is what preserves the precision: the else arm still counts, a branch that computes rather than skips still counts, and a call inside the sentinel store still counts except for math.Inf and math.NaN, which the same note records as constant loads and which are what a mask bail-out stores. An arbitrary cap on branch length was written first and removed because no fixture could isolate it and it left a skip branch calling something exempt at any length.

## PERF-MASKED-BACKWARD-TARGET-001
WHEN the masked-attention backward is converted, the optimizing agent SHALL patch BOTH mask variants and gate with a digest frozen on the pre-change code, since the generic arm is dead and the existing tests compare the kernel with itself.

Rationale: OpMHAMaskedBackward is registered in ref only and runs on every default-backend caller: BenchmarkMHAMaskedBackward_512h8 is 89.8 ms with 97.35 percent of the profile in one closure. Its dQ/dK, dV and score loops are the shapes the identical transform measured at -45.8 percent on the cpu masked backward in T1153 and -49.7 percent on NSA in T1173. Two traps are already measured. The loop body appears twice, once per mask variant, and a patch anchored on it matches both — a one-site edit measures as no change at all, which is how this was found. And there is no live oracle inside ref: f64Data succeeds for both registered dtypes so the generic AtF64 arm is unreachable, the parallel test compares the kernel with itself, and the F32 test is a 1e-5 tolerance. Adding a cpu kernel instead preserves ref as the oracle at the cost of a 361-line port against roughly 60 lines for jamming ref in place.

## PERF-PREWIDEN-LOAD-BOUND-001
IF pre-widening an f32 operand to f64 is proposed to remove per-element conversions from a loop, THEN the optimizing agent SHALL measure the byte traffic first, since the quantized matmul lost 3.9 percent to it.

Rationale: The quantized matmul converts 8 activation elements and 1 weight element per step and runs once per output column, so an f32 activation matrix is converted n times over. Pre-widening it once into an m by k f64 buffer removed all of that and measured WORSE: BenchmarkQuantMamba2Prefill_512 273.5 to 284.1 ms, plus 3.9 percent, with decode flat. The loop is load-bound, so doubling the activation bytes costs more than the conversions saved, and a widening instruction rides in the load pipeline at close to no cost. The lever that follows is the opposite one: block over the output column so the same activation loads feed twice the FMAs.

## PERF-OUTPUT-COLUMN-BLOCK-001
WHEN a load-bound inner loop streams its shared operand once per output column, the optimizing agent SHALL block the output column by 3, passing the per-column scratch as named parameters rather than a slice of slices.

Rationale: The quantized matmul reads 8 activation elements and 1 weight element to do 8 FMAs, once per output column, so the activation matrix is streamed n times. Blocking the column gives 8 or 16 or 24 FMAs for the same 8 activation loads. Swept at 2, 3 and 4 columns it read 237.2, 220.3 and 261.3 ms against 270.8 for the per-column form: three is the optimum and four already spills, since 24 accumulators fit and 32 do not. Passing the scratch rows as named parameters rather than a slice of slices is worth another 4 percent — the swept forms indexed sc[c][ki] and measured 237.2 at two columns where the named form measures 222.5. Final: BenchmarkQuantMamba2Prefill_512 276.2 to 216.4 ms, -21.6 percent, with decode flat because m equals 1 takes the tail.
