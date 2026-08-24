---
schema: v1
---

## intent
- T-01KYJQ78FAFCGA1D639T76XZKC Compute the T5 relative-position bias row directly instead of rebuilding the full matrix per token: LANDED the nlp twin, benchmark-validated on M2 Pro darwin/arm64 go1.26.5. Per call, and the CURVE is the evidence — matmul form roughly quadruples per doubling of pos, gather roughly doubles: pos=32 167,132 -> 1,288 ns (130x, 381,154 -> 2,400 B); pos=128 784,268 -> 3,000 ns (261x, 5,507,630 -> 6,112 B); pos=512 7,080,952 -> 9,929 ns (713x, 86,766,729 -> 19,680 B). End to end, 3 reps of -benchtime 5x, medians: BenchmarkT5Decode500 2,679 -> 556 ms (4.82x) and 13,865,254,414 -> 124,827,840 B/op (111x), allocs 490,802 -> 350,477. The task predicted 10x/100x/1000x at those positions; actual was 130x/261x/713x — same order, higher at small pos and lower at large.\n\nThis also settles why the preceding T5 KV-cache change (T-01KYMEV0VSFFB) measured a marginal 1.15x: this dominated. With it gone, a 500-token T5 decode allocates 125 MB rather than 15.9 GB.\n\nBIT-IDENTITY PROVEN, EXCEPTIONS NAMED: the old value came through a one-hot matmul, a sum of numBuckets products of which all but one were 0*Table[j][h]. For finite tables that sum IS Table[b][h], so the gather is bit-identical — gated tolerance-0 against a frozen oracle over a table filled with a WIDE EXPONENT SPREAD, because the constructor zero-initializes the table and a zero table would have made the gate pass regardless. It differs for +-Inf/NaN (0*Inf = NaN) and for a stored -0 (0 + -0 = +0); both indicate a broken checkpoint and neither changes a downstream sum. The stronger bar the task set is also met: greedy decoding gives token-for-token identical ids over 128 steps.\n\nMETHOD NOTE worth reusing: both gates run the identical decode with ONLY the bias computation swapped, via a test-only t5BiasViaMatmul flag mirroring kvAppendViaConcat. Without it the token test would have compared a run against itself and proved nothing but determinism — it was written that way first and corrected.\n\nGENERALIZED as perfscan PS2007 build-nxn-use-one-row. PRECISION TOOK THREE PASSES and both rejected cuts are kept as fixtures: asking only whether the driving identifier appears in some index anywhere in the function flagged a.AtF64(j, j) — a diagonal element READ, not a builder — and a square attention mask consumed whole. The rule now requires the result OR A VALUE DERIVED FROM IT to be indexed by the driving position, and requires that position not to be a loop bound (when it bounds a loop the object is walked in full and the identifier appears only as a stride). Tree-wide 6 -> 2 findings, both genuine.\n\nNOT DONE, and it is the task original primary target: llamagpu/t5_decoder.go:123, the byte-identical GPU twin, plus llamagpu/t5.go. GPUT5Decoder is reachable only through the CUDA constructors and MHABias is a metal stub, so it cannot be benchmarked on this host; under the standing rule an unmeasurable change does not ship. The fix is mechanical and identical, and PS2007 will keep pointing at it. The task also proposed a persistent per-session row extended by one entry per step (O(heads) per token instead of O(pos*heads)) and hoisting the per-token NewContext; neither was needed to reach 713x and both remain available.
- R-01KZ14QMNHFDR89C3KK4WX1JJJ Round T1059: headArgmax row-order accumulation -52%, and a pre-existing F32-on-f64 panic found in backend/cpu: Consumed: headArgmax shipped at -52.2 percent with two new benchmarks and a tie-breaking gate that took two attempts to become non-vacuous. Two things recorded rather than fixed: a pre-existing panic in backend/cpu mhaMaskedKernelCPU (Storage().F32() on f64, reached with F64 tensors, invisible to the -short CI lane) which needs its own change; and the misattribution that a panic stack naming an ac [body truncated at tombstone retention cap]
- T-01KYJPSEWVFRSRSMX2VEV7AY95 Measure and fuse GPTDecoder single-token dispatch on M2: validated pass by codex-m2-gptdecoder-validator-20260824 diff 1c44a6e6b4be — Promoted M2 Metal GPT dispatch fusion. Residual recordAdd won 21/21 pairs at 1.076643x median; grouped QKV decode won 21/21 at 1.113765x median. The clean design versus the frozen pre-fusion public-Step binary won 7/7 order-alternated pairs at 1.225737x median and 1.204752x floor with unchanged allocations. Metal owns o [body truncated at tombstone retention cap]
- T-01M0STA78PF4PABXVY1FBKZ98J Implement and benchmark Decoder StepInto: Implemented Decoder.StepInto with exact destination validation, reusable F32/F64/half-safe embedding staging, compatibility Step wrapper, and a two-slot Metal recorder-wrapper pool that preserves one-shot native command buffers. M2 Pro D512/L6/V1024 measurements: baseline Step 7 allocs/op and 8576-8578 B/op; warmed StepInto 0 allocs/op and 0-2 B/op; seven order-alternated pairs produced a median b [body truncated at tombstone retention cap]
- ADR-01M0STA4CNESSRA8MS30X6VDFY Preserve Step as a wrapper over caller-buffer StepInto: Decision implemented: Step remains the allocating compatibility wrapper over StepInto; destination length is checked before state mutation; embedding rows write into caller-owned storage; Metal decoder wrappers alternate through a bounded two-slot owner-local pool while every native command buffer remains fresh. This removed the final recorder allocation without changing decode scheduling or logit [body truncated at tombstone retention cap]
- P-01M0ST9BK2FGYTT3D762HHYHVB Add zero-allocation shared Decoder stepping: Delivered the zero-allocation shared Decoder host boundary. StepInto is public, documented, example-covered, bit-identical to Step, and allocation-free after warmup at the tracked M2 boundary. The original seven allocations and 8576-8578 B/op fell to zero with median paired throughput at 1.029 times baseline. Spectackle rules and symbol anchors cover semantics, length guards, embedding staging, re [body truncated at tombstone retention cap]
- T-01M0SWB40KF9GR1EZHGRTDD4AA Implement and benchmark zero-allocation Decoder prefill: Implemented StepNInto and StepNLastInto with exact pre-mutation length guards, direct downloads, recurrent StepInto routing, and a Decoder-owned exact high-water embedding staging slice cleared by Release. M2 D512/L6/V1024 at 16 tokens: historical StepNLast 36864 B/op and 2 allocs/op; retained wrapper 4096 B/op and 1 alloc/op; warmed StepNLastInto 0 allocs/op and 0-1 B/op. Seven in-process order-r [body truncated at tombstone retention cap]
- ADR-01M0SW7XPYFV69M6PYKDM7EHNR Use caller buffers and retained high-water embedding staging for Decoder prefill: Decision validated and implemented: caller buffers plus receiver-owned exact high-water host staging preserve allocating wrappers and exact semantics while removing 32768 staging bytes per 16x512 prefill. Rejected eager Ctx-sized residency, sync.Pool, and duplicated execution graphs remain rejected. M2 paired median throughput ratio 1.001x.
- T-01M0SXVBXBFWQB2XB30NCJC88B Implement and gate allocation-free GPT decode and prefill: Implemented GPTDecoder StepInto, StepNInto, and StepNLastInto over shared execution graphs; exact destination guards precede cache mutation; token and learned-position embeddings gather into one reusable row or exact high-water batch slice; Release clears staging; NewGPT uses the bounded mRecPool with fresh native command buffers. GPT-2-small M2: Step 210992-210994 B/op and 4 allocs became 204800- [body truncated at tombstone retention cap]
- ADR-01M0SXS636FJBA7HAYV1199C07 Use caller buffers, reusable embedding staging, and bounded recorder wrappers for GPT: Decision implemented and validated. GPT caller buffers, receiver-owned embedding staging, and decoder-local recorder-wrapper reuse remove all warmed boundary allocations without duplicating execution graphs. Eager Ctx-sized host storage and sync.Pool remain rejected. Compatibility wrappers preserve result ownership and M2 paired medians remain at 1.000x decode and 1.027x prefill.
- P-01M0SXR7C3E2MRKNF1YVZ48G65 Eliminate GPT decode and prefill boundary allocations: Proposal delivered. GPT StepInto and StepNLastInto reach 0 allocations at GPT-2-small geometry; Step and pp16 StepNLast retain only one 204800-byte result allocation; exact logits and cross-backend API behavior are preserved; M2 throughput clears the 0.97 floor; all local preflight lanes pass.
- T-01M0SZRTGPEJ995A6G4WZPCY9J Implement and gate allocation-bounded GPT generation: Implemented one Vocab-sized logits slice reused by StepNLastInto and every StepInto call while retaining the final cache-advancing decode. Exact Metal token parity and bit-identical continuation logits passed. GPT-2-small maxNew=8 measured 614784 B/op and 4 allocs/op versus 2253184 B/op and 12 allocs/op, removing exactly 1638400 bytes and 8 allocations. Seven order-reversed 50x campaigns ranged 0. [body truncated at tombstone retention cap]
- ADR-01M0SZNGW5ER893A6C9QN52QC3 Reuse caller-owned logits inside GPT generation without changing cache advancement: Adopted a Generate-local caller-owned logits buffer and retained the final StepInto call. Rejected skipping the final decode because continuation cache state is observable; exact subsequent-step logits match the historical Step-based control.
- P-01M0SZN5AHE0TS0198C0CVC36N Reuse one GPT generation logits buffer across tokens: Completed GPT generation result-buffer reuse. The promoted path removes 1638400 B/op and 8 allocs/op at GPT-2-small maxNew=8 with median 1.004 throughput, exact token/cache parity, green local gates, and perfscan issue #895 capturing the generalized loop-use-site opportunity.
- T-01M0T2W1RQE8NBV10BEFP591PH Implement and validate shared Decoder generation logits reuse: Implemented in 9407cc45. Decoder.Generate now allocates one local Vocab F32 slice for host sampling and reuses StepNLastInto/StepInto; the device stepInto branch is unchanged. M2 Vocab 32000 maxNew 8 measured 4 versus 12 allocs/op and a nominal exact 1048576 allocator-byte saving. Seven alternating 50-pair ratios were 1.010, 1.013, 1.053, 1.011, 1.007, 1.010, and 0.974, median 1.010. Exact tokens, [body truncated at tombstone retention cap]
- ADR-01M0T2VZJVER0T9BKZZM90YDKD Keep shared-decoder generation reuse local to the host-sampling path: Adopted: caller-owned local reuse is allocation-efficient without adding decoder-global result state. The existing device-resident sampler path remains separate and unchanged; the historical allocation control exists only for same-binary attribution.

## METAL-F16-KV-CACHE-001
The Metal quant decoder SHALL expose NewQuantF16KV with retained K/V storage at exactly 2 bytes per element while NewQuant retains f32 storage and all non-Metal constructors remain unchanged.

## METAL-ROPE-F16KV-SCOPE-001
WHEN the decoder path is not Metal f16-KV single-token separate-QKV full-RoPE with dk equal to 64, the decoder selector SHALL execute the established RoPE and paired cache-copy chain and issue zero fused append dispatches.

## METAL-ROPE-F16KV-COUNT-001
WHEN the trained 22-layer TinyLlama decode is profiled with fusion enabled, the Metal decoder SHALL replace 20 RoPE and 10 paired-copy events with 10 fused events while preserving 12 grouped-layer events.

## METAL-ROPE-PAIR-F16KV-SCOPE-001
WHEN the path is not Metal f16-KV single-token grouped-QKV full-RoPE with dk equal to 64, the decoder selector SHALL execute rope_pair plus paired append and issue 0 grouped fused dispatches.

## METAL-ROPE-F16KV-COMBINED-COUNT-001
WHEN trained TinyLlama decode is profiled with both fusions enabled, the Metal decoder SHALL replace 54 RoPE and copy events with 10 separate and 12 grouped fused events.

## METAL-F16KV-SPLITK-FUSED-E2E-001
WHEN 3 valid token-interleaved M2 TinyLlama campaigns measure f16-KV decode at contexts 512 and 1536, the f16-KV fused split-K promotion gate SHALL require a median paired speedup of at least 1.01 times in every cell and campaign.

## METAL-GPT-QKV-OWNERSHIP-001 {applies: go:llamagpu.newGPTDecoder}
The Metal GPT decoder SHALL store exactly one resident fused QKV weight per block and bound grouped-output scratch to min(context, 63) times 3 times model width floats.

Rationale: Prevent a decode optimization from duplicating approximately 17 percent of GPT-2-small F32 model weights.

## METAL-GPT-QKV-DECODE-001 {applies: go:llamagpu.GPTDecoder.Step}
WHEN single-token GPTDecoder.Step executes on Metal, the decoder recorder SHALL issue 1 grouped QKV matrix multiplication, 2 direct K/V cache blits, and 0 split Q, K, or V projection matrix multiplications per block.

## METAL-GPT-QKV-PREFILL-001 {applies: go:llamagpu.mRec.F32QKVBands,go:metal.Recorder.MatMulStridedB}
WHEN GPT prefill executes on Metal with 64 or more rows, the decoder recorder SHALL issue 3 strided views of the one resident grouped QKV weight and copy 0 weight bytes.

## PORTABLE-GPT-QKV-STORAGE-001 {applies: go:llamagpu.newGPTDecoder}
WHILE a backend has not opted into fused F32 QKV recording, the GPT decoder SHALL retain 3 split QKV weights, allocate 0 grouped QKV weights, and allocate 0 grouped-output scratch floats.

## GPT-RESIDUAL-EPILOGUE-001 {applies: go:llamagpu.GPTDecoder.recordAttentionResidual,go:llamagpu.GPTDecoder.recordFFNResidual}
WHEN a GPT attention-output or FFN-down projection updates the running residual, the decoder recorder SHALL use the projection recordAdd epilogue and issue 0 standalone residual Binary-add dispatches.

## M2-GPT-DISPATCH-FUSION-PERF-001 {applies: go:llamagpu_test.BenchmarkGPTDecodeStepMetal,go:llamagpu_test.TestGPT2ScalePipeline}
WHEN a GPT dispatch-fusion slice is promoted on M2 Pro at GPT-2-small geometry, the benchmark gate SHALL require at least 1.03 times paired median speedup, 21 of 21 non-regressing pairs, and exact greedy-token equality over 256 generated tokens.

## GPT-RESIDUAL-DEAD-SCRATCH-001 {applies: go:llamagpu.newGPTDecoder,go:llamagpu.GPTDecoder.recordAttentionResidual,go:llamagpu.GPTDecoder.recordFFNResidual}
WHEN F32 GPT residual projections use recordAdd, the GPT decoder constructor SHALL allocate 0 attention-output scratch floats and 0 FFN-output scratch floats.

## M2-GPT-DISPATCH-CUMULATIVE-PERF-001 {applies: go:llamagpu_test.BenchmarkGPTDecodeStepMetal}
WHEN the complete GPT dispatch-fusion design is promoted on M2 Pro at GPT-2-small geometry, the public-Step benchmark gate SHALL require at least 1.20 times paired median speedup, 7 of 7 wins, and no allocation increase across an order-alternated campaign.

## METAL-GPT-BIAS-GELU-STRUCTURE-002 {applies: go:llamagpu.GPTDecoder.recordBiasGELU,go:llamagpu.TestMetalGPTBiasGELUProfileProvesFusedActivation}
WHEN the enabled Metal GPT FFN activation executes, the decoder SHALL record exactly one bounded BiasGELU dispatch and zero split AddBias or unary GELU dispatches for that activation.

Rationale: Remove one dispatch and context-capacity work amplification from Metal GPT.

## GPT-BIAS-GELU-FALLBACK-002 {applies: go:llamagpu.GPTDecoder.recordBiasGELU,go:llamagpu.TestMetalGPTBiasGELUStepAndStepNMatchSplitControl}
WHILE the bounded BiasGELU capability is unavailable or disabled, the decoder SHALL retain the established AddBias followed by exact unary GELU activation chain.

Rationale: Preserve portable CUDA, Vulkan, CPU, and same-binary control behavior.

## DECODER-LOGITS-RESIDENCY-001 {applies: go:llamagpu.Decoder.allocScratch,go:llamagpu.newGPTDecoder,go:llamagpu.TestGPTDecoderLogitsResidencyGrowthAndRelease,go:llamagpu.TestDecoderLogitsResidencyAndEagerControl}
WHEN GPTDecoder or Decoder is constructed with standard backend operations, the decoder SHALL retain exactly Vocab F32 logits elements for Step and StepNLast and retain 0 multi-row logits elements.

Rationale: Make dominant decoder residency scale with active output rows rather than maximum context.

## DECODER-FULL-STEPN-LOGITS-001 {applies: go:llamagpu.logitsForRows,go:llamagpu.GPTDecoder.gptStepN,go:llamagpu.TestGPTDecoderLogitsResidencyGrowthAndRelease}
WHEN full StepN requests more logits rows than the resident buffer holds, the decoder SHALL allocate exactly requested rows times Vocab F32 overflow elements, reuse that buffer for every smaller request, and grow only for a larger request.

Rationale: Preserve full StepN semantics without per-call allocation churn or lifetime maximum-context residency.

## DECODER-FULL-LOGITS-LIFETIME-001 {applies: go:llamagpu.growBuffer.ensure,go:llamagpu.Decoder.Release,go:llamagpu.GPTDecoder.Release,go:llamagpu.TestGrowBufferReleasesBeforeFailedReplacement,go:llamagpu.TestGPTDecoderLogitsResidencyGrowthAndRelease}
WHEN the full-StepN overflow buffer grows or its decoder is released, the decoder SHALL release the previously owned overflow buffer exactly once and retain 0 stale overflow buffer references.

Rationale: Keep lazy residency bounded and release-safe on every backend.

## GPT2-LOGITS-RESIDENCY-PERF-001 {applies: go:llamagpu.BenchmarkGPTDecoderLogitsResidency}
WHEN the same-binary GPT-2-small constructor benchmark compares lazy residency with eager control, the promotion gate SHALL require at least 200000000 fewer B/op and 10 times lower ns/op while public Step and StepNLast retain at least 0.97 times throughput.

Rationale: Validate memory leverage and prevent moving allocation cost into dominant inference paths.

## GPT-ACTIVATION-RESIDENCY-001 {applies: go:llamagpu.newGPTDecoder,go:llamagpu.TestGPTDecoderScratchResidencyGrowthAndRelease}
WHEN GPTDecoder is constructed with standard backend operations, the decoder SHALL retain exactly 1 row of every activation workspace buffer and 0 multi-row activation workspace buffers.

Rationale: Make dominant decode residency scale with active rows instead of maximum context.

## GPT-FULL-WORKSPACE-GROWTH-001 {applies: go:llamagpu.GPTDecoder.scratchForRows,go:llamagpu.GPTDecoder.gptStepN,go:llamagpu.TestGPTDecoderScratchResidencyGrowthAndRelease}
WHEN StepN or StepNLast requests more activation rows than the resident workspace holds, the GPTDecoder SHALL allocate 1 grouped workspace at requested rows, reuse it for smaller requests, and grow only for larger requests.

Rationale: Preserve batched semantics without per-call churn or maximum-context lifetime residency.

## GPT-FULL-WORKSPACE-LIFETIME-001 {applies: go:llamagpu.GPTDecoder.newScratch,go:llamagpu.gptScratch.release,go:llamagpu.GPTDecoder.Release,go:llamagpu.TestGPTScratchPartialAllocationFailureReleasesGeneration,go:llamagpu.TestGPTDecoderScratchResidencyGrowthAndRelease}
WHEN grouped GPT workspace growth fails or its decoder is released, the GPTDecoder SHALL release each prior or partial buffer exactly once and retain 0 stale grouped-workspace references.

Rationale: Keep grouped workspace ownership transactional and backend-independent.

## GPT2-ACTIVATION-RESIDENCY-PERF-001 {applies: go:llamagpu.BenchmarkGPTDecoderScratchResidency}
WHEN the same-binary GPT-2-small activation-residency benchmark compares lazy and eager controls, the promotion gate SHALL require 34000000 fewer B/op, 10 times lower constructor ns/op, and 0.97 times Step and StepNLast throughput.

Rationale: Validate retained-memory leverage without moving cost into dominant inference paths.

## GPT-HIDDEN-WORKSPACE-READBACK-001 {applies: go:llamagpu.GPTDecoder.StepHidden,go:llamagpu.GPTDecoder.StepNHidden}
WHEN GPT StepHidden or StepNHidden completes, the hidden readback SHALL download exactly 1 or len(tokens) final rows from the corresponding selected activation workspace.

Rationale: Preserve Medusa hidden-state semantics after activation workspace right-sizing.

## DECODER-F32-RESIDUAL-SCRATCH-001 {applies: go:llamagpu.Decoder.allocResidualScratch,go:llamagpu.TestDecoderResidualScratchReachability}
WHEN a pre-norm non-MoE Decoder uses only F32 residual projections, the constructor SHALL retain exactly 0 ao elements and 0 mo elements.

Rationale: F32 recordAdd writes directly into the residual and ignores projection scratch.

## DECODER-REQUIRED-RESIDUAL-SCRATCH-001 {applies: go:llamagpu.Decoder.allocResidualScratch,go:llamagpu.TestDecoderResidualScratchReachability,go:llamagpu.TestDecoderScratchOptionalPathShapes}
WHEN a Decoder has quantized weights, post-norm, or sandwich residuals, the constructor SHALL retain 2 resident residual scratch buffers with exactly Dim elements each and materialize Ctx-sized residual scratch only inside an eager control or a selected Ctx-row high-water workspace.

Rationale: These paths use projection scratch for fallback accumulation or output normalization.

## DECODER-MOE-RESIDUAL-SCRATCH-001 {applies: go:llamagpu.Decoder.allocResidualScratch,go:llamagpu.TestDecoderResidualScratchReachability,go:llamagpu.TestDecoderScratchOptionalPathShapes}
WHEN an F32 pre-norm Decoder enables MoE without another scratch requirement, the constructor SHALL retain exactly 0 resident ao elements and exactly Dim resident mo elements.

Rationale: MoE accumulates each expert output through mo while F32 attention residuals need no ao scratch.

## TINYLLAMA-RESIDUAL-SCRATCH-PERF-001 {applies: go:llamagpu.BenchmarkDecoderResidualScratchResidency,go:llamagpu.BenchmarkLlamaDecodeStepMetal,go:llamagpu.BenchmarkLlamaPrefillLastMetal}
WHEN the same-binary TinyLlama residual-scratch benchmark compares lazy and eager controls, the promotion gate SHALL require 33000000 fewer B/op, 10 times lower focused ns/op, and 0.97 times Step and StepNLast throughput.

Rationale: Validate retained-memory leverage without moving work into inference.

## DECODER-ACTIVATION-RESIDENCY-001 {applies: go:llamagpu.Decoder.allocScratch,go:llamagpu.Decoder.makeScratch,go:llamagpu.TestDecoderScratchResidencyGrowthAndRelease}
WHEN constructed with standard backend operations, the shared Decoder SHALL retain exactly 1 row of every common activation workspace buffer and 0 multi-row common activation workspace buffers.

Rationale: Single-token decode is the steady-state path; context-sized transient storage has no live consumer before prefill.

## DECODER-FULL-WORKSPACE-GROWTH-001 {applies: go:llamagpu.Decoder.scratchForRows,go:llamagpu.TestDecoderScratchResidencyGrowthAndRelease}
WHEN StepN or StepNLast requests more activation rows than the resident workspace holds, the shared Decoder SHALL allocate 1 grouped workspace at exactly the requested rows, reuse it for every smaller request, and grow only for a larger request.

## DECODER-FULL-WORKSPACE-LIFETIME-001 {applies: go:llamagpu.decoderScratch.release,go:llamagpu.Decoder.newScratch,go:llamagpu.Decoder.Release,go:llamagpu.TestDecoderScratchPartialAllocationFailureReleasesGeneration}
WHEN grouped workspace growth fails or the Decoder is released, the shared Decoder SHALL release each prior or partial workspace buffer exactly once and retain 0 stale grouped-workspace references.

## DECODER-HIDDEN-WORKSPACE-READBACK-001 {applies: go:llamagpu.Decoder.StepNHidden,go:llamagpu_test.TestMedusaGenerateLlamaAllRejectIsGreedy}
WHEN StepHidden or StepNHidden completes, the shared Decoder SHALL download exactly 1 or len(tokens) final hidden rows from the corresponding selected activation workspace.

## TINYLLAMA-ACTIVATION-RESIDENCY-PERF-001 {applies: go:llamagpu.BenchmarkDecoderScratchResidency,go:llamagpu.BenchmarkLlamaDecodeStepMetal,go:llamagpu.BenchmarkLlamaPrefillLastMetal}
WHEN the same-binary TinyLlama-class activation residency benchmark compares lazy and eager controls on M2, the promotion gate SHALL require at least 150000000 fewer B/op, 10 times lower constructor ns/op, and 0.97 times public Step and StepNLast throughput.

## ROPE-PAIR-EXACT-STRIDE-STORAGE-001 {applies: go:metal.Recorder.RoPEPair,go:vulkan.Recorder.RoPEPair,go:llamagpu_test.TestDecoderMatchesReference,go:llamagpu_test.TestStepNMatchesSequentialSteps}
WHEN a fused QKV buffer has exactly seq times stride elements and both band ends are within stride, the Metal and Vulkan RoPEPair recorders SHALL accept it with exactly 0 offset-padding elements.

## DECODER-STEP-INTO-SEMANTICS-001 {applies: go:llamagpu.Decoder.StepInto,go:llamagpu.Decoder.Step,go:llamagpu_test.TestDecoderStepIntoMatchesStepAndGuardsLength}
WHEN the destination length equals Vocab, the Decoder.StepInto SHALL advance exactly 1 token and write exactly Vocab logits matching Step.

## DECODER-STEP-INTO-LENGTH-GUARD-001 {applies: go:llamagpu.Decoder.StepInto,go:llamagpu_test.TestDecoderStepIntoMatchesStepAndGuardsLength}
WHEN the destination length differs from Vocab, the Decoder.StepInto SHALL return an error and mutate exactly 0 cache rows and 0 recurrent states.

## DECODER-EMBED-STAGING-001 {applies: go:llamagpu.Decoder.gatherEmbedInto,go:llamagpu.embedRowInto,go:llamagpu.TestEmbedRowIntoAllocatesZeroAndMatches}
WHEN single-token stepping gathers a token embedding, the shared Decoder SHALL reuse exactly 1 Dim-element host row and allocate 0 per-token embedding objects.

## M2-DECODER-STEP-INTO-PERF-001 {applies: go:llamagpu.BenchmarkLlamaDecodeStepIntoMetal,go:llamagpu.BenchmarkLlamaDecodeStepMetal}
WHEN StepInto is benchmarked against Step on M2 at the tracked Llama boundary, the promotion gate SHALL require 0 allocations per operation, at least 8000 fewer B/op, and at least 0.97 times Step throughput.

## DECODER-METAL-RECORDER-POOL-SAFETY-001 {applies: go:llamagpu.mRecPool.acquire,go:llamagpu.pooledMRec.Free}
WHEN a pooled Decoder recorder is freed, the Metal Decoder adapter SHALL release exactly 1 native Metal command buffer before returning its Go wrapper to the 2-slot pool.

## DECODER-STEPN-INTO-SEMANTICS-001-001 {applies: go:llamagpu.Decoder.StepNInto,go:llamagpu.Decoder.stepNInto}
WHEN the destination length equals len(tokens) times Vocab, the Decoder.StepNInto SHALL advance exactly len(tokens) tokens and write exactly len(tokens) times Vocab logits matching StepN.

## DECODER-STEPN-LAST-INTO-SEMANTICS-001-001 {applies: go:llamagpu.Decoder.StepNLastInto,go:llamagpu.Decoder.stepNInto}
WHEN the destination length equals Vocab, the Decoder.StepNLastInto SHALL advance exactly len(tokens) tokens and write exactly Vocab logits matching StepNLast.

## DECODER-PREFILL-INTO-LENGTH-GUARD-001-001 {applies: go:llamagpu.Decoder.StepNInto,go:llamagpu.Decoder.StepNLastInto}
WHEN the destination length differs from the method requirement, the Decoder.StepNInto or Decoder.StepNLastInto SHALL return an error and mutate exactly 0 cache rows and 0 recurrent states.

## DECODER-PREFILL-EMBED-STAGING-001-001 {applies: go:llamagpu.Decoder.batchEmbedHost,go:llamagpu.Decoder.stepNInto}
WHEN StepN or StepNLast gathers k token embeddings, the shared Decoder SHALL retain exactly 1 host staging slice at the high-water size k times Dim, reuse it for every smaller request, and grow only for a larger request.

## DECODER-PREFILL-STAGING-LIFETIME-001-001 {applies: go:llamagpu.Decoder.Release,go:llamagpu.Decoder.batchEmbedHost}
WHEN Release completes, the shared Decoder SHALL retain exactly 0 high-water embedding staging elements.

## M2-DECODER-STEPN-INTO-PERF-001-001 {applies: go:llamagpu.BenchmarkLlamaPrefillLastIntoMetal,go:llamagpu.BenchmarkLlamaPrefillHostStagingMetal,go:llamagpu.BenchmarkLlamaPrefillHostStagingPairedMetal}
WHEN StepNLastInto is benchmarked against StepNLast on M2 with 16 tokens and Dim 512, the Decoder prefill promotion gate SHALL require 0 StepNLastInto allocations, 32768 fewer StepNLast bytes, and at least 0.97 times baseline throughput.

## GPT-STEP-INTO-SEMANTICS-001-001 {applies: go:llamagpu.GPTDecoder.StepInto,go:llamagpu.GPTDecoder.stepInto}
WHEN the destination length equals Vocab, the GPTDecoder.StepInto SHALL advance exactly 1 token and write exactly Vocab logits matching Step.

## GPT-STEPN-INTO-SEMANTICS-001-001 {applies: go:llamagpu.GPTDecoder.StepNInto,go:llamagpu.GPTDecoder.gptStepNInto}
WHEN the destination length equals len(tokens) times Vocab, the GPTDecoder.StepNInto SHALL advance exactly len(tokens) tokens and write exactly len(tokens) times Vocab logits matching StepN.

## GPT-STEPN-LAST-INTO-SEMANTICS-001-001 {applies: go:llamagpu.GPTDecoder.StepNLastInto,go:llamagpu.GPTDecoder.gptStepNInto}
WHEN the destination length equals Vocab, the GPTDecoder.StepNLastInto SHALL advance exactly len(tokens) tokens and write exactly Vocab logits matching StepNLast.

## GPT-INTO-LENGTH-GUARD-001-001 {applies: go:llamagpu.GPTDecoder.StepInto,go:llamagpu.GPTDecoder.StepNInto,go:llamagpu.GPTDecoder.StepNLastInto}
WHEN a destination length differs from its method requirement, the GPTDecoder Into methods SHALL return an error and mutate exactly 0 cache rows.

## GPT-EMBED-STAGING-001-001 {applies: go:llamagpu.GPTDecoder.gatherEmbedInto,go:llamagpu.GPTDecoder.batchEmbedHost,go:llamagpu.addEmbedRowInto,go:llamagpu.GPTDecoder.stepInto,go:llamagpu.GPTDecoder.gptStepNInto}
WHEN Step or prefill gathers token and positional embeddings, the GPTDecoder SHALL reuse exactly 1 Dim host row or 1 exact high-water batch slice and allocate 0 embedding objects.

## GPT-EMBED-STAGING-LIFETIME-001-001 {applies: go:llamagpu.GPTDecoder.Release,go:llamagpu.GPTDecoder.batchEmbedHost}
WHEN Release completes, the GPTDecoder SHALL retain exactly 0 host embedding staging elements.

## GPT-METAL-RECORDER-POOL-SAFETY-001-001 {applies: go:llamagpu.NewGPT,go:llamagpu.metalGPTOps,go:llamagpu.pooledMRec.Free}
WHEN a pooled recorder is freed, the Metal GPT decoder adapter SHALL release exactly 1 native command buffer before returning its Go wrapper to the 2-slot pool.

## M2-GPT-INTO-PERF-001-001 {applies: go:llamagpu_test.BenchmarkGPTDecodeStepIntoMetal,go:llamagpu_test.BenchmarkGPTPrefillLastIntoMetal,go:llamagpu.BenchmarkGPTDecodeBoundaryPairedMetal,go:llamagpu.BenchmarkGPTPrefillBoundaryPairedMetal}
WHEN GPT-2-small StepInto and 16-token StepNLastInto are benchmarked on M2, the GPT caller-buffer promotion gate SHALL require 0 allocations and at least 210000 and 352000 fewer bytes respectively.

## M2-GPT-WRAPPER-THROUGHPUT-001-001 {applies: go:llamagpu_test.BenchmarkGPTDecodeStepMetal,go:llamagpu_test.BenchmarkGPTPrefillLastMetal,go:llamagpu.BenchmarkGPTDecodeBoundaryPairedMetal,go:llamagpu.BenchmarkGPTPrefillBoundaryPairedMetal}
WHEN GPT-2-small Step and 16-token StepNLast are benchmarked on M2, the GPT compatibility-wrapper promotion gate SHALL require at most 204804 B/op, exactly 1 allocation, and at least 0.97 times baseline throughput for both boundaries.

## GPT-GENERATE-LOGITS-REUSE-001-001 {applies: go:llamagpu.GPTDecoder.Generate}
WHEN Generate emits N tokens through host sampling, the GPTDecoder SHALL allocate exactly 1 Vocab F32 logits slice and reuse it for prefill and every decode step.

## GPT-GENERATE-CACHE-PARITY-001-001 {applies: go:llamagpu.GPTDecoder.Generate}
WHEN optimized Generate returns N tokens after a prompt of P tokens, the GPTDecoder SHALL retain exactly 1 populated cache row per prompt or generated token including 1 row for the final generated token.

## M2-GPT-GENERATE-ALLOCATION-PERF-001-001 {applies: go:llamagpu.BenchmarkGPTGenerateAllocationsMetal,go:llamagpu.BenchmarkGPTGeneratePairedMetal}
WHEN the generation reuse slice is promoted, the GPT-2-small M2 maxNew 8 benchmark gate SHALL require 1638400 fewer B/op, 8 fewer allocs/op, and at least 0.97 times historical-control throughput.

## DECODER-GENERATE-LOGITS-REUSE-001
WHEN Decoder.Generate emits N tokens through host sampling, the shared Decoder SHALL allocate exactly 1 Vocab F32 logits slice and reuse it for prefill and every decode step.

## DECODER-GENERATE-CACHE-PARITY-001
WHEN optimized Decoder.Generate returns N tokens after a prompt of P tokens, the shared Decoder SHALL retain exactly 1 populated cache row per prompt or generated token including 1 row for the final generated token.

## DECODER-GENERATE-DEVICE-SAMPLING-001
WHEN Decoder.Generate selects the eligible device-resident Top-K or pure Top-P sampling path, the shared Decoder SHALL perform exactly 0 full-Vocab decode-logit host downloads after the first sampled token.

## M2-DECODER-GENERATE-ALLOCATION-PERF-001
WHEN the generation reuse slice is promoted on M2 at Vocab 32000 and maxNew 8, the shared Decoder benchmark gate SHALL require at least 1048576 fewer B/op, at least 8 fewer allocs/op, and at least 0.97 times historical-control throughput.
