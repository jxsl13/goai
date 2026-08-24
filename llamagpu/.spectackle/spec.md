---
schema: v1
---

## intent
- T-01KYJQ78FAFCGA1D639T76XZKC Compute the T5 relative-position bias row directly instead of rebuilding the full matrix per token: LANDED the nlp twin, benchmark-validated on M2 Pro darwin/arm64 go1.26.5. Per call, and the CURVE is the evidence — matmul form roughly quadruples per doubling of pos, gather roughly doubles: pos=32 167,132 -> 1,288 ns (130x, 381,154 -> 2,400 B); pos=128 784,268 -> 3,000 ns (261x, 5,507,630 -> 6,112 B); pos=512 7,080,952 -> 9,929 ns (713x, 86,766,729 -> 19,680 B). End to end, 3 reps of -benchtime 5x, medians: BenchmarkT5Decode500 2,679 -> 556 ms (4.82x) and 13,865,254,414 -> 124,827,840 B/op (111x), allocs 490,802 -> 350,477. The task predicted 10x/100x/1000x at those positions; actual was 130x/261x/713x — same order, higher at small pos and lower at large.\n\nThis also settles why the preceding T5 KV-cache change (T-01KYMEV0VSFFB) measured a marginal 1.15x: this dominated. With it gone, a 500-token T5 decode allocates 125 MB rather than 15.9 GB.\n\nBIT-IDENTITY PROVEN, EXCEPTIONS NAMED: the old value came through a one-hot matmul, a sum of numBuckets products of which all but one were 0*Table[j][h]. For finite tables that sum IS Table[b][h], so the gather is bit-identical — gated tolerance-0 against a frozen oracle over a table filled with a WIDE EXPONENT SPREAD, because the constructor zero-initializes the table and a zero table would have made the gate pass regardless. It differs for +-Inf/NaN (0*Inf = NaN) and for a stored -0 (0 + -0 = +0); both indicate a broken checkpoint and neither changes a downstream sum. The stronger bar the task set is also met: greedy decoding gives token-for-token identical ids over 128 steps.\n\nMETHOD NOTE worth reusing: both gates run the identical decode with ONLY the bias computation swapped, via a test-only t5BiasViaMatmul flag mirroring kvAppendViaConcat. Without it the token test would have compared a run against itself and proved nothing but determinism — it was written that way first and corrected.\n\nGENERALIZED as perfscan PS2007 build-nxn-use-one-row. PRECISION TOOK THREE PASSES and both rejected cuts are kept as fixtures: asking only whether the driving identifier appears in some index anywhere in the function flagged a.AtF64(j, j) — a diagonal element READ, not a builder — and a square attention mask consumed whole. The rule now requires the result OR A VALUE DERIVED FROM IT to be indexed by the driving position, and requires that position not to be a loop bound (when it bounds a loop the object is walked in full and the identifier appears only as a stride). Tree-wide 6 -> 2 findings, both genuine.\n\nNOT DONE, and it is the task original primary target: llamagpu/t5_decoder.go:123, the byte-identical GPU twin, plus llamagpu/t5.go. GPUT5Decoder is reachable only through the CUDA constructors and MHABias is a metal stub, so it cannot be benchmarked on this host; under the standing rule an unmeasurable change does not ship. The fix is mechanical and identical, and PS2007 will keep pointing at it. The task also proposed a persistent per-session row extended by one entry per step (O(heads) per token instead of O(pos*heads)) and hoisting the per-token NewContext; neither was needed to reach 713x and both remain available.
- R-01KZ14QMNHFDR89C3KK4WX1JJJ Round T1059: headArgmax row-order accumulation -52%, and a pre-existing F32-on-f64 panic found in backend/cpu: Consumed: headArgmax shipped at -52.2 percent with two new benchmarks and a tie-breaking gate that took two attempts to become non-vacuous. Two things recorded rather than fixed: a pre-existing panic in backend/cpu mhaMaskedKernelCPU (Storage().F32() on f64, reached with F64 tensors, invisible to the -short CI lane) which needs its own change; and the misattribution that a panic stack naming an ac [body truncated at tombstone retention cap]
- T-01KYJPSEWVFRSRSMX2VEV7AY95 Measure and fuse GPTDecoder single-token dispatch on M2: validated pass by codex-m2-gptdecoder-validator-20260824 diff 1c44a6e6b4be — Promoted M2 Metal GPT dispatch fusion. Residual recordAdd won 21/21 pairs at 1.076643x median; grouped QKV decode won 21/21 at 1.113765x median. The clean design versus the frozen pre-fusion public-Step binary won 7/7 order-alternated pairs at 1.225737x median and 1.204752x floor with unchanged allocations. Metal owns o [body truncated at tombstone retention cap]

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

## DECODER-REQUIRED-RESIDUAL-SCRATCH-001 {applies: go:llamagpu.Decoder.allocResidualScratch,go:llamagpu.TestDecoderResidualScratchReachability}
WHEN a Decoder has quantized weights, post-norm, or sandwich residuals, the constructor SHALL retain 2 resident residual scratch buffers with exactly Dim elements each and materialize Ctx-sized residual scratch only inside an eager control or a selected Ctx-row high-water workspace.

Rationale: These paths use projection scratch for fallback accumulation or output normalization.

## DECODER-MOE-RESIDUAL-SCRATCH-001 {applies: go:llamagpu.Decoder.allocResidualScratch,go:llamagpu.TestDecoderResidualScratchReachability}
WHEN an F32 pre-norm Decoder enables MoE without another scratch requirement, the constructor SHALL retain exactly 0 resident ao elements and exactly Dim resident mo elements.

Rationale: MoE accumulates each expert output through mo while F32 attention residuals need no ao scratch.

## TINYLLAMA-RESIDUAL-SCRATCH-PERF-001 {applies: go:llamagpu.BenchmarkDecoderResidualScratchResidency,go:llamagpu_test.BenchmarkLlamaDecodeStepMetal,go:llamagpu_test.BenchmarkLlamaPrefillLastMetal}
WHEN the same-binary TinyLlama residual-scratch benchmark compares lazy and eager controls, the promotion gate SHALL require 33000000 fewer B/op, 10 times lower focused ns/op, and 0.97 times Step and StepNLast throughput.

Rationale: Validate retained-memory leverage without moving work into inference.

## DECODER-ACTIVATION-RESIDENCY-001
WHEN constructed with standard backend operations, the shared Decoder SHALL retain exactly 1 row of every common activation workspace buffer and 0 multi-row common activation workspace buffers.

Rationale: Single-token decode is the steady-state path; context-sized transient storage has no live consumer before prefill.

## DECODER-FULL-WORKSPACE-GROWTH-001
WHEN StepN or StepNLast requests more activation rows than the resident workspace holds, the shared Decoder SHALL allocate 1 grouped workspace at exactly the requested rows, reuse it for every smaller request, and grow only for a larger request.

## DECODER-FULL-WORKSPACE-LIFETIME-001
WHEN grouped workspace growth fails or the Decoder is released, the shared Decoder SHALL release each prior or partial workspace buffer exactly once and retain 0 stale grouped-workspace references.

## DECODER-HIDDEN-WORKSPACE-READBACK-001
WHEN StepHidden or StepNHidden completes, the shared Decoder SHALL download exactly 1 or len(tokens) final hidden rows from the corresponding selected activation workspace.

## TINYLLAMA-ACTIVATION-RESIDENCY-PERF-001
WHEN the same-binary TinyLlama-class activation residency benchmark compares lazy and eager controls on M2, the promotion gate SHALL require at least 150000000 fewer B/op, 10 times lower constructor ns/op, and 0.97 times public Step and StepNLast throughput.
