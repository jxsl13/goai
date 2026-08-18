---
schema: v1
---

## T-01KYJQZF10F20R8RMTNMG43PBH Finish the attrs-box hoist and slice pooling across the ~24 skipped decode paths
kind: task
state: draft
created: 2026-07-27

A previous sweep applied this to about 6 float decode paths and skipped roughly 24 others, INCLUDING EVERY QUANTIZED MODEL.

SITES, representative, all inside per-layer loops: nlp/quant_llama_decode.go:78,81,86; mixtral_decode.go:86,89,95; olmo2_decode.go:82,85,91; qwen2moe_decode.go:77,80,86; starcoder2_decode.go:78,81,87; stablelm_decode.go:80,83,89; nemotron_decode.go:80,83,89; gemma2_decode.go:145,148; gptneox_decode.go:87; phi_orig_decode.go:86; streaming.go:90,94,98; llama_layerskip_decode.go:80,86; plus quant_mixtral, quant_gemma, quant_gemma2, quant_cohere, quant_falcon, quant_gptneox, quant_olmo2, quant_nemotron, quant_stablelm, quant_starcoder2.

DEFECT, two stacked, BOTH CONFIRMED by go build -gcflags='github.com/jxsl13/goai/nlp=-m':
1. Attrs boxing. RoPEAttrs (backend/attrs.go:167) is about 88 bytes, AttnAttrs (:52) about 56. Escape analysis reports mixtral_decode.go:86:59, :89:59, :95:56 and the same at olmo2_decode.go:82,85,91 and quant_llama_decode.go:78,81 as escaping to heap. The values are layer-invariant (Base/Heads/KV/pos/scale) — exactly the condition the earlier hoist exploited at llama_decode.go:239-241.
2. THE SUBTLE VARIANT THE EARLIER PASS MISSED. quant_llama_decode.go:51 DOES hoist attn := backend.AttnAttrs{...} out of the loop — but as a CONCRETE STRUCT, so the interface conversion still happens at the call site inside the loop: escape analysis reports quant_llama_decode.go:86:39 attn escapes to heap. Hoisting the struct is not the fix; hoisting the BOX (backend.Attrs(...), as done at llama_decode.go:241) is. Same defect at gptneox_decode.go:54, phi_orig_decode.go:54, mpt_decode.go:65, streaming.go:83, t5_decoder.go:413.
3. Unpooled variadic input slices: these paths call exec1 (variadic) for RoPE and MHA instead of the pooled exec1a/exec3 (nlp/gpt.go:160,177). Escape analysis flags the argument slices at mixtral_decode.go:86,89,95, olmo2_decode.go:82,85,91, gptneox_decode.go:87. 81 unpooled exec1(...OpRoPE...) and 59 unpooled exec1(...OpMHA...) call sites remain package-wide.
Combined: about 6 avoidable heap allocations per layer per token relative to an already-fixed path.

FIX: mechanical per path — hoist backend.Attrs(...) BOXES above the layer loop, switch RoPE to exec1a and MHA to exec3. ONE CAVEAT that must not be skipped: at nlp/streaming.go:90 the q-RoPE PosOffset: n-1 reads n := cache.K[l].Shape()[0] PER LAYER. In practice every layer appends and bounds identically so n is uniform, but it is not invariant by construction — either compute n once before the loop and assert uniformity, or leave :90 alone and hoist only :94 and :98.

VALIDATION GATE (benchmark only): BenchmarkQuantLlamaGenerate500 (1,661,535,000 ns, 278,594 allocs) covers quant_llama_decode.go directly. BenchmarkFalconDecode and BenchmarkGemmaDecode are the templates written for exactly this fix — clone them per architecture. Historic recorded deltas for single-model hoists were -0.86% to -1.28% allocs, -4.42% for project pooling, -2.33% for RoPE/MHA pooling. DO THEM IN BATCHES WITH AN A/B PER BATCH — individual deltas sit near the noise floor, so a per-path A/B will not resolve them.

EXPECTED: about 1-3% allocs per path individually; roughly 15-25% of total allocs if propagated across all 24. Real-time effect is smaller than the alloc delta since these are small objects. High confidence in the mechanism (escape-analysis-confirmed), medium in the aggregate.

BIT-IDENTITY BAR: none — a pure allocation-lifetime change, identical values reaching identical kernels. The one behavioral guard is the existing one: exec1a/exec3 pool only when ctx.Recorder == nil, so a taped training context keeps the fresh-slice path. That check already lives inside the helpers and must not be bypassed.

PERFSCAN RULES REQUIRED, two. (i) Loop-invariant interface boxing: a CompositeLit, OR an Ident bound to one outside the loop, passed to a parameter of INTERFACE type at a call site inside a loop, where every field initializer is invariant with respect to the loop variable and the struct exceeds one word. It MUST match both the inline-literal form and the hoisted-concrete-struct form — the latter is precisely what the earlier pass missed, so a rule catching only the first would repeat the mistake. (ii) Unpooled variadic sibling: a call to a variadic helper f(..., xs ...T) with fixed arity n inside a loop, when a non-variadic fixed-arity sibling exists in the same package. Detectable from the signature set alone.

## T-01KYJQZFEHEFRA5QX6P8Z2W0NM Route QuantLlama.embedOne through the bulk embedRow helper
kind: task
state: draft
created: 2026-07-27

ONE LINE. Ranked LAST deliberately — it closes the final hole in an otherwise completed sweep, and the honest expectation is that the end-to-end A/B shows noise.

SITE: nlp/quant_llama_decode.go:19-29, the loop at :25-27 — for j := range d { x.SetF64(m.TokEmb.AtF64(token, j), 0, j) }.

WHY IT QUALIFIES: QuantLlama.Generate -> DecodeStep (:42) -> once per token over the full model Dim, and also once per PROMPT token, since QuantLlama.Generate (:145-152) has no batched prefill and steps the prompt through DecodeStep. 28 of the package's 29 embedOne implementations already call the bulk embedRow (nlp/embed_row.go:13); this is the only holdout. m.TokEmb is a plain dense F32 [vocab, dim] tensor (nlp/quant_llama.go:23, built by f32Clone at :58) — NOT a quantized tensor — so the bulk path applies directly with no unpacking concern.

FIX: return embedRow(m.TokEmb, token, d), nil, keeping the existing vocab bounds check at :20-22. embedRow already has an F32 arm and falls back to the per-element loop for exotic dtypes.

VALIDATION GATE (benchmark only): the micro A/B already exists and is already documented as this exact pattern — BenchmarkEmbedOnePerElement 13,903 ns versus BenchmarkEmbedOneRowCopy 1,778 ns (nlp/decode_perf_test.go:31,43; vocab 32k, dim 2048) = 7.82x with identical 16,584 B / 6 allocs on both sides, so the win is pure dispatch elimination. End to end: BenchmarkQuantLlamaGenerate500 (nlp/quant_decode_perf_test.go:15).

EXPECTED, stated honestly: about 12 us/token saved at dim 2048. At the benchmark's dim 256 that is under 0.1% of a 3.3 ms quantized token, so EXPECT THE E2E A/B TO SHOW NOISE. The real win is at production geometry (dim 4096+), and even there it is roughly 1-2% of a token. High confidence in the microbenchmark ratio AND high confidence that the e2e delta will be small. Do it because it is one line and closes a completed sweep, not because it is a large win — and do not let a noisy e2e result be read as the change being wrong.

BIT-IDENTITY BAR: none. Identical float64/float32 round-trip argument; embedRow's F32 arm is a bit-exact copy. This exact substitution is already pinned by the TestQuantLlamaDecodeMatchesForward-family equivalence tests on the other 28 paths.

## R-01KYMVHB75F3ETH2YCC3NVZGEQ DECLINED: hoisting the loop-invariant in quant_mamba2's SSD scan — QMatMul is 76% of the decode step
kind: research
state: draft
created: 2026-07-28

perfscan PS5003 flagged nlp/quant_mamba2.go:452 — `bi*(xc[hOff+j]*delta)` rebuilds a value that varies with the inner index but not the outer, N times per j. The pattern is REAL and the finding is correct: it is the exact shape whose fix in the float sibling nlp/mamba2_decode.go measured 1.08-1.10x on prefill, and nlp/mamba2.go already carries the hoist.

DECLINED ANYWAY, on measurement rather than shape. A CPU profile of QuantMamba2 DecodeStep (2 layers, d_model 256, 3000 iterations):
  gguf.QMatMul                 63.45% flat, 75.86% cumulative
  QuantMamba2Mixer.step         2.07% flat
The SSD scan does not surface at all. The hoist removes one of three multiplies in the update loop; against a step three-quarters spent in the quantized matmuls, the ceiling is about 1% and below the interleaving noise floor — unmeasurable here, so not shippable here.

THE ENCLOSING WORK DECIDES, not the code shape. Sixth validation of that heuristic. The same source expression is worth 1.10x in the float path, where no quantized matmul competes with it, and worth nothing in the quantized one.

WHERE THE LEVERAGE ACTUALLY WAS: the profile pointed at QMatMul, where Q4_0 decode ran slower than Q8_0 — a fused single-token path existed for Q8_0 alone. Fusing Q4_0/Q4_K/Q6_K measured 1.40x/1.40x/1.52x on the same benchmark. See R-01KYMVGRENEND. Profiling before optimizing turned a sub-1% candidate into a 1.4x one in the same function's caller.

STATUS: the PS5003 finding at that line stands and is NOT suppressed. It becomes shippable if the quantized matmuls ever stop dominating the step — recorded here so the next agent to see the finding reads the measurement instead of repeating it.
