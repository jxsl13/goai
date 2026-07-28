---
schema: v1
---

## T-01KYJQZE63ERCSRSWEFEJAFE69 Rewrite keepSinkRecent as typed row copies
kind: task
state: done
created: 2026-07-27

HIGHEST-IMPACT PER-ELEMENT DEFECT FOUND IN THE PACKAGE.

MEASURED BASELINE on this host (M2 Pro, darwin/arm64): BenchmarkEmbedOnePerElement 13,903 ns vs BenchmarkEmbedOneRowCopy 1,778 ns at vocab 32k / dim 2048 — 7.82x, i.e. 6.79 ns/element for AtF64/SetF64 against 0.87 ns for a typed copy, with identical B/op and allocs/op on both sides, so the entire gap is dispatch.

SITE: nlp/streaming.go:36-53 keepSinkRecent, called from nlp/streaming.go:86-87 inside Llama.StreamStep's per-layer attendFunc.

WHY HOT: StreamGenerate -> StreamStep -> blockStack closure -> per layer, per token, TWICE (K and V). Once the stream exceeds sinks+window — the steady state that is the entire point of StreamingLLM — the rows <= sinks+window early return at :38 never fires again, so every call runs the full copy. At sinks=4, window=1020, kvWidth=1024, 32 layers that is roughly 67 million per-element dispatches per token.

DEFECT: :42-51 is a nested AtF64/SetF64 loop over (sinks+window) x d elements. AtF64 is variadic plus a flatOffset stride loop plus an interface dispatch into storage. But the two retained regions — rows [0, sinks) and rows [rows-window, rows) — are CONTIGUOUS ROW BLOCKS, so the whole function is two copy() calls. Worse, :86-87 first runs concatRows (a full realloc and copy of the bounded cache) and THEN keepSinkRecent copies it again: four full cache traversals per layer per token, one of them per-element.

FIX: rewrite on the model of copyRows (nlp/rowbuf.go:165-174) and concatRows' own typed arm (nlp/decode.go:56-69) — tc := t.Contiguous(), then per dtype (F32/F64/F16/BF16 via Storage().U16()) two copy() calls: dst[0:sinks*d] from src[0:sinks*d], and dst[sinks*d:] from src[(rows-window)*d:]. KEEP the existing per-element loop as the default arm for exotic dtypes. Second, independent step, land separately: replace concatRows at :86-87 with a ring-buffer append so the concat disappears too.

VALIDATION GATE (benchmark only): none exists — nlp/streaming_test.go:31 and example_methods_test.go:132 use sinks=2/window=6, far too small to expose this. Write BenchmarkStreamStepBounded (Llama with Vocab 1000, Dim 512, Heads 8, Layers 4, Hidden 1024; cache pre-filled PAST the bound, then StreamStep in the loop) plus a BenchmarkKeepSinkRecent micro at d=2048, sinks=4, window=512 mirroring the existing ConcatRows/RowBuf A/B pair so the isolated change is measurable.

EXPECTED: 5-8x on keepSinkRecent itself, bounded by the measured 7.82x embed ratio; at realistic width it should be the dominant term of StreamStep. High confidence on the ratio, MEDIUM on its share of StreamStep because no StreamStep benchmark exists yet — measure before implementing, do not assume the share.

BIT-IDENTITY BAR: none — SetF64(AtF64(...)) on F32 storage is float32(float64(x)), an exact round trip, and identity on F64; a typed copy moves the same bits. No reduction reordering, no accumulation-width change. This is the same argument concatRows already relies on at nlp/decode.go:54-55. Guard with a table test asserting the new output equals the old loop's output BITWISE across F32/F64 and with rows both above and below the bound.

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

## T-01KYMEV0VSFFB9VEXJPZRHKJV2 Build a synthetic T5Decoder so its KV cache can be benchmarked, then move it to the row buffer
kind: task
state: draft
created: 2026-07-28

BLOCKED ON A MISSING TEST FIXTURE, NOT ON THE OPTIMIZATION. The two remaining PS2006 sites are nlp/t5_decoder.go:435-436, where the decoder self-attention cache is reassigned to concatRows of itself per block per token — the identical quadratic shape already fixed in GPT (2.21 GB -> 159 MB, 13.9x) and CLA (1.16 GB -> 140 MB, 8.3x).\n\nTHE EDIT ITSELF IS KNOWN AND WAS ALREADY WRITTEN ONCE: add bufs kvBufs to T5DecoderCache and replace the whole if/else with cache.selfK[l], cache.selfV[l] = cache.bufs.appendKV(cache.selfK, cache.selfV, l, kt, vt). Note this also removes the nil special case, where the first token ADOPTED the kernel output kt/vt as the cache slot — the row buffer copies into its own backing instead, which is the same values and strictly less sharing. That edit was reverted unmerged because it could not be measured, and under the standing rule an optimization is validated solely by benchmark.\n\nWHAT IS ACTUALLY NEEDED FIRST: a synthetic T5Decoder constructor for tests. T5Decoder is today only constructible via T5DecoderFromHF from real safetensors, so there is no way to build a small one in a benchmark. It needs Config, Shared, RelBias (nn.T5RelativeBias), FinalNorm, LMHead and a slice of T5DecoderBlock, each with self-attention, cross-attention and FFN weights — roughly 20 tensors, randomly initialized from a fixed seed exactly as NewCLA and NewLlama do. Put it beside the existing fixtures rather than in the benchmark file, since parity tests would benefit from it too.\n\nTHEN benchmark T5Decoder.Generate(ctx, encoderOut, maxNew, startToken, eosToken, Greedy()) with a synthetic encoderOut, as a ConcatRows/RowBuf pair driven by the bench-only kvAppendViaConcat flag in nlp/rowbuf.go — that flag flips ONLY the append and holds every other instruction identical, which is what makes the A/B clean. Mirror benchCLADecode and benchLlamaGenerate in nlp/kvcache_perf_test.go.\n\nEXPECT THE MEMORY FIGURE, NOT THE WALL CLOCK, TO BE THE RESULT. Across both landed cases the byte reduction was 8-14x while wall clock moved only 1.22-1.32x; the original brief predicted 2-3x time by analogy to the Llama pair and was wrong twice. Also budget for benchmark noise: at -benchtime 3x the GPT pair straddled 1.16-1.30x and was unreportable; 5 iterations were needed before three reps agreed within 1%.\n\nALIASING AUDIT, already done for the package and recorded so it need not be redone: nothing in nlp/ writes into a cache tensor in place, which is the only pattern the view semantics break; retained views stay valid because appends only ever write row t and beyond. For T5 specifically, confirm the cross-attention cache (crossK/crossV, computed once from the encoder) is untouched by this change — it is not appended per token and must stay on its current path.
