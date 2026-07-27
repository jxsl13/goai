---
schema: v1
---

## T-01KYJQZE63ERCSRSWEFEJAFE69 Rewrite keepSinkRecent as typed row copies
kind: task
state: draft
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

## T-01KYJQZEK7E1RV0AH9K8MFT89M Propagate the rowBuf KV append to the GPT, CLA, T5 and streaming caches
kind: task
state: draft
created: 2026-07-27

The fix already exists, is already proven by an in-repo A/B, and simply never reached four of the caches.

MEASURED, in-repo, on this host: BenchmarkKVCacheGrowthConcatRows 62,444,861 ns / 1.076 GB / 3,076 allocs versus BenchmarkKVCacheGrowthRowBuf 616,583 ns / 10.3 MB / 1,066 allocs at width 2048, T=512 — 101x time, 104x bytes. End to end: BenchmarkLlamaGenerate500ConcatRows 869,713,097 ns / 2.20 GB versus BenchmarkLlamaGenerate500RowBuf 335,683,472 ns / 150 MB — 2.59x, and that is at only dim 256 / 4 layers / 500 tokens; the gap widens with context length.

SITES still on the quadratic path: nlp/decode.go:104-105 MHA.StepKV (reached from GPT.DecodeStep at :183); nlp/cla.go:330-331 CLA DecodeStep; nlp/t5_decoder.go:434-436 T5 decoder self-attention step; nlp/streaming.go:86-87 (bounded, so O(window) rather than O(T), but still two full copies per layer per token).

DEFECT: each is inside a per-layer loop of a per-token decode step. concatRows (nlp/decode.go:42) allocates a fresh [t+1, width] tensor and recopies all t existing rows EVERY token, giving O(T^2) copy traffic and O(T^2) bytes over a T-token decode. LlamaCache already adopted kvBufs.appendKV (nlp/rowbuf.go:207), which writes one row in place into a doubling backing tensor and returns a zero-copy contiguous view. These four did not.

FIX: embed bufs kvBufs in KVCache (nlp/decode.go:18), CLA's cache and T5's decoder cache, and replace each concatRows pair with cache.bufs.appendKV(cache.K, cache.V, l, k, v). appendKV is already fully generic over (K, V []*tensor.Tensor, l int), so it is a drop-in. CLA needs the GROUP index g rather than l as the buffer key.

VALIDATION GATE (benchmark only): the harness already exists and is exactly right — the ConcatRows/RowBuf pairs above. Add the sibling e2e pairs for GPT, CLA and T5 by reusing the existing bench-only kvAppendViaConcat flag (nlp/rowbuf.go:203), which flips the append while holding every other instruction identical — that is what makes the A/B clean.

EXPECTED: 2-3x end to end at 500 tokens and small dim, more at longer contexts and larger KV width. High confidence — this is a measured in-repo A/B, not an estimate.

BIT-IDENTITY BAR: value-identity is already pinned by TestRowBufAppendMatchesConcatRows (nlp/kvcache_perf_test.go:112) across F32/F64, and there is no reduction reordering or accumulation-width change. THE REAL RISK IS ALIASING, NOT NUMERICS: concatRows returns a FRESH tensor each step while appendKV returns a VIEW into a growing buffer. Any caller that retains an earlier cache.K[l] and assumes later appends cannot touch it changes behavior. rowBuf.owns (nlp/rowbuf.go:179) resynchronizes when a caller replaces the entry, but EACH of GPT, CLA and T5 must be audited for retained views BEFORE the swap — CLA especially, since it shares one cache slot across a group (cache.K[g], nlp/cla.go:330). Do that audit as the first step of the task and record what you found.

PERFSCAN RULE REQUIRED: quadratic accumulate-by-reallocation in a per-token loop. AST shape: an AssignStmt whose LHS is an IndexExpr into a cache-like slice field and whose RHS is concat(<same IndexExpr>, x) — i.e. c.F[i] = f(c.F[i], v) where f allocates output sized from both operands — inside a loop nested in a function whose name matches DecodeStep|StreamStep|Step\w*. Generalizes past tensors to append-free slice and buffer concatenation.
