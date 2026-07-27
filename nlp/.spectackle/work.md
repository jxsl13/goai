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
