---
schema: v1
prefix: TOK
---

## TOK-003 {applies: go:nlp.BPETokenizer.EncodeSpecial}
The a control or special token SHALL be minted only from the specials-parsing path such as EncodeSpecial, never from untrusted text passed through plain Encode.

Rationale: This is a security property, not an ergonomic one: the chat template emits its markers as literal text, so untrusted content that could encode to the same control ids would forge conversation structure. Migrated from cavekit SPEC.md V30.

## TOK-004 {applies: go:nlp.BPETokenizer.EncodeSpecial}
The a new tokenizer or segmentation path SHALL close both the marker-matching pre-split and the segmentation search itself, and prove the gate by mutation testing that an always-parse design fails.

Rationale: Both paths have been live. GGUF stores control tokens inside the main vocabulary, so Unigram's Viterbi and WordPiece's MaxMatch search all of it and can mint a control token with no specials pass involved, which the Hugging Face analogy hides. The injection case is a required test. Migrated from cavekit SPEC.md V30.

## DEC-003 {applies: go:nlp.Llama.SelfExtendForward}
The a model architecture's per-block features such as biases, QK-norm and the Granite scalars SHALL be applied in one block stack that DecodeStep, StreamStep and SelfExtendForward all route through, with attention as the only pluggable part.

Rationale: Duplicated decode logic drifts silently and asymmetrically: streaming once dropped all six hooks while the normal path kept them, so the same model and prompt produced different tokens through different entry points with no error. Migrated from cavekit SPEC.md V31.

## DEC-004 {applies: go:nlp.Llama.SelfExtendForward}
IF a duplicated decode path must remain, THEN the package SHALL name it in godoc and pin it with a cross-path test comparing tokens, using nonzero feature values plus a feature-free control.

Rationale: Without the control, a failing gate cannot distinguish real drift from a broken harness. Comparing logits rather than tokens hides divergence that only shows after argmax. Migrated from cavekit SPEC.md V31.

## intent
- T-01KYJQZEK7E1RV0AH9K8MFT89M Propagate the rowBuf KV append to the GPT, CLA, T5 and streaming caches: LANDED the GPT cache only, benchmark-validated on M2 Pro darwin/arm64 go1.26.5 via the existing bench-only kvAppendViaConcat flag (3 reps of -benchtime 5x, medians): BenchmarkGPTGenerate500 2,211,321,932 -> 159,364,963 B/op (13.9x less memory), 833 -> 630 ms (1.32x), allocs 310,777 -> 294,412. No benchmark covered GPT decode at all before this; the A/B pair was added beside the existing Llama one.\n\nTHE TIME PREDICTION WAS WRONG AND THE MEMORY ONE WAS RIGHT. The brief expected 2-3x wall clock by analogy to the Llama pair 2.59x; GPT measures 1.32x. A first 3-iteration run straddled 1.16-1.30x and was too noisy to report at all — 5 iterations were needed before three reps agreed within 1%. Anyone quoting this class of fix should lead with the memory figure, which is an order of magnitude and stable, not the wall clock.\n\nTHE BRIEF WAS PARTLY STALE: it claimed only LlamaCache had adopted appendKV. Cohere, Falcon, Gemma2 and DeepSeek-V2 had all adopted it since, so GPT, CLA and T5 were the actual remainder.\n\nALIASING AUDIT, done first as required, and recorded so it is not redone: no caller anywhere in nlp/ writes into a cache tensor in place, which is the only pattern that breaks under a view; retained views stay valid because appends only ever write row t and beyond; EvictStreaming replaces entries outright through GatherRows, which rowBuf.owns detects as a foreign view and resynchronizes from. Value identity was already pinned across F32/F64 by TestRowBufAppendMatchesConcatRows.\n\nNOT DONE, and left as the remainder: nlp/cla.go (2 sites) and nlp/t5_decoder.go (2 sites). CLA is the one needing real care — it shares ONE cache slot across a group, keyed by g rather than l, and followers read cache.K[g] after the leader appends. nlp/streaming.go was DECLINED: it is already bounded to sinks+window, so it is O(window) not O(T), and its concat is nested inside keepSinkRecent, which would need eviction-aware truncation rather than a drop-in append.\n\nGENERALIZED as perfscan PS2006 quadratic-cache-append with 5 fixtures and a mutation-probed detector. It finds exactly those 4 remaining sites and is silent on the fixed GPT path. Extending exprEqual to IndexExpr was required to prove a concat accumulates into its OWN slot; PS5001 shares that helper and stayed at 78 findings. Documented gap: the rule does not flag a concat nested inside another call, which is exactly the streaming shape.
