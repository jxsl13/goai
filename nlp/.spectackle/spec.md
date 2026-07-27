---
schema: v1
prefix: TOK
---

## TOK-001 {applies: go:nlp.BPETokenizer.EncodeSpecial}
The a control or special token SHALL be minted only from the specials-parsing path such as EncodeSpecial, never from untrusted text passed through plain Encode.

Rationale: This is a security property, not an ergonomic one: the chat template emits its markers as literal text, so untrusted content that could encode to the same control ids would forge conversation structure. Migrated from cavekit SPEC.md V30.

## TOK-002 {applies: go:nlp.BPETokenizer.EncodeSpecial}
The a new tokenizer or segmentation path SHALL close both the marker-matching pre-split and the segmentation search itself, and prove the gate by mutation testing that an always-parse design fails.

Rationale: Both paths have been live. GGUF stores control tokens inside the main vocabulary, so Unigram's Viterbi and WordPiece's MaxMatch search all of it and can mint a control token with no specials pass involved, which the Hugging Face analogy hides. The injection case is a required test. Migrated from cavekit SPEC.md V30.

## DEC-001 {applies: go:nlp.Llama.SelfExtendForward}
The a model architecture's per-block features such as biases, QK-norm and the Granite scalars SHALL be applied in one block stack that DecodeStep, StreamStep and SelfExtendForward all route through, with attention as the only pluggable part.

Rationale: Duplicated decode logic drifts silently and asymmetrically: streaming once dropped all six hooks while the normal path kept them, so the same model and prompt produced different tokens through different entry points with no error. Migrated from cavekit SPEC.md V31.

## DEC-002 {applies: go:nlp.Llama.SelfExtendForward}
IF a duplicated decode path must remain, THEN the package SHALL name it in godoc and pin it with a cross-path test comparing tokens, using nonzero feature values plus a feature-free control.

Rationale: Without the control, a failing gate cannot distinguish real drift from a broken harness. Comparing logits rather than tokens hides divergence that only shows after argmax. Migrated from cavekit SPEC.md V31.
