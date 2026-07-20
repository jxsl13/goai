---
name: tokenizer-trust-boundary
description: Special/control tokens are a security boundary — closing the obvious marker-matching path is only half the fix; the segmentation algorithms reach them too.
metadata: 
  node_type: memory
  type: project
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

In GoAI (§V30, §B72, fixed 2026-07-18), a control token must be mintable ONLY from the specials-parsing path (`EncodeSpecial`), never from untrusted text through plain `Encode`.

**Why:** the chat template emits its markers (`<|im_start|>` etc.) as literal text. If untrusted user content encodes to the same control ids, it forges conversation structure. So this is a security property, not an ergonomic one — and it means the naive fix ("just match specials everywhere") is itself the vulnerability.

**How to apply:**
- Two independent paths reach control tokens, and BOTH were live. The obvious one is the marker-matching pre-split — do it only on the trusted path (llama.cpp's `parse_special` flag IS the boundary; mirror it). The non-obvious one: **GGUF stores control tokens INSIDE `tokenizer.ggml.tokens`**, so Unigram's Viterbi and WordPiece's MaxMatch search the whole vocabulary and already emitted bare control ids with no special-matching pass involved. HF never hits this because its added tokens live outside the model vocab — so reasoning by analogy to HF actively misleads here.
- Any new tokenizer or segmentation path must gate on both. The injection case is a REQUIRED test.
- Verify by **mutation**: rebuild the naive always-parse design and confirm the injection gate fails with the forged token visible in the id stream. An assertion that merely passes proves nothing (see [[self-policing-guard-pattern]]).
- When llama.cpp and HF disagree on a detail, prefer the deterministic contract over bug-for-bug fidelity — llama.cpp's per-marker pass uses an unstable sort, leaving equal-length overlap ties unspecified; HF's leftmost-longest is total. Document which you matched and why.
- Class of bug that hides this: the end-to-end path was untested. Fifteen `Render(` call sites in tests, and **none** piped their output into `Encode` — they compared strings, and `Decode` round-trips the STRING correctly even when the IDS are wrong. When two components are only ever tested separately, the seam between them is where the bug lives.

Related: [[integration-audit-method]], [[self-policing-guard-pattern]], [[goai-autonomous-loop]].
