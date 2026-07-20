---
name: integration-audit-method
description: "GoAI lesson — \"exported with tests\" ≠ usable; audit call sites, integrate via small adapters, unblock experiments by training in-repo, verify with parameter-extreme equivalences."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 89975edc-f6ec-4912-922c-d5efd862e3d7
---

From the GoAI T434–T444 integration era: a library can be full of well-tested exported
algorithms that NO user-facing path can actually run (found: penalties, Mirostat,
watermarking, regex-guided decoding, CFG, Jacobi, DoLa, Medusa — all formula-only).

**Why:** Unit tests prove correctness of a function, not reachability from the public
workflows. THREE mechanical signals, all grep-able: (1) zero-call-site exports,
(2) "takes a concrete type where an interface belongs", (3) docs referencing a
helper that was never built (GoAI's SequenceLogProbs was cited by five loss docs
and didn't exist — building it unlocked the whole alignment family, T464/T465:
GRPO 0.04→0.98 reward, DPO 100% preference accuracy, both on a real model).

**How to apply:**
1. Audit: for each exported symbol, count references outside its defining file; also
   check whether generation/training entry points accept interfaces or concrete types.
2. Integrate with SMALL composition adapters (wrap into the existing interface, e.g.
   a TokenSampler wrapper) or thin loops mirroring an existing loop's structure —
   never rewrite the algorithm.
3. "Needs model files / trained artifacts" blockers dissolve by TRAINING IN-REPO with
   the library's own loop (minutes, deterministic corpus) — see [[perf-gap-vs-python]]
   §T434: this also exposed that high draft acceptance ≠ speedup when dispatch-bound.
4. Verify new paths with parameter-extreme equivalences that COLLAPSE them onto known
   paths (γ=1 ⇒ plain decode, ε=1 ⇒ reject-all ⇒ greedy, α=1 ⇒ argmax-only,
   no-penalties ⇒ bit-identical rng stream) — sharper than any soft quality assert.
5. Assert what the THEORY guarantees; LOG the rest. Some paper claims are
   scale-dependent and legitimately absent at toy scale (GoAI found three: attention
   sinks, DARE's redundancy premise, and — inverted — NEFTune DID show). A failed
   naive assertion is often the finding (Mirostat is a ceiling, not a thermostat;
   reward hacking). Related discipline: regularizers are judged on held-out data,
   never train-loss-matched; model agreement is measured teacher-forced.
6. Audit ROUND 2 lesson (§T519, 2026-07-13): pure math helpers that predate an
   implementation drift into UNLINKED spec functions — SelfExtendRelPos/Positions
   existed but SelfExtendForward built its positions inline. Fix pattern: extract
   the construction the implementation uses, then pin implementation == pure-math
   spec with a consistency test over ALL pairs. The pure function becomes the
   TESTED spec, not dead documentation.
7. Fuzz-sweep companion (§T520/§B47, 2026-07-13): when a fuzzer finds a bug,
   grep for the CLASS across siblings before closing (gguf's overflow existed at
   2 sites; the tokenizer id-alloc bug existed latent in WordPieceFromJSON which
   fuzz had NOT hit). Bounds checks use the subtraction form (a > n ∨ b > n−a),
   never a+b > n (uint wrap). Fuzz harnesses recover() and hide stacks — use a
   temporary no-recover repro test to get the real frame.
8. Gradient-scale bugs (§B49, 2026-07-13): shape/parity/gradcheck suites are BLIND
   to a doubled gradient (autograd tests run on cpu; unit nets lack the affected op
   on the tape). Only tight trained-model CE bars catch them — keep those bars
   tight, run them in every full sweep, and treat a "barely missed the halve bar"
   failure as a possible gradient-scale bug, not flakiness. Structural rule (§V25):
   kernels must never see the tape recorder; recording happens once at the Execute
   choke point. Escalation ladder that worked: live fix → grep the CLASS (46 sites)
   → make the class impossible by construction.
9. Family-building pattern (§T548–T560, 2026-07-13): when a curated source list
   yields a gap FAMILY (8 i-quants, 3 sparse-attention variants), one recipe per
   family carries every member: (a) find a LOCAL cross-reference implementation
   first (gguf-py inside .venv beat any web fetch — grep site-packages before
   reaching for the network); (b) extract constant tables PROGRAMMATICALLY, never
   by hand (a hand-typed hex grid had a silent one-nibble typo); (c) per member:
   verbatim port + cross-impl golden + hostile sizes + short fuzz; (d) pin each
   NEW mechanism by an exact COLLAPSE onto an existing verified path (topK≥all ⇒
   full attention; uniform channels ⇒ the scalar predecessor; length-1 ⇒ the
   token-level sibling) plus ONE structural test of the property the mechanism
   exists for (isolation/routing/erasure). Collapse tests also catch missing
   conventions (KDA's L2 norm surfaced as a 1e-3 collapse failure).
10. Topic-discovery gaps are often INTEGRATION gaps, not absences (§T648, 2026-07-15):
    "found" aux-loss-free MoE balancing absent via a CONCEPT grep (`aux.loss.free|
    bias.*balance` → 0 files) and booked it — but `nn/lossfree.go LossFreeBalancer`
    already existed, just unwired into SparseMoE. The real task was wiring an orphan,
    not building from scratch. A concept grep matches your GUESSED phrasing, not the
    repo's filename ("lossfree" ≠ "aux.loss.free") — same class as the §T646 crude
    "grep op in cpu files" miss. Before booking a discovery gap as absent: ALSO
    `ls *<concept>*` for likely filenames and read the target module; treat "a
    primitive exists but nothing calls it" as the COMMON case (the orphan class). One
    concept grep proves nothing — definitive absence needs a filename + call-site
    sweep. The delegated implementer caught this; a good delegation brief says "reuse
    any existing primitive" so a fresh context surfaces it.
11. Audit ALL forward paths, not just the one your test covers (§T741, 2026-07-16):
    adding Qwen2 q/k/v bias to Llama's `hiddenFromEmbed` (full-sequence Forward) and a
    green forward-parity test LOOKED complete — but the KV-cached decode path
    `DecodeStep` (behind `Generate`) projects q/k/v separately and silently DROPPED the
    bias, so generation would diverge from Forward with zero test failure. Pattern: a
    per-block/per-op edit must be replicated in EVERY path that re-implements that block
    (prefill Forward, cached decode, any fused variant) — grep the op you touched
    (`b.Wq`, `project(`) across the package, not just the file your test exercises. The
    regression test that pins it: decode-vs-Forward last-row parity (was 0.0 bit-identical
    once fixed). Corollary: a new architecture with NO separate decode path (e.g. Gemma
    T742, Forward-only) has no such gap — but confirm that absence explicitly. Reinforces
    the HF-loadable-architecture sweep now at 8 families (added Qwen2, Gemma) — the
    systematic-sweep meta-lesson from [[t650-topic-discovery-round]] keeps paying out.
