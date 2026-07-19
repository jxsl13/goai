# ADR-0025 — Byte Latent Transformer (BLT): the "needs ragged tensors" deferral is invalid

- Status: Accepted and implemented
- Date: 2026-07-15
- Task: §T650/§T654/§T655 (BLT was repeatedly deferred as "needs ragged/segmented tensors")
- Related: ADR-0022 (Titans — same "spike overturns an infra deferral" pattern), ADR-0023 (diffusion-LM), ADR-0024 (CLA); `nn/mod.go` (the data-dependent-dimension precedent), `nn/qknorm.go` (the composed-mask trainable attention precedent)

## Context

BLT (Meta 2024, arXiv:2412.09871) is a tokenizer-free byte-level LM: an entropy patcher
segments the byte stream into variable-length patches, a local encoder pools bytes→patches,
a latent transformer runs over patches, a local decoder maps patches→bytes. It was deferred
three times as needing ragged/segmented-tensor infra. A read-only spike re-examined that with
the "spike-before-deferring" lesson (Titans + diffusion-LM both looked infra-blocked, both
shipped) — and the deferral is WRONG.

## Findings (spike, file:line)

- **Ragged is not needed.** `numPatches` is a PER-FORWARD dimension decided host-side, exactly
  like MoD's top-k `k` (`nn/mod.go:72-107` builds a data-dependent-sized one-hot `S[k,seq]` at
  forward time and runs differentiable gather/scatter through plain OpMatMul). GoAI trains
  single-sequence rank-2 `[seq,d]` (gpt/diffusion/CLA), so `[numPatches,d]` just varies per
  step — no ragged tensor, and (without a batch axis) not even padding/masking. The boundary
  decision is non-differentiable IN THE PAPER too (entropy model trained separately, boundaries
  frozen) → a host-side Go loop producing `[]int` offsets is faithful, not a compromise.
- **Patch-restricted (cross-)attention is already shipped, trainable.** The fused OpMHA has no
  free mask and OpMHAMasked is inference-only (no VJP) — but `nn/qknorm.go:49-98` already
  composes a fully-differentiable masked attention from OpMatMul→additive-mask OpAdd→OpSoftmax→
  OpMatMul (all VJP'd, gradcheck'd). Byte↔patch pooling is that with a host-built patch-assignment
  block mask. Zero backend/kernel/VJP work; the GPU-ops-all-backends rule is not triggered.
- **Entropy patcher is trivial**: a GPT with Vocab=256 (the diffusion-LM retarget precedent) +
  a host-side `−Σp·log p` loop over softmax rows + a boundary rule. No new op.
- **Reuse**: GPT block stack for the latent transformer; OpEmbed for byte + hash-n-gram
  embeddings; windowed OpMHA (`AttnAttrs.Window`) for the local byte layers.

## Decision — GO-BUT-LARGE, implemented

BLT is buildable on the current engine with NO autograd/backend/kernel change and no cuda-lane
collision — PURE composition. Correct the SPEC deferral (ragged-tensors reason is invalid).
But it is L-sized: three sub-models + patcher, ≈700–1000 lines (calibration: diffusion-LM 354,
CLA 375, Titans 642). New `nlp/blt.go` (BLTConfig, EntropyPatcher, LocalEncoder with masked
cross-attn pooling, latent GPT, LocalDecoder, Forward/Loss/Generate) + tests including a collapse
test (stride-1 patches + identity pooling ≈ byte-GPT, the Titans/CLA pinning pattern) and an e2e
on the ASCII grammar corpus (chars are bytes → CE-halves + valid-grammar generation fits §V16).
Optional perf follow-up: a trainable free-mask fused attention (OpMHAMasked VJP) to replace the
composed-primitive path.

## Outcome

Implemented in nlp/blt.go on 2026-07-15 using the composition above, without ragged tensors or
new backend/autograd work. Fresh 2026-07-19 validation kept the stride-1 byte-GPT collapse,
patch-span isolation, 52-parameter gradient check, entropy patcher convergence (CE 5.477 to
0.654), and end-to-end byte-LM convergence (CE 5.631 to 0.184) with valid generation green.
