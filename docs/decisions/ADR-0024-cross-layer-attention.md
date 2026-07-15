# ADR-0024 — Cross-Layer Attention (CLA): isolated KV-sharing variant

- Status: Accepted (feasibility spike GO; building autonomously as an isolated variant)
- Date: 2026-07-15
- Task: §T654(b) (the "cheapest quick-win" of the round-4 discovery)
- Related: ADR-0023 (diffusion-LM; same reuse-the-GPT-primitives pattern), §R245 (Hymba — the "own projections + direct OpMHA dispatch, zero refactor" precedent)

## Context

CLA (Brandon et al. 2024, arXiv:2405.12981) groups adjacent transformer layers; within a
group ONE layer computes K,V and the others REUSE it (only Q differs per layer), shrinking
the KV cache by the group factor `Share`. Complements GQA/MLA (which shrink KV within a layer).

## Findings (spike, file:line)

- **OpMHA already takes decoupled q,k,v** (`nlp/mha.go:93` dispatches `OpMHA{q,k,v}`); the
  q=k=v-source coupling is only in the nlp wiring, not the kernel — so a block can call OpMHA
  with its OWN q and a SHARED (leader-layer) k,v with no backend change. Precedents:
  `nlp/llama_decode.go`, `nlp/streaming.go`, Hymba (§R245) all dispatch OpMHA directly.
- **Isolated, not a core edit.** Threading CLA through the shared `MHA`/`gpt.go`/`decode.go`
  would make three shared files conditional (nullable Wk/Wv, group-routing, cache slot-mapping)
  AND would propagate into the just-landed `nlp/diffusion_lm.go` + Medusa/LoRA/kv-evict which
  all sit on the same `Block`/`MHA`/`KVCache`. A new `nlp/cla.go` with its own block type + loop
  needs ZERO edits to gpt.go/mha.go/decode.go and reuses the same-package `concatRows`/`exec1`.
- **Fully differentiable, no infra.** OpMHA has a fused VJP (`autograd/vjp_transformer.go`); a
  shared k,v consumed by several layers' OpMHA calls is handled by the tape's fan-out summation
  (`autograd/autograd.go` "sum at fan-out points"). No backend/autograd change.
- **No collision** with the parallel `backend/cuda/*` lane — CLA touches only two new nlp/ files.

## Decision — GO, isolated variant, size S

New `nlp/cla.go`: `CLAConfig{Vocab, Ctx, Dim, Heads, Layers, Share, Eps}` (validate
`Layers % Share == 0`; `Share=1` degenerates to plain GPT). `CLABlock` holds Wq/Wo/W1/W2 always
and Wk/Wv only on LEADER layers (`l % Share == 0`); the forward loop: a leader projects+stashes
k,v from LN1(x), every layer computes its own q and calls `OpMHA(q, kStash, vStash)` + Wo, then
the standard GELU FFN. `CLACache` sized `Layers/Share`; decode leader appends via concatRows,
followers read the updated group slot. Tests: `Share=1 ≡ GPT` logits (same weights), full
gradcheck (exercises fan-out grads into the shared k,v), decode≡forward parity, cache-size
invariant `len(cache.K)==Layers/Share`, e2e grammar-corpus train (Share=2, loss halves),
Example. Reuses LayerNorm/OpMHA/concatRows/XavierUniform; only the two new files change.
Follow-ups (not v1): compose with the KV-evict/quant cache tooling; GQA composability.
