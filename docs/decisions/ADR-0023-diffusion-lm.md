# ADR-0023 — LLaDA-style masked-diffusion language model: design + GO verdict

- Status: Accepted and implemented
- Date: 2026-07-15
- Task: §T654(a) (empty-backlog topic-discovery round 4 — the standout gap)
- Related: `nn/ddpm.go` (continuous diffusion, the sibling this complements), ADR-0022 (Titans; same "does it need new infra?" spike pattern — here the answer is NO)

## Context

The repo has CONTINUOUS diffusion (DDPM/DDIM, flow matching) but no DISCRETE/text
diffusion. LLaDA (Nie et al. 2025, arXiv:2502.09992) is a non-autoregressive LM:
a BIDIRECTIONAL transformer trained to predict masked tokens, generating by iterative
unmasking — shown competitive with autoregressive models at 8B. A read-only feasibility
spike checked whether GoAI can express it; verdict below.

## Findings (evidence)

- **Bidirectional attention is fully supported, forward + VJP, all surfaces.**
  `backend/attrs.go` AttnAttrs.Causal defaults false; ref forward/backward
  (`backend/ref/mha.go`, `mha_backward.go`) parameterize on it; `autograd/gradcheck_test.go`
  runs a passing non-causal MHA gradcheck; the ViT (`vision/vit.go` via `nlp/mha.go`)
  is built on exactly this. Metal/vulkan/cpu ALREADY parity-test the non-causal path
  (verified 2026-07-15: metal_test.go "mha"/"mqa", vulkan_test.go "mha"/"gqa3"/
  "swa-noncausal", cpu mha_test.go "plain"/"mqa" — all with Causal false/omitted), so
  bidirectional attention is fully covered; no added kernel or coverage work.
- **Backbone reuses the GPT block verbatim.** `nlp/gpt.go` Forward is mask-agnostic;
  the only causal coupling is one constructor line (`attn.Causal = true`). The diffusion
  backbone is the same GPT shape with Causal=false and a vocab+1 embedding/head row for [MASK].
- **Masked weighted loss needs no backend change.** `OpEmbed` (differentiable row-gather,
  grad scatter-adds to the table) gathers the masked-position logits; `nn.CrossEntropy`
  over them; the LLaDA 1/t ELBO weight is a per-sequence scalar `m/(L·t)`. ~30 lines.
- **Masking + sampler are pure Go with heavy reuse.** Masking mirrors `nlp/mlm.go` MLMMask
  (absorbing-state: always to [MASK]). The sampler reuses the per-position `nlp/sample.go`
  Sampler/Dist (+ Gumbel) in a Jacobi-like (`nlp/jacobi.go`) iterative loop: start all-[MASK],
  per step forward → sample masked positions → keep top-confidence so round(L·s) stay masked.
- **Blockers: none.** All first-order autograd over existing ops — no new kernels, no new
  VJPs, no second-order (unlike Titans).

## Decision — GO, implemented at size S/M

Reuse non-causal OpMHA + the GPT-shaped block + OpEmbed-gather CE; add three new pure-Go
pieces in `nlp/diffusion_lm.go`: DiffusionMask (forward corruption), the masked weighted-CE
training step, and DiffusionGenerate (iterative unmasking). Tests: unit (mask stats, CE
gather, sampler convergence) + an e2e char-LM on the in-repo grammar corpus (loss halves +
generation matches the grammar), mirroring `nn/newblocks_lm_e2e_test.go`. No backend
coverage work: the non-causal OpMHA path is already parity-tested on cpu/metal/vulkan
(verified), so diffusion-LM inherits verified bidirectional attention.

Implemented on 2026-07-15 in `nlp/diffusion_lm.go`. The shipped validation covers
the masked objective against a hand reference, mask statistics, all-parameter
finite-difference gradients, the unmasking schedule, and a trained grammar task
(eval CE 2.924→0.147). The implementation corrected one spike detail: greedy
generation ranks confidence by the chosen token's softmax probability; a generic
sampler-distance score would degenerate under greedy selection.
