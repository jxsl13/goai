# Training on real models: measured findings (§T483–§T485)

> **In plain terms:** "training" is teaching a model from data. GoAI ships a
> box of optimizers and training tricks; this page shows how each one behaved
> when teaching the same small model under identical conditions — so you can
> pick between them based on measurements instead of folklore.

Companion to [`benchmarking.md`](benchmarking.md), [`alignment.md`](alignment.md)
and [`inference.md`](inference.md): the training toolbox exercised on a REAL
transformer for the first time. Harness: identical architecture, identical init
seed, identical data order — only the component under test differs; models train
in-repo per run (`-short`-skipped tests in `llamagpu/`).

## The optimizer zoo (`optimizers_trained_test.go`, 120 steps, CE 3.135 start)

| Optimizer | Final CE | Note |
|-----------|---------:|------|
| SOAP | **1.188** | second-order leads |
| ScheduleFreeAdamW | 1.342 | `Eval()` before evaluation |
| Shampoo | 1.416 | needs `WithShampooRootEvery` (see below) |
| Sophia (GNB every 10, §T503) | 1.414 | lr ~6× below AdamW; labels resampled from the model for the Hessian |
| Muon(2D)+AdamW(1D) | 1.468 | Muon is 2-D-only by contract; this composite is the paper's recipe |
| AdamW | 1.489 | the reference |
| Adafactor | 1.490 | at its relative-step defaults |
| AdEMAMix | 1.491 | |
| LAMB | 1.513 | trust-ratio scaling wants a LARGE lr (0.01 here; 3e-3 failed the bar) |

Sophia joined in §T503 with the faithful GNB harness (labels resampled from the
model, ĥ = B·∇L̂², refreshed every 10 steps).

**Bug the zoo caught (§T483):** Shampoo recomputed its eigendecomposition-based
inverse roots every step — a 384×384 eigendecomposition per FFN matrix per step,
unusable at transformer scale (600 s timeout). `WithShampooRootEvery(k)` now
amortizes the roots (k=10–50 typical); the default stays the exact paper rule.

## The wrappers (`wrappers_trained_test.go`)

GaLore 1.380 (leads plain AdamW), Lookahead 1.473, Grokfast 1.502, CautiousAdamW
1.514, SAM 1.555 — SAM exercised through its real two-pass contract (gradients
recomputed at the perturbed point). Grokfast's λ=2 amplification roughly doubles
the effective step: retune the wrapped lr down (the paper does), or it
underperforms.

## NEFTune (`neftune_trained_test.go`)

On an engineered overfitting run the documented integration (noise between
`Embed` and `ForwardFromEmbed`, training only) delivers the textbook
regularization signature: train CE 1.184 → 1.366 (higher, by design) while
held-out CE improves 1.912 → 1.848. Unlike the attention-sink phenomenon, this
claim manifests at toy scale.

## Assertion discipline for training components

- An optimizer/wrapper must HALVE the loss under the shared harness — hyperparams
  may be tuned once, honestly, per family (LAMB's lr, Grokfast's compensation).
- A REGULARIZER's train loss sits above the plain baseline by design — require
  learning (within +0.5 CE of baseline), never baseline-matching; judge it on
  held-out data (§T485's corrected bar).

## Diffusion end-to-end (`nn/diffusion_e2e_test.go`)

The DDPM/DDIM pieces compose into a working generative model: a 2-input MLP
noise-predictor trained with `DDPMForward`/`DDPMLoss` on a radius-2 ring (2500
steps, loss 1.02 → 0.59) and sampled with 50 deterministic `DDIMStep`s from pure
noise reproduces the geometry — mean radius 1.91, spread 0.21, all half-planes
evenly populated (no mode collapse). Pure-Go, ~3 s. FLOW MATCHING passes the
identical harness (`nn/flowmatch_e2e_test.go`): the learned velocity field,
Euler-integrated from noise in 50 steps, lands at radius 2.09 / spread 0.27 —
both generative formulations reconstruct the distribution.

## Continual learning: EWC (`nn/ewc_e2e_test.go`)

The two-task harness (XOR quadrants → unit circle, task-indicator input): plain
sequential fine-tuning forgets task A completely (50% = chance) while EWC with a
20-batch diagonal Fisher and λ=5000 retains A at 89% with B at 98%. Two design
lessons baked into the test: tasks must be JOINTLY representable (pointwise
label conflicts make retention impossible — hence the indicator feature), and λ
must scale with the fine-tune pressure (500 retained only ~60%).

## Model merging (`llamagpu/merge_trained_test.go`)

Two dialect-specialized fine-tunes of a shared base (verb subsets of the grammar)
merged back into one model: each specialist is strong on its dialect and drifts on
the other (worst-case CE 1.49–1.99); the TIES merge reaches worst-case 1.38 and
the UNIFORM SOUP 1.25 — both beat every specialist's weak side. Note the soup
beating TIES here: fine-tunes from the SAME base stay linearly mode-connected,
which is exactly the soup's home regime; TIES's trim/sign machinery pays off for
more divergent donors.

## VQ-VAE (`nn/vqvae_e2e_test.go`)

Encoder → `VectorQuantizer` → decoder trained jointly on an 8-cluster ring:
reconstruction MSE 2.00 → 0.12 (94% of the data variance explained through the
discrete bottleneck), 6 of 16 codes active (no codebook collapse), and the
encoder demonstrably trains THROUGH the straight-through estimator — the wiring
that silently breaks first.

## Self-supervised learning: SimSiam (`nn/simsiam_e2e_test.go`)

A conv encoder pretrained WITHOUT labels (two jitter+shift views per image,
SimSiam loss with its stop-gradient) reaches perfect view alignment (loss → −1.0)
without representation collapse (feature std 0.11), and a linear probe on the
FROZEN features classifies the three shapes at 94% (chance 33%). The SSL loop —
augment → encode → project → predict → stop-grad loss — works end-to-end on the
library's own conv stack.

## Post-transformer architectures: a Mamba char-LM (`nn/mamba_lm_e2e_test.go`)

An attention-free language model assembled from `MambaBlock`s (one-hot embedding
→ 2 × RMSNorm+Mamba+residual → head) verifies the SSM block in its real role:
CAUSALITY is asserted structurally (mutating a later token changes no earlier
logits), the LM trains to CE 0.12 on the toy grammar (uniform 2.71), and greedy
generation runs deterministically. Pure Go, ~4 s.

RetNet passes the identical LM harness (`nn/retnet_lm_e2e_test.go`): causal by
retention decay, CE 3.13 → 0.16, deterministic generation — both post-transformer
families (SSM and retention) verified in real language-model roles.

Merging addendum (§T496): GreedySoup also beats the specialists' worst cases;
DARE is a LARGE-model tool — at toy scale the 0.9 drop rate clearly degrades and
0.5 is noise-level, the redundancy premise being the third scale-dependent claim
of the series (after the attention sinks).

A Jamba-style Mamba+MoE LM (`nn/moe_lm_e2e_test.go`) verifies sparse routing in
its real role: CE 3.64 → 0.11, causality preserved through the per-token experts,
and the auxiliary balance loss keeps all four experts at 20–28% utilization —
no routing collapse.

MLA (DeepSeek latent attention) passes the same LM harness
(`nn/mla_lm_e2e_test.go`): the low-rank KV compression preserves causality
bit-exactly and the LM trains to CE 0.21 — the architecture-e2e set now spans
SSM, retention, sparse MoE, and latent attention.

## PEFT on the built-in GPT: LoRA (`llamagpu/lora_trained_test.go`)

`nlp.ApplyLoRAGPT` attaches rank-r adapters to every attention projection —
attaching is a bit-exact no-op (zero-init B), training only the 24 adapter
tensors improves the target dialect's CE 1.19 → 1.07, and every base weight
stays bit-identical (the PEFT contract, asserted). This corrected an earlier
wrong blocker analysis: the projections are ordinary matmuls around the fused
SDPA core, not inside it.

## QLoRA (§T552, 2026-07-13)

The paper's recipe composed from the library's pieces: a trained base's attention
projections frozen as double-quantized NF4, rank-4 LoRA adapters trained on top.
Dialect CE: full base 1.190 → NF4-frozen 1.190 (NF4 is effectively lossless here,
the paper's fidelity claim) → QLoRA-tuned 1.080, with the 4-bit base bit-identical
after training. The math is the paper's; packed 4-bit storage during training is
not claimed (the frozen base is materialized dequantized in memory).

## Further reading

- Goodfellow, Bengio & Courville, *Deep Learning* (MIT Press 2016) — the standard textbook behind the optimizer/regularization vocabulary used here.
- Ruder 2016, *An Overview of Gradient Descent Optimization Algorithms* — the classic survey connecting SGD to the adaptive family.
- Lialin, Deshpande & Rumshisky 2023, *Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning* — the PEFT survey covering the LoRA-family methods measured here.
