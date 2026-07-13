# Alignment in GoAI: recipes and measured findings (§T464–§T470)

> **In plain terms:** "alignment" means steering a trained model toward
> behavior people actually want, usually by rewarding good outputs. This page
> documents GoAI's alignment methods (RLHF, DPO, GRPO and friends), including
> a reproduced case of the model "gaming" its reward — and the fix.

Every claim below is enforced by a test that trains a REAL model (the torch-golden
GPT in `nlp/testdata`) end-to-end with library pieces only; file names are given so
the experiments re-run on every full suite pass.

## The bridge: from logits to every alignment objective

`nn.TokenLogProbs(ctx, logits, targets)` and `nn.SequenceLogProbs` are the
differentiable, numerically stable link (max-shift log-softmax + one-hot gather,
composed from dispatched ops) between a language model's logits and the inputs of
every preference/RL loss. Before §T464 the loss docs referenced this helper but it
did not exist — the entire family was unrunnable. Validated by parity, a
1000-magnitude stability row, and finite-difference gradient checks
(`nn/logprobs_test.go`).

## RL-based: GRPO

`nlp/grpo_e2e_test.go` — sample a group of rollouts with `Generate`, normalize a
reward with `nn.GroupAdvantage`, backprop `nn.GRPOLoss` through the per-token
log-probs. Mean reward 0.042 → 0.979 in 40 iterations. Lessons: sparse all-zero
reward groups give zero advantage (use longer rollouts / larger groups), and the
KL anchor must be weak enough to allow movement (β=0.001 here).

## Preference-based: DPO and its family

`nlp/dpo_e2e_test.go`, `nlp/pref_variants_e2e_test.go` — each loss honors its real
contract: DPO/IPO take frozen-reference sequence log-probs; SimPO takes
LENGTH-NORMALIZED policy log-probs and no reference; CPO adds a chosen-NLL term
(asserted separately: the chosen response's own log-prob must rise); KTO takes
UNPAIRED examples with desirability labels. All five flip an initially NEGATIVE
chosen-vs-rejected margin (−1.37) decisively: DPO +margins on 3/3 pairs, IPO →
+3.5, SimPO → +11.8, CPO → +7.3, KTO → +12.9.

## The reward model, its failure mode, and the fix

`nlp/rlhf_pipeline_e2e_test.go` (static) and `nlp/rlhf_iterated_e2e_test.go`
(iterated; `-short`-skipped, ~47s):

| Pipeline | Learned reward | True objective (τ-rate) |
|----------|---------------:|------------------------:|
| Static: reward model trained once, then GRPO | −3.43 → −0.81 ✓ | 0.042 → **0.000** ✗ |
| Iterated: reward model retrained every 5 policy updates on freshly labeled samples | rises ✓ | 0.042 → **1.000** ✓ |

Findings, all reproduced deterministically:

1. **Reward models need on-distribution data.** A head trained on a handful of
   synthetic sequences does not generalize to sampled rollouts (73–80% pair
   accuracy); ranked samples from the policy itself fix that.
2. **Reward hacking is the default, not the exception.** With a static reward
   model at >90% pair accuracy, GRPO drives the LEARNED reward up while the true
   objective falls to zero — the policy drifts off-distribution and finds
   sequences the imperfect head scores highly. Reproduced at every head capacity
   tried (linear last-position, linear pooled, MLP).
3. **Iterated retraining is the lever, and its FREQUENCY matters.** Refreshing the
   reward model on freshly labeled current-policy samples every 5 updates rescues
   the true objective completely (→ 1.000); an 8-update cadence managed only
   0.104. KL anchoring alone (β up to 0.01) did not prevent the hacking.

## Reward-model architecture note

For count-like preferences, score PER POSITION and mean-pool
(`head(hidden_i)` averaged over response rows) — a single last-position readout
plateaued at 73% pair accuracy, pooled-then-scored at ~87%, per-position MLP >90%.

## GSPO (§T549, 2026-07-13)

Qwen3's sequence-level objective (`nn.GSPOLoss`): one length-normalized likelihood
ratio per response, tight sequence-level clip (ε≈3e-4), no KL term. On the same
harness as the GRPO flagship it trains the policy identically (mean reward
0.042→0.958 in 40 iterations) with a simpler update: one loss call over the
concatenated group instead of a per-rollout loop. Exact collapse onto GRPO(β=0)
at sequence length 1; a clipped sequence contributes zero gradient to all its
tokens (the stability property it exists for).

## Further reading

- Ouyang et al. 2022, *Training Language Models to Follow Instructions with Human Feedback* (InstructGPT) — the RLHF pipeline all of these methods descend from.
- Casper et al. 2023, *Open Problems and Fundamental Limitations of RLHF* — the survey; the reward-hacking case study reproduced here is one of its central failure modes.
- Rafailov et al. 2023, *Direct Preference Optimization* — the paper that started the reward-model-free family (DPO/IPO/KTO) measured in this document.
