# ADR-0007 — CrossEntropy as a fused op (log-softmax + NLL)

- Status: accepted (autonomous loop, §T16)
- Date: 2026-07-05
- Relates: §T16, §V1, §V12, §B12, §B18

## Context

Cross-entropy over logits needs log-softmax. Composing it from existing ops
requires column-broadcast subtraction ([b,c] − [b,1]), which does not exist
(§B18); and a naive exp/sum/log composition overflows for large logits.

## Decision

One fused op `OpCrossEntropy(logits[b,c], targets[b]) → scalar mean loss`:

- Kernel (per row, f64 accumulation §V10): `m = max(z)`,
  `lse = m + log(Σ exp(z−m))`, `loss_i = lse − z[target_i]`; output = mean.
  The max-shift makes it exact for arbitrarily large logits (§V12).
- VJP: `∂L/∂z[i,j] = g · (softmax(z)[i,j] − 1[j = tᵢ]) / b`; targets are
  non-differentiable (nil grad).
- Targets are class indices carried in a float tensor of the logits' dtype
  (exact ≤ 2^53; int dtypes pending, §B12 precedent).

## Rationale

- This is the canonical design: PyTorch's `CrossEntropyLoss` fuses log_softmax
  + NLL for exactly these stability and gradient-quality reasons; the fused
  backward `(softmax − onehot)/N` is a textbook result.
- A ref-only kernel suffices today: the cpu backend lacks it and `Execute`
  falls back to ref automatically (§I4) — the fallback path earns its keep.

## Consequences

- MSE stays a pure composition (Sub/Mul/Mean) — no new kernel, gradients come
  from existing VJPs.
- A general broadcast layer (§B18) may later allow unfused variants; the fused
  op remains for stability/perf.

## Revisit if

Softmax probabilities are needed as a standalone op (§T21 attention) — then a
stable `OpSoftmax` lands and CE can share its row-max machinery.
