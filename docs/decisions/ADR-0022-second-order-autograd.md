# ADR-0022 — second-order / create-graph autograd, and how much of it Titans/TTT actually need

- Status: Proposed (feasibility spike; awaiting user priority)
- Date: 2026-07-15
- Task: §T650(a) (Titans / test-time-learning family), amends the earlier "deferred, needs second-order infra" verdict
- Related: §V25 (kernels never see the tape recorder), `autograd/checkpoint.go` (nested-tape precedent), `nn/gated_deltanet.go` (delta-rule = linear-memory closed form)

## Context

Titans (arXiv:2501.00663) and TTT-layers update a neural memory by a gradient step
AT TEST TIME, so training them naively needs to differentiate THROUGH that inner
gradient step — second-order / "create-graph" autograd. The earlier §T650(a) verdict
deferred the whole family as blocked on missing infra. A read-only feasibility spike
of `autograd/` refined this: the blocker is softer than it looked, and two paths reach
most of the value with today's first-order engine.

Findings (evidence):

- **Backward is non-differentiable BY DESIGN, but the seam exists.** The tape runs
  backward through a non-recording `exec` context (`autograd/autograd.go:32,60-63`);
  `autograd/vjp.go:13` states VJPs execute through a non-recording ctx. ≈15 of 69 VJPs
  are Execute-dispatched (recordable if handed a recording ctx: MatMul, Add/Sub/Mul,
  GELU/SiLU, AddBias, the fused transformer/CE/norm backwards); the other ≈54 are raw
  scalar loops, deliberately de-op-ified for backward speed (§T353/§T362/§C25).
- **Precedent**: `autograd/checkpoint.go:71-87` already spins a nested RECORDING sub-tape
  during a backward pass — tape re-entrancy is proven machinery.
- **§V25 is not violated** by create-graph: VJPs are not kernels; they call Execute,
  which strips the recorder before the kernel and records once after — no double-record.

## Options

1. **Hand-derived inner gradient (zero engine change).** Titans' inner objective
   ℓ = ‖M(k)−v‖² over a FIXED small MLP has an inner ∇ℓ that is a closed-form expression
   of ordinary forward ops (σ, mul, sub, matmul, transpose) — all of which have first-order
   VJPs. Write that gradient expression as forward ops; the outer tape then differentiates
   through it with today's engine. Cost: one manual derivation per memory architecture, no
   reuse; momentum/decay just lengthen the expression. Ships a real Titans/TTT layer now.
2. **Linear memory (already done).** For linear M, ∇ℓ = (Mk−v)kᵀ is the delta rule —
   `nn/deltanet.go` / `nn/gated_deltanet.go` already implement the gated form, differentiable
   w.r.t. q,k,v,α,β. TTT-Linear and linear-memory Titans (with momentum/decay, still closed
   form) are expressible today.
3. **Generic create-graph infra (M-sized, opt-in).** Add `Backward(out, ...BackwardOption)`
   with `WithGradTape(outer *Tape)` that swaps `t.exec` for a recording ctx onto a caller
   tape (≈20 lines in autograd.go). The ≈15 Execute-dispatched VJPs work unchanged; raw-loop
   VJPs gain a `if recording { op-composed } else { fast loop }` branch (preserving first-order
   perf). Correctness traps: the MatMul VJP's view-`Transpose` must become `OpTranspose` when
   recording; forbid Checkpoint+create-graph in v1. Scope: **M** for the Titans op subset
   (MatMul/Add/Sub/Mul/SiLU/Sum), **XL** for all-ops (double-backward of the fused MHA/CE/scan
   monsters is research-grade). Incremental + strictly additive + flag-gated.

## Decision (proposed)

Prefer **option 1/2 first**: ship a hand-derived (or linear) Titans/TTT layer with NO engine
change — that covers ≈80% of the paper's variants. Land the **M-sized option-3 subset** only
as a follow-up if a SECOND inner-loop architecture appears (at which point create-graph pays for
itself as a reusable meta-learning primitive: MAML-style inner loops, learned optimizers). Do not
attempt all-ops second-order up front. Awaiting user priority before implementing either.
