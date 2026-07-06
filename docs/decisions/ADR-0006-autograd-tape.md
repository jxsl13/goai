# ADR-0006 — Tape-based reverse-mode autograd

- Status: accepted (autonomous loop, §T13)
- Date: 2026-07-05
- Relates: §I.L2, §T5 (Recorder seam), §T13, §T14, ADR-0003

## Context

L2 needs reverse-mode AD over the L1 dispatch without touching L1 ops. T5
provided the seam: `backend.Recorder` is called by `Execute` after every
forward op.

## Decision

1. **Tape** (Wengert list): a `Tape` implements `Recorder`; recording contexts
   come from `tape.Context()`. Nodes store (op, inputs, outputs, attrs) in
   execution order — already topologically sorted, so `Backward` is a single
   reverse iteration.
2. **Identity by tensor pointer.** Gradients are keyed by `*tensor.Tensor`.
   Views (different pointer, shared storage) are distinct nodes; propagating
   grads through view ops is future work (§B15).
3. **VJP registry** `map[Op]VJP` mirroring the kernel registry: adding an op's
   gradient = registering a VJP, no interface change (T14 fills the table).
4. **Seed = ones.** `Backward(y)` seeds ∂y/∂y = 1 (ones tensor of y's shape);
   scalar losses give the standard gradient. Explicit seeds can be added later
   without breaking this API.
5. **Backward is not recorded.** VJPs execute in a non-recording context, so
   the tape never grows during Backward (no accidental higher-order taping).
   Higher-order AD would use a recording backward context — deliberately out of
   scope now.

## Rationale

- Tape + reverse walk is the canonical, proven design (PyTorch eager, JAX
  linearize): simple, correct for DAGs including fan-out (gradient
  accumulation by addition at reuse points).
- Pointer identity avoids intrusive IDs on Tensor; the tensor values recorded
  in the node are exactly the forward values needed by the VJPs.
- Registry keeps L2 open for new ops the same way L1 is (ADR-0003).

## Consequences

- Grads for a view of a tensor do not flow to its base (§B15) until view-VJPs
  land.
- Memory: the tape holds forward tensors alive until Backward; fine at current
  scale, checkpointing is a later optimization.

## Revisit if

Higher-order gradients are needed (record backward), or view-grad demand
arrives (add view ops as recorded ops with stride-aware VJPs).
