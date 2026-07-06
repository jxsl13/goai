# ADR-0003 — Opcode dispatch, single Execute choke-point, Recorder hook

- Status: accepted (autonomous loop, §T5)
- Date: 2026-07-05
- Relates: §I.L1, §V9, §V14, §T13 (autograd), review BLOCK B7 & the T5/T13 gate

## Context

T5 must define the Backend/Kernel interface such that (a) an async GPU backend
does not later break the API (§V14, B7), and (b) autograd (§T13) can intercept
op dispatch without rewriting L1 ops (T5↔T13 gate). Two shapes were considered:

1. **Method-per-op interface** — `Backend { Add(...); MatMul(...); ... }`. Fat
   interface (20+ methods), grows with every op, and every accel backend must
   reimplement all of them or embed a base.
2. **Opcode dispatch** — ops are `Op` codes; a backend exposes
   `Kernel(op, dtype) (Kernel, bool)`; all execution funnels through one
   `Execute(ctx, op, inputs, attrs)`.

## Decision

Opcode dispatch (2), with:
- `Kernel func(ctx, inputs, attrs) (outputs, error)` — pure w.r.t. autograd.
- A single `Execute` choke-point that: looks up the kernel, runs it, falls back
  to the reference backend if the active one lacks the kernel (§I4), then — if
  `ctx.Recorder` is set — calls `Recorder.Record(op, in, out, attrs)`.
- `Backend.Synchronize()` defines the execution/sync model (§V14): sync backends
  return immediately; async backends block until returned outputs are final.

## Rationale

- **One interception point.** Autograd is a `Recorder` plugged into `ctx`. The
  same forward path serves eager and grad modes → L1 ops need no autograd
  awareness (satisfies the T5↔T13 gate; §V9 keeps the ref backend as truth).
- **Thin, stable interface.** Adding an op = registering a kernel, not changing
  the interface → accel backends implement lookup + Synchronize only.
- **Sync model is explicit** in the interface, so a future async GPU backend
  slots in without an API break (§V14, closes B7).
- **Fallback is structural** — missing accel kernel routes to the reference
  backend at the choke-point (§I4).

## Consequences

- Attrs are dynamic (`map[string]any`) — read once per op, negligible cost;
  op-specific typed accessors guard misuse.
- Homogeneous input dtype assumed per op for now; mixed-dtype ops error until a
  promotion policy is specified (parked).

## Revisit if

Dynamic attrs or the `any`-typed kernel signature show up in profiles, or a
promotion policy for mixed dtypes becomes necessary.
