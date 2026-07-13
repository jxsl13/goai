# ADR-0014 — Typed per-op parameters replace the `map[string]any` attrs bag

Status: accepted (2026-07-06). Supersedes the attrs design in ADR-0003 (dispatch).

## Context

Every op carries optional parameters (attention heads, RoPE base, loss β, …)
through the one dispatch choke-point `Execute(ctx, op, inputs, attrs)`, onto the
autograd tape, and into its VJP. Since ADR-0003 `Attrs` was `map[string]any` with
typed accessors:

```go
scale := attrs.Float("attn_scale", 1)              // read
Execute(ctx, OpMHA, ins, Attrs{"heads": 2, "causal": true}) // write
```

This is the one clearly un-idiomatic type in the library. Its costs:

- **Stringly-typed keys** — a typo (`"attn_scael"`) compiles and fails *silently*
  by returning the default; ~30 keys with no single source of truth.
- **`any` boxing** — no compile-time type safety; a wrong-typed value silently
  falls through to the default.
- **Implicit per-op contract** — which keys an op honours, their types, and their
  defaults were discoverable only by reading the kernel; each default was
  duplicated between the forward kernel and the backward VJP and could drift.
- **Untyped construction** — `Attrs{"heads": 2}` had no field checking.

The user asked to make the whole library idiomatic Go, with one consistent style.

## Decision

Make `Attrs` a **sealed interface** with **one concrete struct per op** (grouped
where ops genuinely share a parameter set):

```go
type Attrs interface{ opAttrs() } // sealed by the unexported marker method

type AttnAttrs struct {
    Heads   int; KVHeads int; Causal bool; Scale float64; ALiBi bool; Window, Block int
}
func (AttnAttrs) opAttrs() {}
func (a AttnAttrs) WithDefaults() AttnAttrs { /* Heads→1, KVHeads→Heads, Scale→1 */ }
```

- **Construct** with a checked struct literal:
  `Execute(ctx, OpMHA, ins, AttnAttrs{Heads: 2, Causal: true})`.
- **Read** by type-asserting (comma-ok, so a nil/mismatched attrs degrades to the
  zero value exactly as the old accessor degraded to its default):
  `p, _ := attrs.(AttnAttrs); p = p.WithDefaults()`.
- **Defaults live once**, in each struct's `WithDefaults`, called by both the
  kernel and the VJP — they can no longer drift apart.
- The `Execute` / `Kernel` / `Recorder.Record` / `VJP` signatures are **unchanged**
  (the parameter is still named `Attrs`), so dispatch, taping, and fallback stay
  op-agnostic; only the read/write ends changed.
- Ops that share a backward pass share a struct (OpMHA + OpFlashAttn → `AttnAttrs`,
  since `mhaVJP` serves both); OpLayerNorm + OpRMSNorm → `NormAttrs`; the reductions
  → `ReduceAttrs`. All others are per-op.
- Zero-value-meaningful sentinels are preserved: `ArgMaxAttrs.Axis == ReduceAll`
  (the exported `math.MinInt` sentinel) still means "flatten every axis", gated on
  the assertion's ok so a real axis 0 is distinguishable from an absent one.

## Consequences

- **+** Wrong field name or type is now a **compile error**, not a silent default.
- **+** Each struct's fields are the op's self-documenting, godoc-rendered contract
  (dual-audience field comments, §C13); no more reading the kernel to learn the
  knobs.
- **+** Defaults are single-sourced (kernel and VJP share `WithDefaults`), closing
  a latent divergence class.
- **+** Consistent with the library's functional-options house style (§C12): typed,
  discoverable, no magic strings anywhere.
- **−** One struct per parameterised op (~19 types) plus a `WithDefaults` each —
  more declarations, but purely mechanical and self-documenting. The `*Attrs`
  structs are op-parameter bags, so apicheck exempts them from the per-type Example
  rule by a category rule (`backend.*Attrs`), the same category as the former
  `backend.Attrs` (§C13/§V19).
- **−** A caller can still pass the wrong concrete `Attrs` type to an op; the
  comma-ok assertion then yields defaults rather than a compile error (the op↔attrs
  binding is by convention, not by the type system, because dispatch is over a
  runtime `Op` enum). Gradcheck (§V2) and parity (§V1) catch a genuinely wrong
  wiring.

## Alternatives rejected

- **Functional options on `Execute`** (`Execute(ctx, op, ins, Heads(2), Causal(true))`)
  — consistent with §C12 and lighter, but the options are not bound to a specific
  op (any option compiles with any op → mismatches stay runtime-only) and the read
  side would still marshal through a map internally. Weaker than typed structs.
- **Typed key constants** (`attrs.Get(KeyHeads)`) — still `map[string]any`
  underneath; the least idiomatic option. Rejected.
- **Generics keyed by op** — impossible: dispatch is over a runtime `Op` enum, not
  a compile-time type, so `Execute` cannot be parameterised per op.
