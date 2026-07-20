---
name: prefer-idiomatic-go-functional-options
description: Write idiomatic Go; prefer functional options / variadic args over long positional parameter lists
metadata:
  node_type: memory
  type: feedback
  originSessionId: 56de00c6-6c80-4734-a46b-f5e03083b2b4
---

In GoAI, always write idiomatic Go. In particular, use **functional options** (variadic `...Option` args) for functions that take several optional/configurable parameters, instead of a long list of positional parameters.

**Why:** The user explicitly asked for idiomatic Go and named variadic-argument functions. Long positional signatures (e.g. `KTO(ctx, pl, rl, labels, beta, zRef, lambdaD, lambdaU)` — four positional floats) are error-prone at the call site and un-Go-like.

**How to apply:**
- Required inputs (the tensors) stay positional; the hyperparameters become `...Option`.
- Provide a config struct with sensible defaults + `func Opt(v) Option` setters (Rob Pike's self-referential functions pattern). Guard invalid values (e.g. β≤0 keeps the default).
- Reserve options for ≥2 optional params or where the set will grow; a single well-named param can stay positional (over-abstraction is also un-idiomatic).
- Example applied: the preference-alignment family shares `nn.PrefOption` with `nn.Beta`, and KTO-only `nn.ReferencePoint`/`nn.DesirableWeight`/`nn.UndesirableWeight`; `nn.DPO/IPO/KTO(ctx, …tensors, opts ...PrefOption)`.
- Extend the same pattern to future APIs (and, going forward, the other loss/sampler hyperparameters). See [[goai-autonomous-loop]].
