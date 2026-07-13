# ADR-0015 — Typed backend names replace magic-string identifiers

Status: accepted (2026-07-06). Extends the idiomatic-Go / no-magic-strings line of
ADR-0014 (typed op parameters) from op attrs to backend identifiers.

## Context

Backends were referred to by bare string literals — `backend.Get("metal")`,
`SetPreference("cuda", "cpu")`, `preference = []string{"cuda","metal","vulkan","cpu"}`,
and each backend's `Name() string { return "metal" }` — ~114 occurrences across the
tree. Stringly-typed identifiers are typo-prone (`Get("mtal")` compiles and returns
not-found at runtime), give no discoverability (no constant to jump to), and read
as un-idiomatic Go. The user asked to eliminate magic strings project-wide in
favour of typed constants for type-safety.

## Decision

Introduce a small string enum `type Name string` (backend/names.go) with one
constant per backend and thread it through the whole selection surface:

```go
type Name string
const (
    CPU    Name = "cpu"
    Ref    Name = "ref"    // the constant; Reference() still returns the backend
    Metal  Name = "metal"
    CUDA   Name = "cuda"
    Vulkan Name = "vulkan"
)
```

- `Backend.Name() Name`; `Get(Name)`, `SetPreference(...Name)`, `Preference() []Name`,
  `Available() []Name`; the registry map and preference order are keyed by `Name`.
- Every backend's `Name()` returns its constant; the default preference order is
  `[]Name{CUDA, Metal, Vulkan, CPU}`; all call sites use the constants.
- The underlying type stays `string`, so a `Name` is still a ready map key and
  prints as itself (`String()` returns the plain string) — no churn at the display
  or registry-key layer.

The enum name for the reference backend is `Ref` (not `Reference`) because
`Reference()` is already the accessor function that returns the reference backend
instance.

## Consequences

- **+** Backends are named by a checked constant (`backend.Metal`); a typo is now a
  compile error (undefined identifier), and the constants are discoverable/jumpable.
- **+** Function signatures document intent (`Get(Name)` not `Get(string)`); a
  `string` variable can no longer be passed without a deliberate conversion.
- **+** Consistent with ADR-0014 and §C12: the library's identifiers are typed
  end-to-end (dtype, device kind, op, op-attrs, and now backend name are all enums).
- **−** The underlying type is still `string`, so an *untyped string literal*
  (`Get("metal")`) still compiles. This residual hole is closed by a mechanical
  guard (§V21, `TestNoMagicBackendNameStrings`): a go/ast pass fails on any
  backend-name string literal outside the two files where such a literal is the
  enum's own definition — backend/names.go and the separate tensor.DeviceKind
  stringer (tensor/device.go), which maps an already-typed device-kind enum to its
  canonical display name.

## Alternatives rejected

- **Keep `string` but add unexported const helpers** — no type-safety at call
  sites, and `Available()`/`Preference()` stay `[]string`. Rejected.
- **A full `iota` int enum with a name table** — backends are looked up by a
  human-facing key (registry map, `Get`), so a string-valued enum is the natural
  fit; an int enum would need a bijection to strings anyway. Rejected.
