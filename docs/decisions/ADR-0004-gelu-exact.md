# ADR-0004 — GELU uses the exact (erf) definition

- Status: accepted (autonomous loop, §T6)
- Date: 2026-07-05
- Relates: §T6, §V1

## Context

GELU has two common forms: the exact definition
`gelu(x) = 0.5·x·(1 + erf(x/√2))` and a tanh approximation
`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`. They differ by up to ~1e-3 and
must not be conflated in a parity test.

## Decision

The reference GELU is the exact erf-based form. `math.Erf` (Go) and `math.erf`
(Python golden) both provide it.

## Rationale

- It is the canonical mathematical definition and PyTorch's default
  (`approximate='none'`), so it is the right numeric truth (§V9) and golden
  target (§V1).
- The tanh approximation is a speed hack; it belongs behind an explicit
  `approximate` attr if ever needed, not as the default.

## Consequences

- A future tanh-approx GELU is a separate op/attr with its own golden.

## Revisit if

A model format requires the tanh approximation for bit-parity with a published
checkpoint — then add it as an opt-in variant.
