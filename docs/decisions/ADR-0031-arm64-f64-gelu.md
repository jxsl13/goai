# ADR-0031 — measure a separately gated ARM64 F64 GELU intrinsic path

- Status: accepted experiment; production promotion pending
- Date: 2026-09-09
- Research: `R-01M23B9YZ1E16`
- Proposal: `P-01M23BZPZNF9V`
- Task: `T-01M23C5EWPE8P`
- Base: `b9c059464edb4a339073c75ea8037c5472bf9aea`

## Context

GELU is an activation function used in transformer feed-forward networks. Its
exact-erf definition and derivative are:

```text
GELU(x) = 0.5*x*(1 + erf(x/sqrt(2)))
GELU'(x)*g = g*(0.5*(1 + erf(x/sqrt(2))) + x*exp(-0.5*x*x)/sqrt(2*pi))
```

"Exact-erf" identifies the mathematical definition, not a claim that floating-point
evaluation has no rounding error. ADR-0004 excludes replacing it with the common
tanh approximation. The default CPU implementation is bit-exact against the
reference backend; the existing AMD64 SIMD implementation has an explicitly
tolerant comparison instead.

The previous M2 scalar callback-specialization experiment failed all three
performance campaigns and was reverted. Its diagnostic profile pointed at scalar
`math.Erf` and `math.Exp`; that profile is motivation, not evidence of a SIMD
speedup. See the retained
[negative experiment](../../internal/benchcompare/leadership/evidence/m2-f64-gelu-direct-20260909/README.md).

The Go 1.27.1 ARM64 `simd/archsimd` API exposes `Float64x2` arithmetic, comparisons,
selection, bitcasts, integer conversion, and ties-to-even `Round` (`VFRINTN`).
Its method names differ from the AMD64 API. The existing ARM64 F64 Softplus
assembly demonstrates a register-resident composite with coefficient-bank
reloads. An intrinsic implementation is a smaller candidate to test before
expanding that assembly surface; its code generation and performance are not
assumed to match the handwritten implementation.

## Decision

Test a two-lane intrinsic implementation under `arm64 && goexperiment.simd`.
Keep Go as the CPU dispatcher and retain the existing output allocation,
contiguous-input conversion, parallel chunking, and reference fallback structure.
No graph, public API, GPU, or scheduler redesign is part of this experiment.

Use a dedicated `vgeluF64Fast` capability for GELU forward and backward. Keep
ARM64's global `vexpF64Fast` false. Default builds, AMD64 arithmetic, F32, and
unrelated operations remain unchanged.

The erf rational regions reuse the existing AMD64 Cephes coefficients: the
small-input rational for `abs(y)<1`, the complementary form for `1<=abs(y)<6`,
and signed one for `abs(y)>=6`, where `y=x*invSqrt2`. The exponential uses the
already verified ARM64 degree-13 reduction and a scalar operation-order twin.
Avoid full-length temporary buffers. Inspect generated code for real two-lane
arithmetic, spills, and unexpected scalar helper calls.

Backward must evaluate its two exponential arguments separately. The rounded
values of `-y*y` and `-0.5*x*x` need not be equal. Sharing their mathematical
identity is not permission to share a floating-point result. A future one-exp
variant would need its own numerical decision and measured evidence.

## Numerical and memory boundaries

Only the opt-in ARM64 SIMD GELU path adopts the existing AMD64-style bound:

```text
abs(candidate-reference) <= 1e-12 * max(1, abs(reference))
```

This absolute floor matters around negative-tail cancellation and derivative
zeros. It is not a universal one-ULP or relative-error claim. Default-build exact
oracles and their one-ULP mutation sensitivity remain intact.

A wrapper call is eligible only when every input is finite with `abs(x)<=32`.
Backward additionally requires every gradient to be zero, or finite with
`1e-150<=abs(g)<=8`. These are implementation bounds, not public input
restrictions. Any failing lane sends the **entire wrapper call** through the
unchanged scalar formula. All finite outputs of that fallback call, including
ordinary companion lanes, must match scalar bits; NaN outputs must match class.

A public operation can contain several parallel wrapper calls. One ineligible
chunk does not make other, eligible chunks scalar-exact. Tests distinguish the
wrapper contract from this public-operation composition explicitly.

Within fully eligible spans, vector-body and odd-tail outputs must have identical
bits. Tests cover both sides of every erf and eligibility boundary, exponential
range-reduction boundaries, signed zero, subnormals, large gradients, nonfinite
inputs, odd lengths, tensor views, and input-storage immutability. Special-value
tests remain active on ARM64 SIMD. Exact destination/input aliasing is supported
at the internal leaf; arbitrary shifted overlap is not a new API contract.
Leaf and wrapper execution must allocate zero heap objects.

## Measurement and promotion

Freeze the same benchmark source into control and candidate binaries built with
the same Go 1.27.1 SDK. Keep direct preallocated leaf measurements separate from
public `Execute` measurements, which include output allocation. Exercise forward
and backward on both active-range and mixed-erf-region fixtures, at 2,048 and
262,144 elements, with `GOMAXPROCS=1` and `12`.

Three isolated, alternating count-seven campaigns must each show, for both large
public operations and both input distributions:

- At least 1.25x serial speedup, with `p<0.05`.
- At least 1.05x parallel speedup, with `p<0.05`.
- No reproducible control slowdown above 3% or allocation increase.

Tiny inputs are diagnostics, not a leadership claim. Record noisy or failing
campaigns rather than selecting favorable samples. Revert an unproven runtime
candidate and retain its tests and evidence. A fresh verifier reruns correctness
checks independently before any production promotion. Neither an internal A/B
win nor an intrinsic instruction listing proves leadership over a pinned
external library or an end-to-end model workload.

## Source provenance and notice

The coefficient source is SciPy's XSF
[`ndtr.h` at commit `4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b`](https://github.com/scipy/xsf/blob/4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b/include/xsf/cephes/ndtr.h),
the submodule pinned by SciPy 1.16.1. Its T/U/P/Q coefficients match GoAI's
existing AMD64 implementation. The source identifies Cephes Math Library
Release 2.2 (June 1992), copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier,
and its 2024 C++ translation by SciPy developers. The new Go code is an
ARM64 intrinsic adaptation, not a verbatim copy of the C++ implementation.

The pinned XSF license is retained below:

```text
BSD 3-Clause License

Copyright (c) 2024, SciPy

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Further reading

- [ADR-0003: backend dispatch](ADR-0003-backend-dispatch.md).
- [ADR-0004: exact-erf GELU](ADR-0004-gelu-exact.md).
- [ADR-0005: CPU/reference separation](ADR-0005-cpu-backend-simd-split.md).
- [ADR-0028: shared NEON transcendental design](ADR-0028-shared-neon-transcendental-leaf.md).
- [Pinned XSF license](https://github.com/scipy/xsf/blob/4fff9b2cb2b5c31a0cf0b0f609d2699a5eeac53b/LICENSE).
- [Shared-leaf perfscan findings](https://github.com/jxsl13/perfscan/issues/917).
