# ADR-0021: SIMD + GPU combination — configurable split, overlap, heterogeneous

Date: 2026-07-14 · Status: accepted (revised same day per user directive C23/C24) · §T629

## Context

Can CPU SIMD (the amd64 archsimd kernels) and a GPU backend (metal / vulkan /
cuda) be combined to push inference throughput past either alone? Three distinct
"combinations" get conflated; they have very different payoffs here.

## Decision

**1. Op-level co-execution (split one op across CPU-SIMD and GPU) — CONFIGURABLE,
off by default (C23).** Splitting a single matmul so the CPU computes some rows
while the GPU computes the rest is what llama.cpp does — but only because the
model does not fit in VRAM. In GoAI's regime *on a device that fits in VRAM* the
CPU delivers ≈40–80 GFLOP/s (SIMD) against the GPU's ≈400–1400, so the CPU's
share of a split op is ≈5 % while it adds a transfer + synchronization barrier at
the split boundary — a net loss (ADR-0008 / §C3, per-op offload of memory-bound
work fails the transfer ≫ compute gate). So the split stays **off by default**.

But per user directive (C23) the *mechanism* is required and configurable — and
it has one legitimate use where it is not about speed at all: **the low-VRAM case
(C24)**. On a device whose VRAM cannot hold the model, offloading the overflow
layers to CPU-SIMD is what makes the model *runnable* — slower than an
all-in-VRAM device, but functional instead of failing. That is the llama.cpp
rationale, and it is a first-class GoAI goal (C24). So: a configurable per-op /
per-layer backend split (T630), off by default, auto-engaged when the model
exceeds VRAM (T631) or when explicitly measured beneficial.

**2. Pipeline overlap (CPU-SIMD host work ∥ GPU forward) — the clean lever.**
The decode loop (`llamagpu.Decoder.Generate`) is currently serial: GPU `Step` →
CPU `SampleWithHistory` → next `Step`. The GPU idles while the CPU samples and
vice-versa. The host-side work (penalties, the truncation samplers, KV
bookkeeping) is CPU-SIMD-eligible and could overlap the *next* step's GPU
forward (§T614 already overlaps command-buffer encoding this way).

Measured host cost per token (`BenchmarkSampleWithHistoryRealistic`,
temp+top-k+top-p+repeat-penalty, V=50257): **≈0.61 ms** — against a ≈4 ms GPU
decode step, so the overlap ceiling is ≈13 %. Real but modest. (This number is
only trustworthy *after* §T629 fixed a quadratic-selection bug that had inflated
it to 7.5 ms — see below; before the fix the host work would have *dominated*
the step and the whole overlap analysis would have been wrong.)

**3. Heterogeneous speculative decoding (CPU-SIMD draft ∥ GPU target) — the
highest-potential lever.** This is the one place SIMD + GPU genuinely multiply: a
small draft model runs on the fast CPU-SIMD path proposing k tokens while the
GPU target verifies them in one batched pass. Draft cost is the lever
(§T434/§T446), and a CPU draft costs the GPU nothing — true different-device
co-execution. `llamagpu.SpeculativeGenerate` exists but runs draft and target on
the same backend today.

## Consequence

The configurable per-op/per-layer split (1) is built as a mechanism (T630), off
by default, whose default use is low-VRAM offload (T631, C24) rather than speed.
Overlap (2) and heterogeneous spec-decoding (3) are measurement-gated follow-ups:
instrument the real decode loop for the host/device split per backend, then build
whichever the data favors — overlap for a large host gap, CPU-draft/GPU-target
for a GPU-bound step with a cheap draft. No heterogeneous executor is built
speculatively (the §V22 discipline: measure the floor before the rewrite). All of
this serves C25: on billions of devices, both the last-percent single-backend
speed *and* the ability to run at all on low-VRAM hardware compound globally.
