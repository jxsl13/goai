---
name: gpu-kernel-runtime-loop-bounds
description: "In GPU kernels, a runtime loop bound over a per-thread array forces it out of registers into memory — costs an order of magnitude and is invisible to inspection."
metadata: 
  node_type: memory
  type: project
  originSessionId: c9844d26-52eb-4368-8358-6d744bfde57b
  modified: 2026-08-15T04:55:14.054Z
---

In a Metal/GPU kernel, `float acc[128]` walked with `for (d=0; d<dk; d++)` where `dk` is a **kernel argument** cannot live in registers. A runtime trip count means dynamic indexing, so the array spills to memory and every element touch becomes a memory access. Compiling the bound as a constant lets the compiler unroll and keep it in registers.

**Why:** measured 2026-08-15. `mha_decode_f32` had an `sk`-independent floor of ~99 µs (dk=64) / ~239 µs (dk=128) — one query row against *eight* keys cost 99 µs. Specializing `dk==64`/`dk==128` with the trip count compiled in gave **5.9–8.7×** on the kernel and 44.3→49.1 tok/s end-to-end (more at long context, since the 5-level simdgroup merge walked all `dk` accumulators at every level regardless of `sk` — that was the whole fixed term).

**How to apply:**
- Suspect this whenever a GPU kernel has a fixed cost that doesn't scale with its input size. It's invisible to inspection — the dimension-agnostic code looks *cleaner* than the specialized version.
- Specialize per concrete dimension and gate on **exact** equality. A first cut gating `dk<=64` with a hardcoded 64 trip count silently computed 64 dims for `dk=32` callers.
- The near-identical wrong hypothesis: array *size*. Shrinking `q[128]`→`q[64]` changed nothing (99.5 vs 99.1 µs). Size was never the issue; dynamic indexing was.
- That null result was only trustworthy because the new kernel was verified **reached** (mutate it to write zeros → tests red). Otherwise "size isn't the problem" is indistinguishable from "my kernel is never called" — see [[self-policing-guard-pattern]] and the Vulkan reachability defect.
- Profiling caveat that hid this for so long: instrumenting a *wrapper* is not instrumenting the path. `flashattn` was counted and read as "attention costs nothing"; the real call went through `MHAAt`→`C.mtl_recorder_mha` directly.
- Measure per [[benchmark-comparisons-must-be-in-session]].
