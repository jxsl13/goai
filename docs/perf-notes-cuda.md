# CUDA optimization playbook: generic rules for backend/cuda

Battle-tested rules for hand-written CUDA kernels in `backend/cuda`, derived from
the measured probes recorded in `SPEC-worker-linux-amd64-cuda.md` (§Tw*, every
one A/B'd per §V22). Target: **NVIDIA Ampere sm_86 (RTX 3060, 12 GB, ~360 GB/s
DRAM), CUDA 12.9 via NVRTC + inline PTX only — no `nvcc`, no `ncu`/`nsight`.**
The no-profiler constraint shapes everything below: we diagnose by *timed
stub-probes*, not by counters.

This is the reference for *all* future optimization attempts. Read the diagnosis
section first — **the most common failure is optimizing the wrong resource.**

---

## 0. The prime directive — measure-first, always (§V22)

Reinforced 13× in our logs, every time it was skipped it cost a wasted build:

- **Never assume the bottleneck. Measure it, then attack only the binding
  resource.** dp4a (Tw58), split-K (Tw56), warp-spec (int8-GEMM 2×), flash-prefill
  (Tw64) were all *correct* builds that *regressed or flatlined* because they
  attacked a resource that wasn't binding.
- **A/B interleaved (A,B,A,B across ≥5 reps), same process, same window.** Decode
  throughput drifts; a single before/after is noise. Report the mean and the
  per-rep spread.
- **Match the reference's ARITHMETIC exactly, not just its math.** An f64-division
  reference vs an f32-reciprocal-multiply kernel flips roundings and masquerades
  as a 1–4% "kernel bug" (cost us a day on W8A8 until an exact-int probe cleared
  the pipeline). Mirror the kernel's operations in the host ref.
- **Non-wins are reverted (§C3) and recorded as a rejection with data.** A logged
  rejection ("dp4a flat because the multiply isn't the bottleneck") is worth as
  much as a win — it closes a lever permanently.

---

## 1. Diagnose the binding resource without a profiler (the stub-probe method)

**Framework — the roofline (Williams et al., CACM 2009).** Every kernel sits at an
*arithmetic intensity* AI = FLOPs ÷ bytes-moved. The hardware has a *ridge point*
(peak-FLOP/s ÷ peak-GB/s); below it you are memory-bound, above it compute-bound.
Batch-1 quantized GEMV has AI ≈ 0.5–1 (each weight byte is read once) → it is
**intrinsically DRAM/transaction-bound — no kernel cleverness changes that, only
moving fewer bytes does.** Tensor-core prefill GEMM has high AI → compute-bound.
Knowing the AI *before* you start tells you which half of this doc applies.

**Canonical attack order (CUDA C++ Best Practices):** (1) maximize achieved
bandwidth, (2) fix memory access patterns/coalescing, (3) *only then* chase
ALU/occupancy. Treat a kernel as memory-bound until achieved-GB/s ≈ peak; don't
optimize instructions while you're still leaving bandwidth on the table.

We cannot run `ncu`. Instead, **mutate the kernel to remove one resource's cost
and re-time**, and compute *effective* GB/s (bytes read+written ÷ elapsed) and
GFLOP/s against the hardware peaks (~360 GB/s; int8/f16 TC roof). This is the
profiler-free form of the Best-Practices "effective bandwidth" method. The delta
tells you what was binding:

| Probe | How | What a *flat* result means | What a *faster* result means |
|---|---|---|---|
| **Memory-floor** | Keep all loads, stub the compute (accumulate a cheap function of the loaded bytes) | you are already at the bandwidth/transaction floor — compute is free | compute (ALU) is the limiter; the gap = ALU headroom |
| **Compute-floor** | Keep the math, replace global loads with a constant | you are ALU/ILP-bound | loads are the limiter (bandwidth or transactions) |
| **Coalesce probe** | Repack the bytes to 16-aligned + read as `uint`/`float4` | layout was fine | you were transaction-bound (uncoalesced), not byte-bound |
| **Occupancy probe** | Fold a starved small-N launch into a larger one (or add/remove warps) | the shape was not latency-starved | you were latency/occupancy-bound at small N |

Worked example (Tw56/Tw58): Q4_K decode GEMV real 167–219 GB/s; memory-floor probe
(loads intact, dequant stubbed) → 285–330 GB/s (79–92% of peak). Conclusion: the
kernel is **~1.5× off the bandwidth floor purely on dequant ALU** → the lever is
the scale-decode ALU, *not* more bandwidth or more warps. That one probe killed
both the dp4a and the split-K ideas before we shipped them.

### The five resource classes (know which one you're in)

1. **DRAM bandwidth** — bytes moved ÷ time approaches ~360 GB/s. Only real at
   large-N, ALU-light GEMV. Lever: move fewer bytes (lower-bit quant).
2. **Memory transactions** — *count* of memory transactions, not bytes. Small,
   un-16-aligned blocks force 1-byte uncoalesced reads → up to ~8× the
   transactions at identical bytes. Lever: repack + coalesce (see §2).
3. **ALU / instruction throughput** — the math (scale-decode unpack, shfl) can't
   keep the load units fed. Lever: fewer ops per element, or hardware
   instructions (ldmatrix, dp4a *only if the multiply is actually the cost*).
4. **Occupancy** — too few resident warps to hide latency, usually from big
   tiles exhausting registers/shared on sm_86. Lever: *smaller* tiles, fold
   starved launches together.
5. **Launch / latency** — kernel-launch and host-sync overhead dominates at
   batch=1. Lever: CUDA-graph capture (collapses per-op launches), on-device
   argmax, elide logit D2H.

---

## 2. Memory-bound quantized GEMV (decode, batch=1) — rules

The decode path is **weight-bandwidth-bound by construction** (one activation
vector, a huge weight matrix). Rules, strongest first:

- **R1 — Transactions, not bytes (§Tw54, §Tw72).** Two quants with identical
  bytes/weight can differ 2–2.7× in decode speed purely on memory-transaction
  count. 32-element blocks (Q4_0/MXFP4, 18/17 B) that aren't 4-aligned force
  1-byte reads; a Q4_K super-block (256 elem) issues ~8× fewer transactions.
  *Hardware reason:* since Pascal, sm_86's L1 services global memory at **32-byte
  granularity** — a misaligned/strided quant block fragments into extra 32 B
  transactions independent of the bytes actually needed. **FIX = upload-time
  repack in the resident constructor**: split each row into a scale region + a
  **16-aligned** nibble/index region so the GEMV reads coalesced `uint`s. Measured:
  Q4_0 2.37×, MXFP4 2.93×, both then *beat* Q4_K.
- **R2 — Vectorize the activation reads (`float4`).** Coalescing the *weight*
  nibbles alone got only +8%; the bulk of the repack win came from making the
  *activation* reads `float4` (they were strided scalar). Always load activations
  as `float4`/`float2`, 16-aligned.
- **R3 — Occupancy-cliff law: fusion/parallelism pays ∝ starvation.** A GEMV at
  small N is *latency*-bound (GQA k/v N=256 ran at only 17% of peak); folding it
  into a larger launch (fused QKV) lifts those starved warps → +3.7%. The same
  fusion on an unstarved shape (gate/up N=5632 = 55% of peak) gave only +1.1%.
  **Corollary:** before fusing/parallelizing, measure the shape's % of peak — if
  it's already >~50%, the occupancy lever is nearly tapped.
- **R4 — Adding warps (split-K) only helps latency-bound shapes.** On ALU-bound
  FFN shapes (5632/2048), split-K *regressed monotonically* (more warps-in-flight
  doesn't raise bandwidth when the limiter is per-lane ALU). Split-K is **not** an
  assumed win — measure first.
- **R5 — Scale-decode ALU is the Q-K-quant decode ceiling.** Q4_K/Q5_K/Q6_K
  decode is limited by per-block scale unpack (f16 d/dmin + 6-bit unpack + shfl),
  *not* the multiply. So `dp4a` (which cuts multiply cost) is flat, and the only
  real levers left are **lower-bit quant** (fewer bytes) or **pre-decoding scales
  at upload** (trade weight bytes for no in-kernel unpack). Confirmed 3× (Tw44,
  Tw56, Tw58).
  - **Worked example — the pre-decode probe (`ResidentBQ4KPre`, R5 in action).** We
    repacked Q4_K to carry the 8 sub-block scales as f32 (c1=d·sc6, c2=dmin·min6) so
    the GEMV drops the `get_sm` unpack + f16 decode + 2 shfls entirely — **bit-exact**
    (rel 0.00e+00), at the cost of +33% weight bytes. The A/B is a textbook R1-vs-R5
    collision and the result maps *exactly* onto the theory: it **wins where the
    shape is not bandwidth-bound** (k/v 2048×256 −16.7%, the occupancy-cliff-starved
    shape from R3; down 5632×2048 −5.2%, large-K = the most scale-decode ALU) and
    **loses where it is** (q/o 2048×2048 +2.0%, gate/up 2048×5632 +3.1% — the +33%
    bytes dominates). Per-shape routing (pre for starved-N + large-K) nets ≈ −2.5% on
    the *isolated GEMV subset*. **BUT the e2e decode A/B was only +0.3%** (253.4 vs
    252.8 tok/s, TinyLlama-Q4_K_M, token-identical) — the k/v+down GEMVs are too small
    a slice of the decode *step* (attention, RoPE, norms, KV append, sampling, launch
    overhead) for their −5–17% to move the needle. **Two lessons:** (1) an ALU-relief
    transform on a bandwidth-bound path only pays where bandwidth isn't the binder;
    (2) *always close the loop with an END-TO-END A/B* — an isolated-kernel win is
    diluted by everything else in the step, so a −2.5% GEMV subset became a +0.3%
    (noise) decode. The pre-decode is kept **opt-in, not defaulted** — a rule-validating
    probe, not a shipped speedup. This is the R5 mirror of Tw56/58: measure-first,
    then measure *again* end-to-end.
- **R6 — Codebook placement for i-quants.** Small (≤16-entry, IQ4/MXFP4) → inline
  as a kernel `const` array. Large grids (256×8, 512×4 — IQ2/IQ3) → upload once to
  a **shared device buffer** (reconstruct host-side via the public
  `gguf.Dequantize`, no gguf internals). **Never** put a divergently-indexed LUT
  in `__constant__` (it serializes per-lane divergent reads — 75 µs vs 28 µs) or
  in-kernel `const` for a big table (spills to per-thread *local* memory). Use
  `__device__ const` (L1-cached global) when you must.
  - **The GATHER, not the bytes, dominates i-quant decode — grid → shared (Tw80).**
    Codebook i-quant GEMV (IQ2/IQ3) ran 2–2.5× *slower* than Q4_K *despite fewer
    bytes* → not bandwidth-bound. A stub-probe (remove the `grid[idx]` gather, keep
    the byte reads) → 2.7× faster, pinning the cost on the **random per-lane grid
    gather**, not the byte-assembled block reads — so the R1/Tw72 repack would have
    been the *wrong* fix (measure-first caught it, again). FIX: load the small
    codebook grid into **shared memory** once per block (cooperative load +
    `__syncthreads`, placed *before* the early return so the sync can't deadlock),
    then gather from shared. Measured **~1.5×** (not the stub's 2.7× — shared has
    bank-conflicts on the random gather), bit-exact. Grids fit shared (IQ2 8–16 KB,
    IQ3 4–8 KB); IQ1's 64 KB needs the sm_86 >48 KB opt-in (occupancy tradeoff to
    measure, or pack the ternary grid to 2 bits → 4 KB and decode on lookup).
    *Ceiling (measured):* ~1.5× is the practical ceiling for the shared approach. A
    second stub (force all lanes to grid row 0 — a *broadcast*, conflict-free) hit
    the gather-free floor (12157 vs 21574 ns), proving the residual cost is **shared
    bank conflicts**. But row-stride padding to dodge them (stride-4 → coprime
    stride-5) gave *nothing* (21414 ≈ 21574): each lane reads 8 grid floats (2 rows ×
    4), so 32 lanes = 256 accesses into 32 banks = an ~8-way conflict floor that
    padding can't move — only the broadcast special-case is free, and a random
    codebook gather can't broadcast. Lesson: **for a random codebook lookup, the
    per-lane-float-count ÷ 32 is a hard bank-conflict floor; padding only helps
    when reads are *contiguous/strided*, not randomly gathered.**
    - **Escape route — shrink the per-lane read (Tw80 follow-up for IQ1).** The floor
      is *per-lane-floats ÷ 32*, so the only way under it for a random gather is to
      read fewer bytes per lane. IQ1's grid is **ternary** `{−1,0,+1}`, so pack it to
      **2 bits/entry** (2048×8 → 4 KB, also dodging IQ1's 64 KB occupancy wall) and
      decode `code→{−1,0,+1}` in cheap ALU on lookup. Each lane then reads one
      `uint16` (2 B) instead of 8 f32 (32 B) → the bank floor drops ~16× (256→16
      accesses/warp), potentially *beating* the 1.5× the f32-grid path is stuck at.
      Only works when the codebook is low-entropy enough to pack (ternary/2-bit);
      IQ2/IQ3 store real f32 values and can't. Booked, measure-first, post-IQ1-merge.

---

## 3. Compute-bound tensor-core GEMM (prefill) — rules

- **R7 — `ldmatrix` for fragment loads.** Replacing manual 4-int-load +
  byte-assembly with hardware `ldmatrix.x4`/`x2` freed the ALU that was competing
  with the per-block scale epilogue → +14% on the per-row MMQ (a bigger win than
  the raw GEMM's +4%, because the epilogue ALU pressure was higher). Bit-identical
  fragments, so it's a free win.
- **R8 — Big tiles / warp-specialization REGRESS on sm_86.** A correct
  warp-specialized 64×128 int8 GEMM (4 producer + 8 consumer warps) came in ~16%
  *slower* than a modest ldmatrix tile — sm_86's limited registers/shared cap
  occupancy, and every big-tile variant we tried lost. The hand-NVRTC int8 GEMM
  ceiling is ~22–23 TOPS (beats cuBLAS-f16 by ~7%, ~22% of int8 peak); the 2×
  needs CUTLASS-level ptxas tuning we can't reach from NVRTC. **On sm_86, prefer
  smaller tiles + higher occupancy over big tiles + reuse.** (This is the practical
  face of Volkov, GTC 2010, *"Better Performance at Lower Occupancy"*: occupancy
  only needs to be *enough* to hide latency, per Little's law, and past that
  registers/ILP per thread can beat more warps — but the *converse* bites here,
  because a big tile that spills or over-allocates shared drops occupancy below the
  latency-hiding threshold and there's no ILP headroom on NVRTC-generated code to
  compensate.)
- **R9 — A quantized-GEMM epilogue is only free when *truly* fused.** Int8 GEMM at
  ~2× f16 rate is *strictly dominated* if the i32→f32 dequant is a *separate*
  bandwidth-bound pass over [M,N] — it eats the rate advantage (W8A8 was only
  1.09× vs f16's 1.65×). Fuse the epilogue into the GEMM (cublasLt epilogue) or
  don't bother.
- **R10 — Attention at seq≤1024/hd=64 is GEMM-compute-bound, not memory-bound.** A
  scalar online-softmax flash kernel (even tiled, no score materialization) ran
  ~2.8× *slower* than the materialized cuBLAS-batched path — beating a hand kernel
  needs tensor-core MMA (WMMA), which NVRTC-without-`mma.h` blocks. f16 tensor
  cores on the attention GEMMs also gave nothing (K=hd=64 too skinny to fill MMA
  tiles). Don't hand-roll attention flash on this stack.

---

## 4. Launch/latency at batch=1

- **R11 — CUDA-graph capture is the big batch=1 lever (3.6×).** The unified
  per-op-submit path was 44 tok/s vs 161 tok/s for the same weights under graph
  capture + on-device argmax + elided logit D2H. Per-op kernel launches and a
  full-[1,V] D2H every token dominate; collapse them.

---

## 5. When to STOP (ceiling awareness)

Optimization has a Pareto frontier; recognize it and log it rather than grinding
a closed lever:

- If the **memory-floor probe** shows you within ~10–20% of peak bandwidth and the
  remaining gap is ALU that's *fundamental* to the format (scale-decode), you're at
  the ceiling — the only move is a *different format* (lower-bit), not a better
  kernel.
- If **every big-tile/warp-spec variant regresses**, you're occupancy-capped by
  the hardware; stop reaching for the textbook 2× (it assumes a bigger register
  file / ptxas tuning you don't have from NVRTC).
- A closed lever is a *deliverable*: record it as a PERF-* rejection with the
  number and the reason, so no future fire re-treads it.

---

## 6. Reusable mechanisms (not perf, but reused every time)

- **Reconstruct a device codebook via the public API.** To get a quant format's
  grid onto the device without touching gguf internals: craft blocks with a known
  scale + all-positive signs, run the *public* `gguf.Dequantize`, and invert
  (`grid[idx][k] = dequant/db`). Uploaded once behind `sync.Once`. Used for every
  i-quant grid (IQ2/IQ3).
- **Validate a new kernel by parity vs `gguf.Dequantize` + a host matmul**, rel
  tolerance ~1e-5 (only f32 summation-order deviation allowed, ~1e-7 typical;
  assert *relative to output magnitude* since random scales inflate weights).
- **CI/throughput A/B evidence lives in `docs/benchmarking.md`; every rule here
  traces to a §Tw row in `SPEC-worker-linux-amd64-cuda.md`.**

---

## 7. External cross-check (research-lite, CONFIRMED)

A schema-free web-research pass (`.claude/workflows/research-lite.js`, 3 agents +
synthesis, primary NVIDIA / roofline sources) **corroborated every locally-measured
rule above** against independent authority. Mapping:

| Our rule (measured on sm_86) | External confirmation |
|---|---|
| §1 "which resource" | **Roofline model** — AI vs machine balance / ridge point (Williams et al., CACM 2009) |
| §1 stub-probe timing | *Effective-bandwidth* method — GB/s & GFLOP/s vs peak (CUDA C++ Best Practices §Effective Bandwidth) |
| §1 attack order | "maximize BW → fix coalescing → then ALU/occupancy" (CUDA C++ Best Practices) |
| R1 transactions-not-bytes | **32 B L1 transaction granularity** since Pascal (NVIDIA *Access Global Memory Efficiently*) |
| R2 float4 loads | 128-bit vectorized loads cut *both* transactions and instruction count (NVIDIA *Vectorized Memory Access*) |
| R3/R4 occupancy-cliff, split-K | Little's-law latency-hiding; occupancy rarely the true binder (Volkov, GTC 2010) |
| R7 ldmatrix + cp.async | ldmatrix→mma.sync + multistage cp.async double-buffer (CUTLASS *efficient_gemm*) |
| R9 fused epilogue | fuse dequant+activation to kill global round-trips (CUDA C++ Best Practices §Fusion) |

**One place our measurement REFINES the textbook (keep it in mind):** the generic
advice is "put read-only lookup tables in **constant memory**." That is correct
only for *uniform* (warp-broadcast) access. Our i-quant codebooks are indexed by a
**divergent per-lane** value, and `__constant__` *serializes* divergent reads (we
measured 75 µs vs 28 µs). So R6 stands: divergently-indexed LUTs go in
`__device__ const` (L1-cached global) or shared, **never** `__constant__`. General
rules assume uniform access; always check your access pattern against the
assumption before applying them.

**Net:** the roofline is the map, the stub-probe is our compass without a
profiler, and the transformation catalog (§2–§4) is the terrain we've actually
walked. Nothing in the external literature contradicted a measured result; it
only supplied the *names* and the *hardware reasons* for what the probes found.

---

## 8. Where GoAI actually loses vs llama.cpp — aim here, not at decode

Optimization effort is only worth spending where there's a real gap. Measured
standing on this box (RTX 3060) vs llama.cpp-Vulkan:

- **Decode (token generation): ≈ parity / ahead.** GoAI ~253 tok/s (TinyLlama-Q4_K_M,
  graph-captured) vs llama.cpp ~246. The decode path is graph-captured + on-device
  argmax + native-quant GEMVs at their Pareto ceiling. **Do not grind decode** — it's
  won. (The Q4_K pre-decode probe in §2/R5 re-taught this: a −16.7% isolated k/v GEMV
  became +0.3% e2e because the GEMVs are a small slice of an already-tight step.)
- **Prefill (prompt processing): 0.36–0.54× — THIS is the gap.** GoAI-f16 0.54×,
  GoAI-MMQ int8 0.36× of llama.cpp-Vulkan. Prefill is GEMM-compute-bound (FFN GEMM
  ~54% of the step, attention ~14%).

So the "beat the incumbent" lever is **prefill GEMM throughput**. But note what's
already been closed there (§3): the int8 tensor-core GEMM is at its hand-NVRTC ceiling
(~22–23 TOPS, beats cuBLAS-f16 by ~7% but ~22% of int8 peak; the 2× needs CUTLASS-level
ptxas tuning), warp-spec/big-tiles regress on sm_86 (R8), and flash-prefill + f16
tensor-core attention both lost (R10). **The prefill gap is HW/tooling-limited**: closing
it needs `mma.h`/WMMA (blocked on the NVRTC-without-mma.h wall) or a CUTLASS-class kernel
— not another hand-NVRTC micro-opt. Until that wall moves, prefill is at its practical
ceiling too.

**Bottom line for future fires:** both the decode and prefill frontiers are
characterized and at their hand-tool ceilings on this stack. New wins require either
(a) a *capability* GoAI lacks (new quant format, longer-context path) rather than a
speed micro-opt, or (b) a toolchain change (mma.h access, a newer Go with faster
int-vector codegen for the amd64 SIMD side). Don't re-grind a characterized ceiling —
§0 and §5 exist to stop exactly that.
