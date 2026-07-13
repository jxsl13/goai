# GoAI — LLM Frontier Techniques & Performance Roadmap (2026-07-06)

Research + benchmarking scouting pass (user directive: find documentation gaps,
new well-cited LLM techniques worth implementing, and concrete performance
optimizations toward the Go/cgo limit). All external findings verified via the
`research-lite` workflow (never `/deep-research`, per LOOP.md); two focused,
schema-free passes, each CONFIRMED unanimously across three sub-agents. This
document *proposes* §R rows and §T candidates; `SPEC.md` is the authority once
amended.

---

## Part A — LLM frontier techniques worth implementing next

The library already has: RoPE + linear PI + YaRN, GQA/MQA, sliding-window + ALiBi
attention, MoE **gating + load-balance loss** (not full dispatch), DPO/PPO+GAE/
KTO/IPO, Lion/Adam/AdamW, LoRA, speculative decoding, beam search, min-p/top-p/
top-k sampling, GGUF + safetensors, quantized Q8_0/Q4_0 matmul.

Ranked by (value × feasibility) **for this codebase** — feasibility is high where a
candidate reuses parts already built. Every row is a proposed §T task.

| # | Technique | Paper | Builds on (existing) | Effort | Why |
|---|-----------|-------|----------------------|--------|-----|
| 1 | **GRPO** (Group Relative Policy Optimization) | arXiv:2402.03300 | PPO+GAE, advantage norm | Low | Critic-free RL behind DeepSeek-R1; drop the value net, normalize rewards within a sampled group. Highest value×feasibility. |
| 2 | **Full sparse MoE top-2 dispatch** | arXiv:2401.04088 (Mixtral) | MoE gating + balance loss (T61), Linear/SwiGLU | Low-Med | Router gather → per-expert FFN → weighted combine. Completes the MoE we already gate. |
| 3 | **DoRA** (weight-decomposed LoRA) | arXiv:2402.09353 (ICML'24 Oral) | LoRA (T40) | Low | Decompose W into magnitude + direction, LoRA on the direction. Thin wrapper over existing LoRA. |
| 4 | **SimPO** | arXiv:2405.14734 | DPO path (T-DPO) | Very Low | Reference-free, length-normalized reward margin. ~10-line loss swap. |
| 5 | **ORPO** | arXiv:2403.07691 | DPO path, CrossEntropy | Very Low | Single-step SFT + odds-ratio preference loss, no reference model. |
| 6 | **FlashAttention-2 tiling (numerics)** | arXiv:2307.08691 | MHA/SDPA VJP, GQA/SWA/ALiBi | Med | Online-softmax blocked attention, O(N) memory, bit-exact. A pure-Go loop; composes with existing masks. Also an inference-memory win. |
| 7 | **MLA** (Multi-head Latent Attention) | arXiv:2405.04434 (DeepSeek-V2) | RoPE, GQA, KV-cache | Med-High | Low-rank joint KV compression (~93% cache cut) + decoupled RoPE. Higher effort, high value. |
| 8 | **KV-cache eviction** (StreamingLLM sinks + H2O heavy-hitters) | arXiv:2309.17453, arXiv:2306.14048 | KV-cache (T35) | Low-Med | Keep attention-sink + recent / evict low-attention KV. Small add, big long-context win. |

Runner-up: **QLoRA/NF4** (arXiv:2305.14314) — 4-bit NF4 base + LoRA; combines
existing Q4_0 quant + LoRA into a QDoRA path.

**Recommended order:** GRPO → full sparse MoE → DoRA → SimPO+ORPO (batch) →
FlashAttention-2 tiling → KV-eviction → MLA. This front-loads the cheap,
high-reuse training-methodology wins (the user's "alle Trainings-Methodiken"
priority) before the heavier inference-architecture items.

---

## Part B — Documentation-gap audit

After T45, every public package carries a dual-audience `doc.go` + runnable
`// Output:` examples, and every algorithm task cites its §R paper. The remaining
gaps were **not** docs but **measurements**: the repo had per-kernel Go benchmarks
(vs its own reference) but **no cross-language baseline** against the Python
libraries it targets for parity. Part C closes that gap.

---

## Part C — GoAI vs PyTorch: measured GEMM baseline (NEW)

New harness: `backend/cpu/gflops_bench_test.go` (Go, reports a `GFLOP/s` custom
metric) and `testdata/bench_torch.py` (torch, same sizes/dtypes). Dense N×N matmul
= 2·N³ flops. Numbers below are indicative darwin/arm64 (Apple M2 Pro), torch
2.12.1 (Accelerate), Go 1.26; treat committed CI runs as the source of truth.

| dtype · N | GoAI cpu (GFLOP/s) | PyTorch (GFLOP/s) | GoAI / torch |
|-----------|-------------------:|------------------:|-------------:|
| f64 512   | 60.6  | 660.3  | 9.2% |
| f64 1024  | 69.3  | 683.6  | 10.1% |
| f32 512   | 58.5  | 2312.8 | 2.5% |
| f32 1024  | 70.4  | 2734.7 | 2.6% |

**Reproduce:**
```sh
go test ./backend/cpu -run '^$' -bench 'GEMM.*gflops' -benchtime=1s
.venv/bin/python testdata/bench_torch.py
```

**Two findings drive the roadmap:**

1. **GoAI GEMM sits at ~10% of PyTorch on f64** — consistent with the documented
   pure-Go ceiling (gonum ≈ 1/10 of OpenBLAS without asm microkernels). PyTorch on
   this host uses Apple's Accelerate/AMX matrix coprocessor, which pure Go cannot
   reach; the realistic pure-Go target is a larger fraction of *scalar* peak, not
   of AMX.
2. **GoAI f32 ≈ f64 (~70 GFLOP/s), but torch f32 is ~4× its f64.** Our kernel does
   not exploit f32's 2× SIMD lane width — the single biggest, most portable
   pure-Go win available (no AMX needed). This alone should roughly double f32
   throughput.

---

## Part D — Performance optimization roadmap (toward the Go/cgo limit)

Verified strategy ladder (BLIS/Goto model; Van Zee & van de Geijn IPDPS'14;
BLISlab arXiv:1609.00076; Williams et al. *Roofline*). Each step must show a
benchmark delta and stay bit-identical to the ref backend (§V3/§V5), and cgo is
only considered after the pure-Go path is optimized to its ceiling and still loses
by the §C3 threshold (§V-CGO) — proposed §T-optimization tasks:

1. **Register + cache blocking (BLIS 5-loop).** Pack A into L2 (mc×kc) panels and
   B into kc×nc panels for contiguous streaming; an mr×nr register-blocked
   microkernel reused from registers. Current cpu GEMM is 4-row-blocked ikj
   (§T12b); moving to a packed BLIS structure is the foundation for everything
   below. Expected: naive/register-tiled reaches ~50-60% of *scalar* peak.
2. **SIMD microkernel.** Two portable paths: `avo`-generated AVX2/AVX-512 `.s`
   kernels (~3× scalar, the mature route gonum uses), or Go 1.26's experimental
   `GOEXPERIMENT=simd` `simd/archsimd` (Float64x8/AVX-512, amd64-only, inlinable,
   ~30% faster than non-inlined avo, >4× with inlining; API unstable). This is the
   step that lifts f32 to its 2× lane advantage. NOTE: amd64-only today → keep
   behind build tags with the scalar fallback green (ties to the parked T11b).
3. **Bandwidth-bound fusion.** softmax / layernorm / elementwise / attention sit
   under the memory roof — fuse passes, keep access contiguous, and reuse
   arena/pool buffers (no per-element allocation; the ref kernels' `Unravel`
   per-element alloc is exactly what the cpu backend removes). Roofline: classify
   each kernel by (runtime, flops, bytes) to know which roof it hits.
4. **Amortize the cgo boundary.** For the GPU/BLAS-cgo backends, batch calls and
   use staging/contiguous buffers so the boundary cost amortizes over large ops
   (the `blase` queue pattern) — consistent with ADR-0008 (GPU only for
   compute-bound / graph-resident work).

**Realistic ceiling (honest):** a from-scratch Goto/BLIS-style kernel with
packing + SIMD microkernel + threads can reach ~90-106% of *OpenBLAS* on x86 — but
that requires asm microkernels, and on Apple silicon the AMX-backed Accelerate
number (2700+ GFLOP/s f32) is not reachable from portable Go. The pure-Go goal is
therefore: **close the f32 SIMD-width gap (≈2×), adopt BLIS blocking, and document
the residual AMX/AVX-512 gap as a cgo-gate candidate** rather than chasing peak.

---

## Part E — Transferable patterns from Go perf/GPU libraries

- **gonum** — a Go BLAS *interface* with amd64 asm kernels, switchable to a cgo
  netlib (OpenBLAS) backend; the win is amortizing/minimizing boundary crossings.
  Mirrors GoAI's backend-registry + auto-select design (§T46).
- **gorgonia** — graph-level fusion over gonum-native BLAS; validates the
  fused-op + tape approach GoAI already uses.
- **blase** — batches cgo BLAS calls through a queue to amortize the boundary
  cost. Directly transferable to the Metal/Vulkan/CUDA backends.
- **Ebiten / Gio** (GPU-heavy) — batch draw calls, minimize GPU state changes,
  reuse buffers. The transferable lesson for our GPU backends: batch kernel
  dispatches and keep tensors GPU-resident (again ADR-0008's graph-resident rule).

---

## Proposed SPEC amendments (handed to /spec)

- **§R67** — pure-Go performance ceiling & BLIS/Goto GEMM ladder (CONFIRMED).
- **§R68** — 2024-25 LLM frontier technique ranking for this library (CONFIRMED).
- **§T candidates** (status `.`): GRPO, full sparse MoE dispatch, DoRA, SimPO+ORPO,
  FlashAttention-2 tiling, KV-cache eviction, MLA; plus a GEMM-optimization task
  (BLIS blocking + f32 SIMD microkernel, cgo-gate per §V-CGO).
