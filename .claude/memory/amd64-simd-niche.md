---
name: amd64-simd-niche
description: "amd64 SIMD GEMM (GOEXPERIMENT=simd/archsimd) is my exclusive lane — ARM M2 can't compile it; state + headroom"
metadata: 
  node_type: memory
  type: project
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

The `GOEXPERIMENT=simd` archsimd/AVX F32+F64 GEMM path in `backend/cpu/gemm_simd.go`
(build tag `amd64 && goexperiment.simd`) is **exclusively mine** — the ARM M2 dev machine
cannot compile it, so I'm the only machine that can build/test/optimize/benchmark it.
Build/run with `GOEXPERIMENT=simd go test/build ./backend/cpu/`. There is a soft `simd`
CI job (ubuntu+windows) that also exercises it.

State (as of PR#81, 2026-07-15):
- GEMM microkernel `gemmF32BandDirect` (m≥4): mature — 4×16 FMA tile, B-reuse, BCE,
  L2 col-blocking. ≈154 GFLOP/s @256, ≈230 @512. Near the practical Go-intrinsics ceiling;
  beating it is high-effort/uncertain — don't force it.
- **Decode GEMV (small-m, m≤3)** `gemmF32SmallM` + its dispatch: FIXED in PR#81 — the
  column-block size now adapts to ≈n/workers (was fixed 512 → only 4 workers for n=2048).
  Bandwidth-bound; 14→27.5 GFLOP/s (+90%) @[1,2048]·[2048,2048]/16 cores.
- **Medium-m GEMM (m=4..~48)**: FIXED in PR#83 — the m≥4 dispatch now grains over 4-row
  TILES (not rows) + splits columns when tiles<workers, so every core runs a full 4-row
  B-reuse tile. Was row-parallel → workers with 1-3 rows hit the slow single-row remainder.
  2.5-4.9× (m=4 15→73, m=32 35→101, m=48 38→119 GFLOP/s). m=512 GEMM + m=1 GEMV unchanged.
  Covers batched/speculative decode + chunked prefill. Method: profile the m-sweep with a
  temp bench (found cliffs at m=4 and the row/tile crossover ~m=64) — that A/B approach is
  the reliable way to find these.
- No AVX-512 on this box (avx2+fma only) → Float32x8 (256-bit) is the max width.
- Both the GEMV and medium-m GEMM gaps are now closed; the microkernels + dispatch are
  mature. Further headroom is shape-dependent/fragile — don't force it without a measured
  multi-shape A/B (column-parallel loses on large-k; row-parallel spiky on m%4).

NEW HIGH-VALUE GAP (2026-07-16): CPU QUANTIZED inference is SLOW. gguf.QMatMul (the CPU
quant matmul, format/gguf/quant_matmul.go) DEQUANTIZES each weight row to f32 then scalar-
dots — no SIMD, allocates per row. Baseline BenchmarkQMatMulQ8_0_M1 ~598 MB/s. This is
llama.cpp's core CPU strength (hand-tuned AVX int8 dot kernels) and a real goai gap in MY
exclusive niche (amd64 SIMD quant GEMV). FIRST ATTEMPT (2026-07-16, DISCARDED): wrote a
SIMD Q8_0 GEMV with the float-convert approach — per 8 weights: LoadInt8x16SlicePart →
ExtendLo8ToInt32 → ConvertToFloat32 → MulAdd(x_f32). Parity PASSED (maxRel ~3e-4, summation
order) but **7× SLOWER than scalar** (87 vs 598 MB/s @2048×2048). Since archsimd F32 is fast
(mature GEMM 154-230 GFLOP/s), the bottleneck is the int8→int32→float32 CONVERSION ops
(ExtendLo8ToInt32/ConvertToFloat32) + partial loads, NOT archsimd generally. 3-FIRE CHARACTERIZATION (2026-07-16, all DISCARDED — direction is HARD, deferred):
(1) float-convert SIMD Q8_0 GEMV (ExtendLo8ToInt32→ConvertToFloat32→MulAdd): 7× SLOWER than
scalar (595→87 MB/s) — the int→FLOAT conversion ops dominate.
(2) RAW int8-dot microbench (LoadUint8x16Slice.DotProductPairsSaturated [VPMADDUBSW] +
Int16x8.DotProductPairs, NO scales, reduce once/row): **13.7× FASTER** (1480→20331 MB/s).
So archsimd int-dot ops (VPMADDUBSW/VPMADDWD) ARE fast in isolation, and inline fine.
(3) FULL Q8_0 int-dot GEMV (quantize activation to int8/block + ExtendToInt16→
Int16x16.DotProductPairs + per-block wscale·xscale, float32x8 accumulator, reduce once/row):
CORRECT (norm-rel-RMS 5.5e-3 vs gguf.QMatMul) but **0.79× scalar (SLOWER, ~460 MB/s)** —
~44× slower than the raw int-dot microbench. The killer is the Q8_0 PACKAGING over 131k
blocks (nb·n): per-block f16le decode + int32→float convert + scale-broadcast + the
ExtendToInt16 widen (int16 path, needed for signed-correctness — the fast VPMADDUBSW is
uint8×int8). Inlining is FINE (checked -gcflags=-m); it's raw op count + packaging.
VERDICT: matching llama.cpp's CPU quant kernels in Go 1.26 archsimd is a HARD, DEDICATED
codegen effort (not a quick win) — the isolated int-dot is fast but the correct-Q8_0
packaging drags it below scalar. To beat scalar likely needs: the VPMADDUBSW path (16 int8/op
vs the int16 path's 8, w/ the +128 uint8 offset + Σw correction term), a much cheaper f16
decode (precompute block scales once, or a table), and minimizing per-block scalar glue.
DEFER unless a dedicated CPU-quant push is warranted. The scalar gguf.QMatMul (~595 MB/s) is
the current CPU quant path; the f32 SIMD GEMM (mature) is unaffected. NEXT DIRECTION: rather
than grind this, consider the f32 SIMD last-percentile OR a fresh CUDA/nn gap.

*** ROOT CAUSE FOUND (2026-07-16) — archsimd INT ops are pathologically slow in Go 1.26 ***
This explains BOTH failed directions (quant GEMV AND transcendentals). PROOF (vexp probe,
discarded): SIMD exp8 (Cephes poly) parity-EXACT vs scalar expF32, and the FLOAT-only poly
is **7.3× FASTER** than scalar (1369 vs 10024 ns @2048). But adding the 2^n step —
`ConvertToInt32().Add(127).ShiftAllLeft(23).AsFloat32x8()` (4 int/convert ops) — balloons it
to 19590 ns = **2× SLOWER than scalar**; the int-op 2^n costs ~14× the entire float poly.
Same root cause as the quant GEMV (int8→int32→float / ExtendToInt16 conversions were slow).
CONCLUSION: Go 1.26 `simd/archsimd` FLOAT FMA ops are fast (→ the mature f32 GEMM at
154-230 GFLOP/s, and 7.3× on a float poly), but the INT-vector ops (ConvertToInt32,
ShiftAllLeft, ExtendToInt16/Int32, AsFloat32x8, etc.) are ~14-60× too slow (likely
non-inlining or codegen fallback — a single VPSLLD/VCVTPS2DQ should be ≈1ns, measured ≈18ns).
IMPLICATION: the amd64 SIMD niche is AT ITS CEILING on Go 1.26 for the remaining workloads —
quantized GEMV (needs int8 dot + int→float) AND SIMD transcendentals/activations (exp/gelu/
silu need 2^n int-bit-assembly) are BOTH blocked. Only pure-float-FMA kernels win, and the
GEMM (the big one) is already done. DON'T re-attempt quant or vexp on archsimd until a newer
Go improves int-vector codegen (worth re-checking each Go release; also worth a Go issue
report: archsimd int-vector ops pathologically slow vs float). Workaround for exp specifically
would need a float-only 2^n (none available). Niche is mature/ceiling — pivot to CUDA/other.

CEILING RE-CONFIRMED (2026-07-16 ~12:20Z, this box Ryzen 7 5700G Zen3 avx2+fma, GOEXPERIMENT=simd):
BenchmarkGEMM_F32 gflops — 512² 177-208, 1024² 202-223 GFLOP/s. SINGLE-CORE (GOMAXPROCS=1) 1024² =
**33.45 GFLOP/s = ≈23% of the ≈147 single-core peak**; scales 5.8× to 8 cores (194.5) + SMT to ≈223
@16. The 8-accumulator gemmF32BandDirectCols kernel (4 rows × 2 Float32x8, 8 FMA chains, L2-blocked)
is well-blocked; 512→1024 gets FASTER (177→223) = overhead-amortizing, NOT memory-bound → so the
strided-B access (bo+=n) is NOT the limiter and B-PANEL PACKING would NOT help (a codegen-bound kernel
gains nothing from better memory layout). This is the Go-1.26-archsimd FLOAT-FMA codegen ceiling
(~23% single-core efficiency), consistent with the root-cause note above. DON'T re-attempt GEMM
micro-opt (B-packing, wider blocking) — confirmed dead. LESSON: check THIS memory before re-profiling
the amd64 GEMM; it's characterized. Niche is CEILING → pivot to CUDA. The tractable "beat llcpp"
frontier is CUDA PREFILL (GoAI-f16 0.54×, GoAI-MMQ 0.36× llcpp-Vulkan) — needs prefill overhead
profiling (GEMM-vs-attention-vs-launch breakdown) to decide graph-capture vs ldmatrix; see
[[cuda-q4-arc-state]].

CEILING RE-RE-CONFIRMED (2026-07-16 ~16:05Z): independent size-sweep 256/512/1024/512×2048×2048/
512×2048×8192 → 143/223/230/205/206 GFLOP/s — FLAT (even at B=64MB ≫ L3), so NOT cache-bound; matches
the ~23%-of-peak codegen-ceiling finding above. The SPEC-worker Nx2 line ("remaining → cache blocking
ADR-0017") is STALE/SUPERSEDED — cache blocking is measured-dead; fix that line next time the spec is
edited. META-LESSON (I re-ran a benchmark this memory already had): RECALL this file BEFORE profiling
the amd64 GEMM — it is fully characterized. Both my niches (amd64 SIMD + CUDA hand-NVRTC) are now at
their practical frontiers; remaining gaps = hand-asm/CUTLASS/newer-Go territory. Productive path =
ship the queued CUDA work, not manufacture marginal probes.

Correctness rule: F32 path uses FMA (ADR-0021 tol, not bit-exact vs scalar); F64 path is
bit-exact (ascending-p, Mul-then-Add, §V10/§V11 tol 0). Any GEMM change must keep parity
via `TestGemmSmallMCrossReference` / `gemm_f32parity_test.go`.

Collision note: `backend/cpu` is also the [[main-machine-concurrent-campaign]] target, BUT
the amd64-simd-tagged files are lower-collision (ARM can't touch them). Keep edits inside
the `amd64 && goexperiment.simd` files; leave the scalar `gemm_nosimd.go` / shared
dispatcher (`gemm.go`) to coordination. Related: [[linux-amd64-worker-role]].
