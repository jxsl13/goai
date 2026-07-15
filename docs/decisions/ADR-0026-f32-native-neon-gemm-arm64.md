# ADR-0026 — f32-native NEON GEMM for arm64 under GOEXPERIMENT=simd

- Status: Accepted (measurement-driven, experiment-gated)
- Date: 2026-07-15
- Task: CPU F32 GEMM perf grind (arm64), amends §V10 (accumulation precision)
  for the arm64 experiment build — the arm64 twin of ADR-0021
- Related: ADR-0021 (amd64 f32-native SIMD GEMM), ADR-0017 / §T74 / §B28
  (arm64 cache blocking previously discarded for the scalar kernel)

## Context

§V10 requires the `cpu` backend to accumulate f32 products in **f64**, keeping
f32 matmul bit-identical to the f64 reference (§V3/§V11, tol 0). On the arm64
(Apple M2 Pro, 8P+4E) host that left F32 GEMM at the scalar f64-accumulating
rate — measured 46/52/61 GFLOP/s at 256³/512³/1024³ — **~42× behind torch-cpu**
(2584 GFLOP/s @1024³, which rides Apple AMX that pure Go cannot reach).

ADR-0021 already resolved the policy question for amd64: under an explicit,
opt-in perf build (`GOEXPERIMENT=simd`), F32 matmul may accumulate f32-native
within a tolerance; the default build stays bit-exact. This ADR extends the
same amendment to `arm64 && goexperiment.simd` — same build-tag opt-in, same
tolerance test contract, default untouched.

Two arm64-specific facts shaped the implementation:

1. **The Go 1.26 arm64 backend does not auto-vectorize.** The canonical
   contiguous `c[j] += a*b[j]` float32 SAXPY compiles to scalar `FMADDS`
   (verified by objdump). A pure-Go f32-native kernel was built and measured
   FIRST per the prefer-Go rule: **~76 GFLOP/s** @1024³ (vs 61 baseline) —
   the scalar-FMA ceiling, nowhere near the NEON potential. The dense inner
   tile therefore drops to a small Plan9 NEON kernel (`gemm_neon_arm64.s`).
2. **`simd/archsimd` is amd64-only**, so the arm64 kernel cannot reuse the
   ADR-0021 intrinsics path.

## Decision

Under `arm64 && goexperiment.simd` only, `cpu` F32 matmul
(`backend/cpu/gemm_neon_arm64.{go,s}`) runs an f32-NATIVE NEON GEMM:

- **4×16 register tile** (`gemmF32Tile4x16Neon`): 16 four-lane f32 accumulators
  (V0–V15) live in registers across the whole k-loop; k unrolled ×4 with one
  128-bit A-quad load per row and **by-element FMLA** (`FMLA Vd.4S, Vn.4S,
  Vm.S[e]`) so no VDUP broadcasts compete with FMLAs for the 4 FP pipes.
  The Go assembler has no by-element VFMLA, so those 64 instructions are
  WORD-encoded; encodings were cross-checked against `clang -arch arm64`.
  Generator (fields per the A64 "Advanced SIMD vector x indexed element"
  encoding, FMLA single-precision: base `0x4F801000`, Q=1, size=10):

  ```python
  for e in range(4):
      for r in range(4):
          for v in range(4):
              Rd, Rn, Rm = 4*r+v, 16+v, (20+r) & 0xF
              H, L = e >> 1, e & 1
              print(f"WORD $0x{0x4F801000|(L<<21)|(1<<20)|(Rm<<16)|(H<<11)|(Rn<<5)|Rd:08X}"
                    f" // FMLA V{Rd}.4S, V{Rn}.4S, V{20+r}.S[{e}]")
  ```

  Single-core: 87 GFLOP/s (VDUP variant) → **104.6 GFLOP/s** by-element,
  ≈93% of the 4-pipe FMA peak (4 FMLA × 4 lanes × 2 flops ≈ 112 @ ~3.5 GHz).
- **B-panel packing**: full 16-column panels packed once, panel-major
  `[n/16][k][16]`, so the tile streams B contiguously (`ldb=16`). Unpacked,
  the tile walks B with an n-element stride; the power-of-two n of LLM shapes
  (1024/2048/8192) lands on 2–4 L1 sets and conflict-misses the whole k-deep
  line column every pass. Gated to `k·n·4 > 1 MiB && m ≥ 8` (measured: 256³
  305→262 net LOSS packed, 512³ neutral, 1024³ +25%, 512×2048×8192 +4.8×).
  Pack buffers come from a NOT-zeroed dedicated pool (`f32PackScratch`).
- **L2 column blocking** (256 KiB budget, j-splits only) — re-measured per the
  §T74/§B28 caveat now that the kernel is vectorized: 512×2048×8192 82→104
  pre-packing, squares neutral. KEPT (the earlier arm64 discard was an
  artifact of the slow scalar kernel, exactly as gemm.go's comment predicted).
- **Dynamic tile scheduling**: static equal `parallelWork` chunks leave the
  wall clock to the 4 E-cores; one pool task per worker pulling 2-tile batches
  off an atomic counter lets P-cores take more tiles. Measured +20%/+33%/+23%
  at 256³/512³/1024³.
- Tails (n%16 columns, m%4 rows, m<4 GEMV) run f32-native scalar Go; k==0 is
  guarded (asm loop is do-while). **F64 is untouched** — still the shared
  scalar band kernel, f64-accumulated, bit-exact in both builds; conv's
  `gemmF64Band` path is untouched.

Policy mechanics mirror ADR-0021 verbatim: the default build keeps the
f64-accumulating scalar `gemmF32` (now in `gemm_f32default.go`) and stays
bit-exact; the experiment build asserts the ADR-0021 tolerance (rtol 2e-3,
atol 1e-4 vs the f64 ref) via the build-tagged `gemmF32Tolerant` switch
(`gemm_f32policy_arm64simd_test.go`). Each C element accumulates its k
products in ascending p order in one fused-FMA chain, so the error carries
the usual K·u_f32 bound; observed max rel err on the parity shapes ≈1e-4
inside genuine outputs.

## Consequences

- **Win (M2 Pro, §V22 A/B, -count=6 medians):**

  | shape | baseline | f32-native NEON | × | torch-cpu ratio |
  |---|---|---|---|---|
  | 256³ | 46 | ~430 | 9.4× | — |
  | 512³ | 52 | ~705 | 13.6× | — |
  | 1024³ | 61 | ~795 | 13.0× | 42× → **3.25×** |
  | 512×2048×2048 | 69 | ~690 | 10× | — |
  | 512×2048×8192 | 60 | ~578 | 9.6× | — |
  | 32×2048×2048 | 41 | ~232 | 5.7× | — |
  | 511×513×515 | 63 | ~374 | 5.9× | — |

  The remaining ≈3× vs torch-cpu (2584) is AMX vs NEON silicon, not software:
  795 aggregate ≈ 7.6 effective P-cores × the 104.6 single-core NEON ceiling.
- **Test policy:** F32 matmul parity on arm64 is now build-tagged like amd64;
  default build asserts exact, experiment build within-tolerance. F64 parity
  stays exact everywhere. Same blast radius as ADR-0021: only the two
  `backend/cpu` matmul cross-ref tests change behavior, only under the
  experiment.
- The m<4 decode-GEMV path stays scalar f32-native (~18.5 GFLOP/s,
  bandwidth-bound); vectorizing it is a separate, smaller-stakes task.

## Alternatives rejected

- **Pure-Go f32-native (no asm)** — measured ~76 GFLOP/s @1024³; the compiler
  emits scalar FMADDS and there is no arm64 auto-vectorizer to structure for.
- **VDUP-broadcast FMLA kernel** — built and measured (87 vs 104.6 GFLOP/s
  single-core): the 4 VDUPs per p-step compete with FMLAs for the FP pipes.
  Superseded by WORD-encoded by-element FMLA; kept only in the k%4 tail.
- **Always-pack / never-pack** — packing regresses 256³ by 14% and never-pack
  loses up to 4.8× on 512×2048×8192; the 1 MiB gate keeps both wins.
- **Keeping §V10 tol-0 for arm64 F32** — leaves a measured 13× on the table
  for the most common ML dtype on the primary dev host.
