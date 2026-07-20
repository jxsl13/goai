## §CPU — CPU-SIMD capability (`internal/simd`, `backend/cpu`)

CPU-1 (elementwise, `internal/simd/simd_avx.go`): archsimd Add/Sub/Mul/Div × F32/F64. BIT-EXACT vs scalar (per-lane IEEE, tol 0), 14-size parity. A/B:
  AddF32 4K(L1) 26.5→60.0 GB/s = 2.26×; 256K(L2) 25.9→57.6 = 2.17×.
  AddF64 4K 52.4→60.0 = 1.15×; 256K 50.8→58.9 = 1.16×.
  finding: scalar f32 ran at HALF f64 bandwidth (wasted 2× density); 8-wide recovers to same ≈60 GB/s ceiling → now bandwidth-bound. f64 already near bw (cf §B27).

CPU-2 (GEMM F64, `backend/cpu/gemm_simd.go` `gemmF64Band`): archsimd, BIT-EXACT (Iw5). nr=8 register blocking (2 `Float64x4`/row = 8 ILP chains; nr=4 was FMA-latency-bound). A/B 1024³: scalar 40.8 → nr4 62.4 → nr8 82.3 GFLOP/s = 1.95× over scalar. conv F64 (im2col→GEMM, shared kernel) inherits ≈2×.

CPU-3 (GEMM F32, f32-NATIVE, Iw4/ADR-0021): `Float32x8`+`MulAdd`, widen→f64 carrier ONCE per tile (`storeF32x8`, ⊥ per-iter convert). nr=16 register blocking (2 `Float32x8`/row = 8 ILP chains; nr=8 was FMA-latency-bound, ≈half-saturated). A/B 1024³: scalar 42.6 → nr8 128 → nr16 153 → DIRECT-store 196 GFLOP/s = 4.7× over scalar. DIRECT store (`gemmF32BandDirect` writes f32 to C, no f64 carrier): eliminating the f64-acc round-trip (doubled store traffic + full narrowing pass) = +28% (153→196); build-tagged `gemmF32` wrapper (default = f64-acc bit-exact, experiment = direct). vendor gap 3.8×→3.0×. blast radius MEASURED = only 2 backend/cpu parity tests (nn/nlp/autograd ⊥ assert F32-exact matmul). same per-element p-order → nr16≡nr8 f32 result, tolerance test unchanged.
  REJECTED (§C3): f64-accumulating F32 SIMD twin (per-iter `LoadFloat32x4Slice`+`ConvertToFloat64`) regressed ≈25× (43→1.7 GFLOP/s) — 128-bit load+widen in hot loop pathological.

CPU-FLOOR (measured pre-SIMD, §V22): scalar pure-Go GEMM 1024³ = F64 42, F32 43 GFLOP/s (F32≈F64 → scalar captured none of f32 density). arm64 M-series ceiling was ≈50 (§T597).
