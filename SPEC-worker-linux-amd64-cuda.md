# SPEC-worker — linux/amd64 + NVIDIA (RTX 3060)

Worker sub-spec of root `SPEC.md` (§ main references this). Scopes the
hardware-specific work + measured results of the Linux/amd64 secondary machine,
by capability. Encoding: caveman (`FORMAT.md`) — LLM-audience, C13-exempt.
Worker mode: PR-only, one dedicated branch+PR per task, auto-merge on green CI.
Consolidates the former `docs/{benchmarking,simd,simd-gemm,cuda}-amd64.md`.

## §H — hardware & toolchain

H1: CPU = AMD Ryzen 7 5700G (Zen 3, 8c/16t). f32-relevant ISA: AVX2 + FMA. ⊥ AVX-512.
H2: GPU = NVIDIA GeForce RTX 3060 (12 GB, Ampere, compute cap 8.6). driver 610.43.02 (ships `libcuda.so`). = ONLY NVIDIA GPU in project.
H3: toolchain = Go 1.26.5 `linux/amd64`, gcc 16.1.1. immutable distro (bazzite) → ⊥ system CUDA.
H4: CUDA 12.9 = USER-SPACE pip wheels in `.venv-cuda` (`nvidia-cuda-nvcc/cublas/cuda-runtime/cccl-cu12`). backend = pure cgo + cuBLAS, ⊥ `.cu` kernels → needs headers+libs only, `nvcc` ⊥ needed (wheel ships `ptxas`+`crt/host_config.h`). `scripts/cuda-pip-env.sh` sets CGO_CFLAGS/LDFLAGS+rpath + unversioned `.so` symlinks. torch-cpu+numpy in `.venv` = vendor-BLAS refs.
H5: role = secondary worker. main dev = macbook M2 on `main`. niche = (a) amd64 f32-SIMD (host-blocked on arm64 per §T11b/§T74/§B13) + (b) NVIDIA CUDA. ⊥ touch `main`; PR + dedicated branch only. push via `gh` (git credential helper).

## §Iw — worker invariants

Iw1: ∀ SIMD kernel behind `//go:build amd64 && goexperiment.simd`; scalar twin gets `!(amd64 && goexperiment.simd)` → exactly 1 def per build. default `CGO_ENABLED=0` build UNCHANGED (§V7). CI simd gate = `GOEXPERIMENT=simd go build -tags=simd ./...` (BUILDS, ⊥ tests the experiment).
Iw2: archsimd = Go 1.26 `simd/archsimd`. types `Float32x8` (8 lanes) / `Float64x4` (4 lanes); `.MulAdd(y,z)`=x*y+z FMA (1 rounding); `.Mul`+`.Add`=2 roundings; `archsimd.X86.AVX()/.FMA()` = runtime feature checks. ! runtime-gate ∀ intrinsics (§I4): built w/ experiment on pre-AVX CPU → scalar fallback, ⊥ illegal instruction.
Iw3: CUDA behind `-tags cuda && cgo && (linux||windows)`. ⊥ CI GPU runner → cuda tests run LOCALLY on this worker only. ! validate cuda on the GPU before push (green CI ≠ cuda coverage).
Iw4 (ADR-0021, AMENDS §V10): cpu F32 matmul = f32-NATIVE (f32 accumulation) under the experiment build ONLY → TOLERANCE vs f64 ref (rel 2e-3, abs 1e-4; observed max ≈1e-4 for K≤128), ⊥ tol-0. default build F32 = scalar f64-accum, bit-exact. F64 bit-exact both builds. conv (`gemmF64Band`) untouched. parity build-tagged: `assertMatMul` + `gemmF32Tolerant` (exact default / tolerance experiment).
Iw5: BIT-EXACT SIMD rule — vectorize the FREE dim (j), ⊥ the reduction (k) → ascending-p order preserved; `Mul`+`Add` ⊥ `MulAdd` (FMA fuses to 1 rounding ≠ scalar's 2). load/store accumulator preserves `+=` contract (conv shares `gemmF64Band`).
Iw6: A/B discipline (§V22) = same host, file-copy toggle baseline (⊥ `git stash`, shallow history), warmup excluded, count≥2 medians, GFLOP/s. every win = 1 §Tw + docs row.

## §CPU — CPU-SIMD capability (`internal/simd`, `backend/cpu`)

CPU-1 (elementwise, `internal/simd/simd_avx.go`): archsimd Add/Sub/Mul/Div × F32/F64. BIT-EXACT vs scalar (per-lane IEEE, tol 0), 14-size parity. A/B:
  AddF32 4K(L1) 26.5→60.0 GB/s = 2.26×; 256K(L2) 25.9→57.6 = 2.17×.
  AddF64 4K 52.4→60.0 = 1.15×; 256K 50.8→58.9 = 1.16×.
  finding: scalar f32 ran at HALF f64 bandwidth (wasted 2× density); 8-wide recovers to same ≈60 GB/s ceiling → now bandwidth-bound. f64 already near bw (cf §B27).

CPU-2 (GEMM F64, `backend/cpu/gemm_simd.go` `gemmF64Band`): archsimd, BIT-EXACT (Iw5). nr=8 register blocking (2 `Float64x4`/row = 8 ILP chains; nr=4 was FMA-latency-bound). A/B 1024³: scalar 40.8 → nr4 62.4 → nr8 82.3 GFLOP/s = 1.95× over scalar. conv F64 (im2col→GEMM, shared kernel) inherits ≈2×.

CPU-3 (GEMM F32, f32-NATIVE, Iw4/ADR-0021): `Float32x8`+`MulAdd`, widen→f64 carrier ONCE per tile (`storeF32x8`, ⊥ per-iter convert). nr=16 register blocking (2 `Float32x8`/row = 8 ILP chains; nr=8 was FMA-latency-bound, ≈half-saturated). A/B 1024³: scalar 42.6 → nr8 128 → nr16 153 GFLOP/s = 3.6× over scalar (nr16 +22% over nr8). blast radius MEASURED = only 2 backend/cpu parity tests (nn/nlp/autograd ⊥ assert F32-exact matmul). same per-element p-order → nr16≡nr8 f32 result, tolerance test unchanged.
  REJECTED (§C3): f64-accumulating F32 SIMD twin (per-iter `LoadFloat32x4Slice`+`ConvertToFloat64`) regressed ≈25× (43→1.7 GFLOP/s) — 128-bit load+widen in hot loop pathological.

CPU-FLOOR (measured pre-SIMD, §V22): scalar pure-Go GEMM 1024³ = F64 42, F32 43 GFLOP/s (F32≈F64 → scalar captured none of f32 density). arm64 M-series ceiling was ≈50 (§T597).

## §GPU — NVIDIA CUDA/cuBLAS capability (`backend/cuda`)

GPU-1 (validation): backend written blind on arm64, NEVER built/run before this worker. all 5 tests PASS on RTX 3060: TestCUDACrossReference (cuBLAS Sgemm==ref), TestCUDARectangular, TestCUDAFallback (non-matmul→pure-Go §I4), TestCUDAGPUTraining (GPU training converged, gpu loss 0.0371==cpu), TestCUDAServesMatmul. row-major C=A·B via col-major Cᵀ=Bᵀ·Aᵀ Sgemm idiom (§R43).
  bug fixed: `BenchmarkMatMulF32_1024_cpu` called `backend.CUDA` (copy-paste) → §C3 gate was cuBLAS-vs-cuBLAS. now `backend.CPU`.

GPU-2 (cuBLAS GEMM, §V22): 512³ 376 vs cpu 42 = 8.9×; 1024³ 808 vs cpu 42 = 19×. TRANSFER-bound (808 ≪ 3060 ≈12.7 TFLOP/s f32 peak): per-call cudaMalloc + H2D/D2H.

GPU-3 (device-buffer pool, `cuda_bridge.c`): gA/gB/gC persist across calls, grow-only to largest M*K/K*N/M*N; fixed-shape loop allocs once. single mutex serializes matmul → shared handle+buffers concurrency-safe (old global handle was unguarded race). A/B: 512³ 374→464 = 1.24×; 1024³ 813→1045 = 1.29×. bit-exact (same Sgemm).

GPU-4 (resident weights, §V14 Phase-1, mirrors metal §T156): `cuda.NewResidentB(w)` uploads weight ONCE, `.MatMul(a)` reuses across activations, skips per-call weight H2D. bridge `cu_upload_f32`/`cu_free_f32`/`cu_matmul_f32_bres`. identical to per-call Sgemm. A/B: 1024/2048 square 1.26×; DECODE M=8 K=N=4096 7.81ms→0.30ms = 26× (transfer-dominated inference shape, skips 64MB weight re-upload).

GPU-5 (activation residency, §V14 Phase-2): `DeviceF32` = on-GPU rank-2 f32 activation; `UploadF32(x)` → device; `ResidentB.MatMulDevice(dact)` → device out (⊥ H2D/D2H); `.ToHost()` downloads. a matmul CHAIN keeps intermediates on-GPU: MLP x·W1·W2 = 1 upload + 1 download (⊥ per-matmul bounce). bridge `cu_alloc_f32`/`cu_download_f32`/`cu_matmul_f32_ddd`. identical to per-call cuda chain (same Sgemm seq), tolerance vs ref. A/B chain M=8 D=4096: per-call 15.9ms → device 0.54ms = 29×.

GPU-7 (on-device kernels via NVRTC, §V14 Phase-2 breadth): custom CUDA-C kernels compiled at RUNTIME by nvrtc (⊥ nvcc — only ptxas+nvrtc in wheels) → PTX → driver-API `cuModuleLoadDataEx`/`cuLaunchKernel`. runtime↔driver interop: retain the runtime's PRIMARY context (`cuDevicePrimaryCtxRetain`) + `cuCtxSetCurrent` per launch (Go goroutines migrate threads → driver ctx not auto-bound like runtime). launch on gStream = ordered w/ matmuls. link `-lnvrtc -lcuda`; libcuda.so=/usr/lib64 (driver), libnvrtc=wheel. `compile_kernel(src,entry)` helper = reusable nvrtc compile+load. on-device ops: `(*DeviceF32).GELU()` (exact erff, ref 5.96e-8), `.SiLU()` (x·sigmoid, stable), `.Add(other)` (residual d+=other), `.RMSNorm(gamma,eps)` (block-per-row DOUBLE-accum reduction; `ResidentVec` gamma), `.Softmax()` (stable, two-reduction, double-accum), `.RoPE(attrs)` (HF rotate_half, freqs from backend.RoPEFreqs on host; XPos not yet), `.Mul(other)` (d*=other, SwiGLU gate·up). COMPOSITION VERIFIED: full SwiGLU FFN block (RMSNorm→Wgate/Wup→SiLU→Mul→Wdown→+residual) assembled from these primitives tracks ref e2e (max rel err 1.24e-3), fully resident 566µs (M=8 D=2048 ffn=5504). FFN block x·W1→GELU→·W2 on-device 4.5× vs host-GELU round-trip (504µs vs 2260µs). matmul→SiLU→residual-add block fully resident (122µs, M=8 D=2048). UNLOCKS all elementwise/activation on device.

GPU-6 (async stream, §V14 Phase-2): whole backend stream-based — 1 cublas stream + cudaMallocAsync/FreeAsync for caller buffers; ALL work (H2D/Sgemm/alloc/free/D2H) queued, host blocks only at data-returning points (cu_matmul*, cu_download) via 1 cudaStreamSynchronize. a DEEP matmul chain pipelines with ~2 barriers (upload+download) not 1/link (cudaMalloc AND cudaFree also implicitly sync → the real cost). A/B deep chain: L8 D1024 303→206µs=1.47×; L16 D512 349→202µs=1.73× (deeper→more syncs cut). parity green (stream ordering).

## §GAP — vendor-BLAS gap on this Zen3 (torch-cpu 2.13, numpy 2.4.4/OpenBLAS)

GAP-1 (1024³ GFLOP/s): F64 goai 84 | torch 177 | numpy 227 → ≈2.7×. F32 goai(f32-native nr16) 153 | torch 580 | numpy 485 → ≈3.8× (was 13× vs scalar 43).
GAP-2: F64 gap partly = bit-exact `Mul`+`Add` (≈2× of FMA peak) + vendor cache blocking (ADR-0017 re-openable on this large-cache x86).
GAP-3 (thread finding): torch FASTER at 8 threads than 16 on 8c/16t (SMT contention, compute-bound GEMM). BUT goai GEMM is SLOWER at 8 than 16 (69 vs 81 GFLOP/s) — its less-saturated kernel benefits from SMT hiding stalls → ⊥ cap parallelWork at physical cores (measured negative).

## §Tw — worker task log (merged PRs)

Tw1|x|floor doc: amd64 GEMM scalar floor + F32≈F64 finding (PR#1)|§V22
Tw2|x|elementwise archsimd overrides F32 2.2× bit-exact (PR#2)|Iw1,Iw2
Tw3|x|F64 GEMM archsimd 1.5× bit-exact (PR#3)|Iw5,CPU-2
Tw4|x|CUDA backend validated on RTX 3060 + pip toolkit + bench bugfix (PR#4)|GPU-1,Iw3
Tw5|x|CUDA device-buffer pool 1.24×/1.29× (PR#5)|GPU-3
Tw6|x|F64 GEMM nr=8 register blocking ≈2.0× (PR#6)|CPU-2
Tw7|x|vendor-BLAS gap measured on Zen3 (PR#7)|§GAP
Tw8|x|f32-native SIMD GEMM 3.0× + ADR-0021 §V10 amend (PR#8)|Iw4,CPU-3
Tw9|x|CUDA resident-weight matmul 26× decode (PR#9)|GPU-4
Tw10|x|f32-native GEMM nr=16 (8 ILP chains) +22% → 3.6× scalar (PR#11)|CPU-3
Tw11|x|CUDA activation residency (device matmul chain) 29× MLP (PR#12)|GPU-5,Nx1
Tw12|x|CUDA async stream (malloc/free async, sync only at download) deep chain 1.5-1.7× (PR#13)|GPU-6
Tw13|x|CUDA nvrtc on-device kernels + GELU; FFN block 4.5× resident (PR#14)|GPU-7
Tw14|x|CUDA on-device SiLU + residual-add (reusable compile helper); residual block resident (PR#15)|GPU-7
Tw15|x|CUDA on-device RMSNorm (block-reduction, double-accum); Llama pre-norm proj resident 123µs (PR#16)|GPU-7
Tw16|x|CUDA on-device stable Softmax (two-reduction); attention QKᵀ→softmax→·V now resident (PR#17)|GPU-7
Tw17|x|CUDA on-device RoPE (rotate_half, RoPEFreqs on host); FULL Llama block kernels resident (PR#18)|GPU-7
Tw18|x|CUDA elementwise Mul + SwiGLU FFN block COMPOSED & verified e2e vs ref (max rel 1.24e-3), fully resident 566µs (PR#19)|GPU-7,Nx1

## §NEXT — open levers

Nx1: ◐ CUDA activation residency — Phase-2 device-matmul CHAIN DONE (GPU-5/Tw11, `DeviceF32`, 29× MLP). remaining = full recorder (arbitrary op chains in one submit, async, non-matmul ops on device) + integrate into an nn/llamagpu decode path (currently a standalone primitive, orphan until wired). BIG. USER PRIORITY.
Nx2: ◐ f32-native nr=16 DONE (Tw10, 153 GFLOP/s). remaining → cache blocking (ADR-0017 re-open this large-cache x86) + FMA-saturation microkernel to close §GAP F32 3.8×/F64 2.7×.
Nx3: ✅ FULL Llama-block kernel set on-device (GPU-7): matmul + GELU/SiLU + residual-add + RMSNorm + Softmax + RoPE. attention (RoPE→QKᵀ→softmax→·V) AND FFN (RMSNorm→matmul→SiLU→matmul→residual) run fully resident. ◐ Nx1: SwiGLU FFN block composed+verified e2e (Tw18). NEXT: compose the ATTENTION block (RoPE→QKᵀ→causal-mask→softmax→·V, multi-head reshape) e2e; then a full decoder layer; real GGUF weights; batched/GQA Sgemm; expose via backend Kernel interface for nn/llamagpu. + XPos RoPE.
