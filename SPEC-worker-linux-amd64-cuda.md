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

CPU-3 (GEMM F32, f32-NATIVE, Iw4/ADR-0021): `Float32x8`+`MulAdd`, widen→f64 carrier ONCE per tile (`storeF32x8`, ⊥ per-iter convert). nr=16 register blocking (2 `Float32x8`/row = 8 ILP chains; nr=8 was FMA-latency-bound, ≈half-saturated). A/B 1024³: scalar 42.6 → nr8 128 → nr16 153 → DIRECT-store 196 GFLOP/s = 4.7× over scalar. DIRECT store (`gemmF32BandDirect` writes f32 to C, no f64 carrier): eliminating the f64-acc round-trip (doubled store traffic + full narrowing pass) = +28% (153→196); build-tagged `gemmF32` wrapper (default = f64-acc bit-exact, experiment = direct). vendor gap 3.8×→3.0×. blast radius MEASURED = only 2 backend/cpu parity tests (nn/nlp/autograd ⊥ assert F32-exact matmul). same per-element p-order → nr16≡nr8 f32 result, tolerance test unchanged.
  REJECTED (§C3): f64-accumulating F32 SIMD twin (per-iter `LoadFloat32x4Slice`+`ConvertToFloat64`) regressed ≈25× (43→1.7 GFLOP/s) — 128-bit load+widen in hot loop pathological.

CPU-FLOOR (measured pre-SIMD, §V22): scalar pure-Go GEMM 1024³ = F64 42, F32 43 GFLOP/s (F32≈F64 → scalar captured none of f32 density). arm64 M-series ceiling was ≈50 (§T597).

## §GPU — NVIDIA CUDA/cuBLAS capability (`backend/cuda`)

GPU-1 (validation): backend written blind on arm64, NEVER built/run before this worker. all 5 tests PASS on RTX 3060: TestCUDACrossReference (cuBLAS Sgemm==ref), TestCUDARectangular, TestCUDAFallback (non-matmul→pure-Go §I4), TestCUDAGPUTraining (GPU training converged, gpu loss 0.0371==cpu), TestCUDAServesMatmul. row-major C=A·B via col-major Cᵀ=Bᵀ·Aᵀ Sgemm idiom (§R43).
  bug fixed: `BenchmarkMatMulF32_1024_cpu` called `backend.CUDA` (copy-paste) → §C3 gate was cuBLAS-vs-cuBLAS. now `backend.CPU`.

GPU-2 (cuBLAS GEMM, §V22): 512³ 376 vs cpu 42 = 8.9×; 1024³ 808 vs cpu 42 = 19×. TRANSFER-bound (808 ≪ 3060 ≈12.7 TFLOP/s f32 peak): per-call cudaMalloc + H2D/D2H.

GPU-3 (device-buffer pool, `cuda_bridge.c`): gA/gB/gC persist across calls, grow-only to largest M*K/K*N/M*N; fixed-shape loop allocs once. single mutex serializes matmul → shared handle+buffers concurrency-safe (old global handle was unguarded race). A/B: 512³ 374→464 = 1.24×; 1024³ 813→1045 = 1.29×. bit-exact (same Sgemm).

GPU-4 (resident weights, §V14 Phase-1, mirrors metal §T156): `cuda.NewResidentB(w)` uploads weight ONCE, `.MatMul(a)` reuses across activations, skips per-call weight H2D. bridge `cu_upload_f32`/`cu_free_f32`/`cu_matmul_f32_bres`. identical to per-call Sgemm. Also `.Embed(ids)` = input embedding row-gather (table[vocab,d] resident, bit-exact vs ref). A/B: 1024/2048 square 1.26×; DECODE M=8 K=N=4096 7.81ms→0.30ms = 26× (transfer-dominated inference shape, skips 64MB weight re-upload).

GPU-5 (activation residency, §V14 Phase-2): `DeviceF32` = on-GPU rank-2 f32 activation; device×device matmul `.MatMul(b)` & `.MatMulBT(b)` (QKᵀ) both activations resident; `.Clone()` (D2D copy); `MultiHeadAttention(q,k,v,heads,causal)` = strided-batched Sgemm; `GroupedQueryAttention(q,k,v,qHeads,kvHeads,causal)` = POINTER-ARRAY batched Sgemm (cublasSgemmBatched; query h→kv head h/group; Llama-3/Mistral GQA); both + per-head mask + softmax on [qHeads·seq,seq], resident; `UploadF32(x)` → device; `ResidentB.MatMulDevice(dact)` → device out (⊥ H2D/D2H); `.ToHost()` downloads. a matmul CHAIN keeps intermediates on-GPU: MLP x·W1·W2 = 1 upload + 1 download (⊥ per-matmul bounce). bridge `cu_alloc_f32`/`cu_download_f32`/`cu_matmul_f32_ddd`. identical to per-call cuda chain (same Sgemm seq), tolerance vs ref. A/B chain M=8 D=4096: per-call 15.9ms → device 0.54ms = 29×.

GPU-7 (on-device kernels via NVRTC, §V14 Phase-2 breadth): custom CUDA-C kernels compiled at RUNTIME by nvrtc (⊥ nvcc — only ptxas+nvrtc in wheels) → PTX → driver-API `cuModuleLoadDataEx`/`cuLaunchKernel`. runtime↔driver interop: retain the runtime's PRIMARY context (`cuDevicePrimaryCtxRetain`) + `cuCtxSetCurrent` per launch (Go goroutines migrate threads → driver ctx not auto-bound like runtime). launch on gStream = ordered w/ matmuls. link `-lnvrtc -lcuda`; libcuda.so=/usr/lib64 (driver), libnvrtc=wheel. `compile_kernel(src,entry)` helper = reusable nvrtc compile+load. on-device ops: `(*DeviceF32).GELU()` (exact erff, ref 5.96e-8), `.SiLU()` (x·sigmoid, stable), `.Add(other)` (residual d+=other), `.RMSNorm(gamma,eps)` (block-per-row DOUBLE-accum reduction; `ResidentVec` gamma), `.Softmax()` (stable, two-reduction, double-accum), `.RoPE(attrs)` (HF rotate_half, freqs from backend.RoPEFreqs on host; XPos not yet), `.Mul(other)` (d*=other), `.SwiGLU(up)` (FUSED SiLU(gate)*up, 1 launch+3n traffic vs SiLU+Mul's 2 launches+5n), `.CausalScaleMask(scale,offset)` (attention pre-softmax: scale + j>i+offset→-inf; nvrtc needs `__int_as_float(0xff800000)` for -inf, ⊥ `-INFINITY`). COMPOSITION VERIFIED: full SwiGLU FFN block (RMSNorm→Wgate/Wup→SiLU→Mul→Wdown→+residual) assembled from these primitives tracks ref e2e (max rel err 1.24e-3), fully resident 566µs (M=8 D=2048 ffn=5504). FFN block x·W1→GELU→·W2 on-device 4.5× vs host-GELU round-trip (504µs vs 2260µs). matmul→SiLU→residual-add block fully resident (122µs, M=8 D=2048). UNLOCKS all elementwise/activation on device.

GPU-6 (async stream, §V14 Phase-2): whole backend stream-based — 1 cublas stream + cudaMallocAsync/FreeAsync for caller buffers; ALL work (H2D/Sgemm/alloc/free/D2H) queued, host blocks only at data-returning points (cu_matmul*, cu_download) via 1 cudaStreamSynchronize. a DEEP matmul chain pipelines with ~2 barriers (upload+download) not 1/link (cudaMalloc AND cudaFree also implicitly sync → the real cost). A/B deep chain: L8 D1024 303→206µs=1.47×; L16 D512 349→202µs=1.73× (deeper→more syncs cut). parity green (stream ordering).

## §GAP — vendor-BLAS gap on this Zen3 (torch-cpu 2.13, numpy 2.4.4/OpenBLAS)

GAP-1 (1024³ GFLOP/s): F64 goai 84 | torch 177 | numpy 227 → ≈2.7×. F32 goai(f32-native nr16) 153 | torch 580 | numpy 485 → ≈3.8× (was 13× vs scalar 43).
GAP-2: F64 gap partly = bit-exact `Mul`+`Add` (≈2× of FMA peak). CACHE BLOCKING re-measured on this Zen3 (ADR-0017 resume condition): packed-B REGRESSED (512 −16%, 1024 −6%) → DISCARDED, x86 resume condition CLOSED with data (kernel ⊥ cache-capacity/B-read-bound; B fits L3). remaining ≈3× vendor gap = microkernel saturation + §V10 f64-accum policy, ⊥ blocking.
GAP-3 (thread finding): torch FASTER at 8 threads than 16 on 8c/16t (SMT contention, compute-bound GEMM). BUT goai GEMM is SLOWER at 8 than 16 (69 vs 81 GFLOP/s) — its less-saturated kernel benefits from SMT hiding stalls → ⊥ cap parallelWork at physical cores (measured negative).

## §PERF — inference throughput (user directive 2026-07-15: benchmark vs industry-standard, then hyper-optimize)

PERF-1 (goai TinyLlama-1.1B prefill, f32, NO KV-cache, RTX 3060, §V22): GPU 1204 tok/s @seq32 / 2002 @seq128; CPU (nlp.Llama f64) 5.9 tok/s @seq32 → GPU ≈204–340× CPU. `BenchmarkTinyLlamaPrefill{GPU,CPU}`.
PERF-2 INDUSTRY BASELINE (llama.cpp b10012 Vulkan, same RTX 3060 + same TinyLlama GGUF, Q8_0, -ngl 99, `scripts/bench-llamacpp.sh`; llama.cpp CUDA impossible here — pip wheels ship only ptxas no nvcc, no Linux CUDA prebuilt published; Vulkan is a legit GPU backend + goai has one too):
  | test | llama.cpp Vulkan Q8_0 | goai CUDA f32 | goai/llcpp |
  |---|---|---|---|
  | prefill pp32 | 3298 tok/s | 1330 | 0.40× (2.5× behind) |
  | prefill pp128 | 8389 tok/s | 2187 | 0.26× (3.8× behind) |
  | decode tg128 | 243 tok/s | 25.7 (KV+fusions, QUIET box) | 0.106× (9.5× behind) |
PERF-GAP (honest): (a) llama.cpp Q8_0 = 4× less weight bandwidth than goai f32 (favours llcpp, esp. memory-bound decode). (b) llama.cpp fuses kernels + batched/flash attention → prefill throughput scales HARD with batch (3298→8389 pp32→pp128); goai rises far less (1204→2002) — many separate cuBLAS + per-op kernel launches per layer. (c) goai has NO KV-cache → no comparable decode number yet (would be O(N²)). goai's from-scratch CUDA prefill is within 2.7–4.2× of a mature impl — a solid start; the levers below close it.
PERF-Q8 (bandwidth arc, PR#47): on-device Q8_0 quantized matmul — ResidentBQ8 (weight stored TRANSPOSED [N,K] int8 + per-32-block f32 scales; symmetric quant) + warp-per-output GEMV kernel (32 lanes split K, coalesced int8 loads, shuffle-reduce). Correct: L1 rel err 0.4% vs f32 (Q8 budget). Decode-GEMV [1,2048]×[2048,2048]: Q8 43µs vs cuBLAS f32 81µs = **1.88× faster** (int8 = 4× less weight bandwidth, the decode memory-bound win). Q8 FULL DECODE wired (PR#48): resLayerQ8 (7 projections + Out Q8) + QMatMulAccInto (beta=1 residual). CORRECT: Q8 decode == f32 TOKEN-FOR-TOKEN 5/5, logit L1<0.05%. BUT end-to-end SLOWER: Q8 20.6 vs f32 25.8 tok/s — decode is LAUNCH-BOUND (host-dispatch), so Q8's GPU-bandwidth win (isolated GEMV 1.88×) is MASKED, and swapping cuBLAS→driver-API kernel ADDS per-launch host overhead (net −20%). ⇒ CUDA GRAPHS is now the PREREQUISITE lever (capture per-token op seq, replay, kill per-launch host cost); THEN Q8 bandwidth pays off. (also suspect: small-N GQA matmuls N=256 underutilize the warp kernel — investigate.)
PERF-NEXT (hyper-opt, biggest first): (1) KV-CACHE decode DONE §Tw33/§Tw34 — GroupedQueryAttentionKV + KVCache(append) + decode loop == full re-forward token-for-token (argmax exact, logits 1e-5); decode 12.6 tok/s, FLAT across context (12.62@p32/12.51@p128 → cache truly O(1)/step). GAP to llama.cpp 243 = 19×, diagnosed LAUNCH/SYNC-BOUND not bandwidth (12.6 tok/s ⇒ 55 GB/s effective, far below 360 GB/s peak; single-token forward = ≈330 op launches w/ same fixed cost as prefill but 1/128th compute). GQA persistent scratch DONE §Tw35 (removed 264 cudaMalloc/free per token) → only ≈1% (12.6→12.75): so decode is NOT alloc-bound. RE-DIAGNOSIS: decode is LAUNCH-COUNT-bound — ≈330 sequential host→driver ops/token (cgo+mutex+cuCtxSetCurrent each), chain is async w/ 1 sync/token; matmul_ddd + elementwise don't sync. REAL LEVER = FEWER launches: kernel FUSION (fuse RMSNorm+matmul, RoPE into proj, SwiGLU chain) + CUDA GRAPHS (capture per-token op sequence, replay — kills per-launch overhead); THEN quant matmul for bandwidth. (2) quantized matmul on device (Q8/Q4 resident → 4× bandwidth, matches llcpp). (3) kernel fusion (RMSNorm+matmul, flash-style attention) + fewer launches — closes the prefill-scaling gap. (4) pooled intermediates. (5) larger prefill batch.

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
Tw19|x|CUDA device×device matmul + transpose-B (QKᵀ) — attention matmul ops (PR#20)|GPU-5,Nx1
Tw20|x|CUDA causal-scale-mask + single-head ATTENTION block composed & verified e2e vs ref (PR#21)|GPU-7,Nx1
Tw21|x|CUDA Clone (D2D) + FULL single-head decoder LAYER composed & verified e2e vs ref (max rel 1.4e-5) (PR#22)|GPU-5,Nx1
Tw22|x|CUDA MULTI-HEAD attention (batched strided Sgemm) verified vs per-head ref, causal+full (PR#23)|GPU-5,Nx1
Tw23|x|CUDA full MULTI-HEAD decoder LAYER composed & verified e2e vs ref (max rel 7.7e-4) (PR#24)|GPU-5,Nx1
Tw24|x|CUDA GQA (grouped-query attn, pointer-array batched Sgemm) verified vs per-head ref (PR#25)|GPU-5,Nx1
Tw29|x|REAL-MODEL: TinyLlama-1.1B block-0 on CUDA path w/ actual GGUF weights == CPU nlp block (OpMHA) max rel 1.4e-6 (PR#31)|Nx1,GPU-5
Tw30|x|REAL-MODEL FULL FORWARD: TinyLlama-1.1B (embed→22 layers→norm→logits) on GPU == CPU nlp.Llama TOKEN-FOR-TOKEN (6/6 greedy argmax, logit 1.7e-5) (PR#32)|Nx1
Tw31|x|inference THROUGHPUT bench: TinyLlama prefill GPU 1204/2002 tok/s (seq32/128) vs CPU 5.9 — baseline for the llama.cpp comparison + hyper-opt (PR#33)|§PERF
Tw32|x|INDUSTRY BASELINE: llama.cpp b10012 Vulkan on same GPU+model — prefill 3298/8389 (pp32/128), decode 243 tok/s; goai CUDA within 2.7–4.2× on prefill, decode gap = KV-cache. Reproducible `scripts/bench-llamacpp.sh` (PR#34)|§PERF
Tw33|x|KV-CACHE attn FOUNDATION: generalized GQA/mask kernels seq→seqQ×seqKV; GroupedQueryAttentionKV (seqQ new query rows over full seqKV cache, causal offset seqKV−seqQ) == tail of full causal GQA (seqQ=1 maxAbs 2.6e-8, seqQ=3/12 exact); no regression in MHA/GQA/block/full-forward parity (PR#35)|§PERF
Tw34|x|KV-CACHE DECODE: cu_copy_rows + KVCache type (resident [maxSeq,wkv] K/V, append post-RoPE) + forwardKV/decode loop; autoregressive greedy decode == full re-forward TOKEN-FOR-TOKEN (argmax exact 5 steps, logits ≤1.1e-5); decode 12.6 tok/s flat across context. Gap to llama.cpp 243=19×, launch/sync-bound (PR#36)|§PERF
Tw35|x|GQA persistent scratch (kill 264 cudaMalloc/free per token) — parity holds, but only ≈1% (12.6→12.75): decode is LAUNCH-COUNT-bound not alloc-bound → real lever = fusion + CUDA graphs (PR#37)|§PERF
Tw36|x|RESIDUAL FUSION: cu_matmul_f32_ddd_acc (cuBLAS beta=1) + MatMulAccInto — fuse 2 residual Adds/layer into Wo/Wd projection matmuls (−44 launches/token). Exact (full-fwd 6/6 1.7e-5, KV==re-forward). Decode 12.75→13.95 @p32 (+9%), prefill 1204→1282/2002→2090 (+5%) — confirms launch-bound (PR#38)|§PERF
Tw37|x|OUT-OF-PLACE RMSNORM: cu_rmsnorm_f32(in,out) + RMSNormTo — drop 2 Clone launches+allocs/layer on residual path (−44/token). Exact (all parity green). Decode 13.95→16.35 @p32 (+17%) / →17.07 @p128; prefill 2090→2353 @128 (+13%). Cumulative decode 12.6→16.35 (+30%) (PR#39)|§PERF
Tw38|x|FUSED ATTN SOFTMAX: cu_attn_softmax folds scale+causal-mask+softmax into 1 kernel (was 2) in gqaCore — −22 launches/token. Exact (all parity). PAIRED A/B (same contention window, since 4 opt-subagents load CPU): decode +4.4% p32 / +4.5% p128. NB launch-bound GPU bench needs paired A/B when CPU contended (PR#41)|§PERF
Tw39|x|RECALIBRATE + reject: with the 4 opt-subagents DONE the box is quiet → TRUE decode 25.7 tok/s (earlier 12.6→16.35 were contention-DEPRESSED; the paired-A/B relative gains still hold, only absolutes were low). Gap to llama.cpp 243 = 9.5× not 15×. REJECTED §C3: fused gate|up projection (combined Wgu matmul + fused-swiglu kernel) — −4% decode (25.7→24.7): the fused-swiglu reads gate[j]&up[H+j] 2H apart (poor coalescing) > saved launch; keep separate contiguous gate/up (PR#46)|§PERF
Tw25|x|CUDA embedding row-gather (bit-exact) — full-model forward-pass op set COMPLETE on-device (PR#26)|GPU-4,Nx1
Tw26|x|CUDA fused SwiGLU (SiLU⊙up, 1 pass) — device traffic 5n→3n, FFN launch fusion (PR#27)|GPU-7
Tw27|x|CPU f32-native GEMM direct-store (drop f64 carrier) +28% → 196 GFLOP/s, 4.7× scalar (PR#28); mr=8 tiling measured as -7% loss, rejected|CPU-3
Tw28|x|CPU GEMM B-packing re-measured on Zen3 (ADR-0017 resume) — REGRESSED -6/-16%, discarded; x86 resume condition closed with data (PR#29)|§GAP

## §NEXT — open levers

Nx1: ◐ CUDA activation residency — Phase-2 device-matmul CHAIN DONE (GPU-5/Tw11, `DeviceF32`, 29× MLP). remaining = full recorder (arbitrary op chains in one submit, async, non-matmul ops on device) + integrate into an nn/llamagpu decode path (currently a standalone primitive, orphan until wired). BIG. USER PRIORITY.
Nx2: ◐ f32-native nr=16 DONE (Tw10, 153 GFLOP/s). remaining → cache blocking (ADR-0017 re-open this large-cache x86) + FMA-saturation microkernel to close §GAP F32 3.8×/F64 2.7×.
Nx3: ✅ FULL Llama-block kernel set on-device (GPU-7): matmul + GELU/SiLU + residual-add + RMSNorm + Softmax + RoPE. attention (RoPE→QKᵀ→softmax→·V) AND FFN (RMSNorm→matmul→SiLU→matmul→residual) run fully resident. ◐ Nx1: SwiGLU FFN block composed+verified e2e (Tw18). ✅ FULL single-head decoder LAYER composed+verified e2e (Tw21, max rel 1.4e-5): pre-norm attn (Q/K/V/O proj + RoPE + causal attn + residual) + pre-norm SwiGLU FFN + residual, all resident. ✅ FULL-MODEL FORWARD-PASS OP SET COMPLETE on-device: embed-gather → decoder layers (MHA/GQA + SwiGLU FFN + norms + residuals) → final RMSNorm → output matmul→logits. All verified vs ref. ONLY REMAINING for real inference: real GGUF weights (needs USER download permission) + tokenizer + the glue (multi-layer loop, load weights into ResidentB). NEXT candidates: multi-layer stack test (synthetic); perf-tune resident reduction kernels (warp-shuffle); XPos RoPE; OR real weights if user approves. then a full decoder layer; real GGUF weights; batched/GQA Sgemm; expose via backend Kernel interface for nn/llamagpu. + XPos RoPE.
