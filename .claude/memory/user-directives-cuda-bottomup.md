---
name: user-directives-cuda-bottomup
description: "User directives (2026-07-14) — build/optimize the full CUDA backend; optimize bottom-up, foundational layer first (most impact)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 491e34f2-de40-467f-89a0-f918efb8a6b3
---

Two standing directives from the user (2026-07-14) for the goai Linux worker:

1. **Implement and optimize the FULL CUDA backend.** Use the RTX 3060 (compute cap 8.6, Ampere). `backend/cuda/` already exists (cuBLAS-based, behind `-tags cuda && cgo`) but the CUDA toolkit (nvcc/cublas) is NOT installed and this is bazzite (immutable Fedora) → install toolkit user-space via **pip CUDA wheels in a venv** (nvidia-cuda-nvcc / nvidia-cublas / nvidia-cuda-runtime cuXX), point cgo CFLAGS/LDFLAGS at site-packages. `register_cuda.go` auto-registers under the tag.

2. **Optimize from the bottom / first layer — most impact first.** Go bottom-up: foundational kernels (GEMM/matmul is the bottom-most, highest-impact) before higher layers. This validates the fire-3/4 GEMM focus; apply the same to CUDA (get the cuBLAS GEMM path working+fast first).

**Why:** explicit user asks that reshape the loop's priorities.

**How to apply:** treat CUDA backend as the major new multi-fire thread; within any backend, optimize the foundational GEMM/matmul layer first, then build up. Advance CUDA and CPU-SIMD threads in parallel (LOOP.md "when one thread is blocked, advance another"). See [[linux-amd64-worker-role]].

**CUDA SETUP DONE (fire 5, PR #4 `linux-amd64/cuda-backend-validate`):** toolkit = CUDA 12.9 pip wheels in `.venv-cuda` (nvidia-cuda-nvcc/cublas/cuda-runtime/cccl-cu12); NO nvcc needed (pure cgo+cuBLAS, no .cu). Build via `source scripts/cuda-pip-env.sh && go test -tags cuda ./backend/cuda/`. All 5 backend tests PASS on RTX 3060 (incl. GPU training convergence). cuBLAS GEMM: 512³ 376 / 1024³ 808 GFLOP/s vs cpu 42 (8.9×/19×). Fixed a mislabeled bench (line 126 used backend.CUDA not CPU). **KEY: backend is TRANSFER-BOUND** (per-call cudaMalloc+H2D/D2H; 808 ≪ 3060's ~12.7 TFLOP/s peak).

**CUDA progress:** Fire 6, **PR #5** `linux-amd64/cuda-bufpool`: device-buffer POOL in cuda_bridge.c (gA/gB/gC grow-only, persist across calls; mutex serializes matmul → handle+buffers concurrency-safe). GEMM 512³ 374→464 (1.24×), 1024³ 813→1045 (1.29×), bit-exact. That's the ALLOC half of the transfer layer.

**CUDA progress cont'd:** Fire 10 **PR #9 (merged)**: **resident-weight matmul** (`cuda.NewResidentB(w).MatMul(a)`) — §V14 Phase-1, mirrors metal §T156 seed. Upload weight ONCE, reuse across activations, skip per-call B H2D. Bridge: `cu_upload_f32`/`cu_free_f32`/`cu_matmul_f32_bres`. Result identical to per-call Sgemm. **26× on decode shape (M=8,K=N=4096: 7.8ms→0.3ms)**, 1.26× on square GEMM. This is THE inference lever (weight fixed, activation varies).

**Device-residency phasing (ADR-0019, metal did T366-T412):** ✅ Phase-1a = resident weights (PR #9). NEXT = keep ACTIVATIONS resident across ops too (a CUDA recorder/stream chaining ops in one submit) — the BIG architectural change; would benefit from coordination on the tensor device-storage model (metal's lives in a backend-specific recorder, NOT generic tensor.Storage — so CUDA needs its own). Also: only OpMatMul f32 on GPU; extend ops later. cuBLAS kernel itself already optimal.

**Merged so far:** PR #1 (floor doc), #2 (elementwise SIMD 2.2×), #3 (F64 GEMM SIMD 1.5×), #4 (CUDA validate). Open: #5 (cuda bufpool). Queued CPU: f32-native GEMM (§V10 ADR needed).

**REAL-MODEL INFERENCE APPROVED (user 2026-07-15, PR #30):** test GGUFs fetched to `models/` (gitignored; `models/README.md` documents them). All Q8_0, verified via `gguf.ReadFile`, fit 12GB VRAM in f32:
- `tinyllama-1.1b-chat-q8_0.gguf` — arch=llama, 22L, GQA 32/4, embd 2048, NO attention bias → **FIRST GPU TARGET** (all ops on the CUDA path: RMSNorm/RoPE/GroupedQueryAttention/SwiGLU/Embed).
- `qwen2.5-0.5b` + `qwen2.5-1.5b` — arch=qwen2, GQA, +QKV-projection bias (needs a resident bias-add first).

**PLAN (next fires):** (1) TinyLlama single-layer parity: dequant GGUF layer → resident CUDA layer vs CPU `nlp.Llama` layer. (2) Full forward: Embed→N layers→final RMSNorm→output matmul→logits, token-for-token vs `nlp.Llama` greedy decode. (3) Qwen QKV bias-add. (4) on-device KV-cache decode (RoPE PosOffset). Weights: `nlp.LlamaFromGGUF`/`gguf.ReadFile` load+dequant to f32 tensors → upload to `ResidentB`/`ResidentVec`.
