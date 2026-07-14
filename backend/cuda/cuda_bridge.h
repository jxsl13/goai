// CUDA/cuBLAS bridge for the optional cuda backend (§T42). Compiled only under
// `-tags cuda` with cgo on linux/windows. Synchronous API: each call copies
// H2D, runs cublasSgemm, copies D2H and syncs before returning (async batching
// and device-resident tensors are a later optimization; §V14 keeps the interface
// stable so that can land without an API break).
#ifndef GOAI_CUDA_BRIDGE_H
#define GOAI_CUDA_BRIDGE_H

// cu_available returns 1 if at least one CUDA-capable GPU is present.
int cu_available(void);

// cu_matmul_f32 computes C[M,N] = A[M,K]·B[K,N], all row-major float32.
// Returns 0 on success, nonzero on failure (see cuda_bridge.c for codes).
int cu_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);

// Resident-B matmul (§V14 Phase-1, mirrors the metal §T156 resident-weight seed):
// upload a weight B[K,N] to the GPU ONCE, then reuse it across many matmuls,
// skipping its per-call H2D. This is the transfer lever for inference, where the
// weight is fixed and only the activation A varies.
//
// cu_upload_f32 copies n row-major floats to a fresh device buffer and returns an
// opaque device handle (NULL on failure). cu_free_f32 releases it (NULL-safe).
void* cu_upload_f32(const float* src, int n);
void cu_free_f32(void* dptr);

// cu_matmul_f32_bres computes C[M,N] = A[M,K]·dB[K,N] with A and C host-side and
// dB a resident handle from cu_upload_f32 (its element count must be K*N). A and
// C use the same pooled device buffers as cu_matmul_f32. Returns 0 on success.
int cu_matmul_f32_bres(const float* A, const void* dB, float* C, int M, int K, int N);

// Fully-device matmul (§V14 Phase-2, activation residency): all three operands
// are device handles, so a chain of matmuls keeps its intermediates on the GPU —
// only the first activation upload and the final download touch host memory.
//
// cu_alloc_f32 returns an uninitialized device buffer of n floats (NULL on fail);
// cu_download_f32 copies n floats device→host. cu_matmul_f32_ddd computes
// dC[M,N] = dA[M,K]·dB[K,N] with every operand resident (no H2D/D2H).
void* cu_alloc_f32(int n);
int cu_download_f32(const void* dsrc, float* dst, int n);
int cu_matmul_f32_ddd(const void* dA, const void* dB, void* dC, int M, int K, int N);

#endif
