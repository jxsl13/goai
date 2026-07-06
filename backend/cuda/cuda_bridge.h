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

#endif
