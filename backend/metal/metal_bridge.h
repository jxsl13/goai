// Metal/MPS bridge for the optional metal backend (§T20). Compiled only under
// `-tags metal` on darwin with cgo. Synchronous API: each call commits and
// waits (async batching is a later optimization; §V14 keeps the interface safe).
#ifndef GOAI_METAL_BRIDGE_H
#define GOAI_METAL_BRIDGE_H

// mtl_available returns 1 if a Metal device with MPS support exists.
int mtl_available(void);

// mtl_matmul_f32 computes C[M,N] = A[M,K]·B[K,N], all row-major float32.
// Returns 0 on success, nonzero on failure.
int mtl_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);

#endif
