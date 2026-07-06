//go:build cuda && cgo && (linux || windows)

// CUDA/cuBLAS side of the cuda backend (§T42). Compiled only under `-tags cuda`
// with a CUDA toolkit. One process-wide cublasHandle, lazily created. Each
// matmul allocates device buffers, copies H2D, runs cublasSgemm, copies D2H,
// and frees — honest about transfer cost; device-resident tensors are a later
// optimization (§V14 keeps the Go interface stable so that lands without a break).

#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

static cublasHandle_t gHandle = NULL;

static int ensure_init(void) {
    if (gHandle != NULL) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    if (cublasCreate(&gHandle) != CUBLAS_STATUS_SUCCESS) { gHandle = NULL; return -1; }
    return 0;
}

int cu_available(void) {
    int n = 0;
    return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}

// cublasSgemm is COLUMN-MAJOR: C_cm = alpha·op(A_cm)·op(B_cm) + beta·C_cm, with
// leading dimensions counting rows of each column-major matrix. A row-major M×N
// matrix is a column-major N×M matrix, so computing C^T = B^T·A^T in column-major
// yields row-major C = A·B. Idiom: pass B (lda=N) then A (ldb=K), dims N,M,K,
// result C (ldc=N). Confirmed vs NVIDIA cuBLAS docs (§R43).
int cu_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    if (ensure_init() != 0) return -1;

    size_t aLen = (size_t)M * K * sizeof(float);
    size_t bLen = (size_t)K * N * sizeof(float);
    size_t cLen = (size_t)M * N * sizeof(float);

    // All declarations up front so the error gotos never jump over an
    // initializer (C89-safe, avoids -Werror=jump-misses-init in CI).
    float *dA = NULL, *dB = NULL, *dC = NULL;
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    int rc = -2;

    if (cudaMalloc((void**)&dA, aLen) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&dB, bLen) != cudaSuccess) goto done;
    if (cudaMalloc((void**)&dC, cLen) != cudaSuccess) goto done;

    if (cudaMemcpy(dA, A, aLen, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -3; goto done; }
    if (cudaMemcpy(dB, B, bLen, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -3; goto done; }

    // Row-major C[M,N]=A·B via column-major C^T = B^T·A^T (see comment above).
    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     dB, N,
                     dA, K,
                     &beta,
                     dC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = -5; goto done; }
    if (cudaMemcpy(C, dC, cLen, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -6; goto done; }
    rc = 0;

done:
    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    return rc;
}
