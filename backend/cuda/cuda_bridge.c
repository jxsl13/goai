//go:build cuda && cgo && (linux || windows)

// CUDA/cuBLAS side of the cuda backend (§T42). Compiled only under `-tags cuda`
// with a CUDA toolkit. One process-wide cublasHandle, lazily created. Each
// matmul copies A/B H2D, runs cublasSgemm, copies C D2H and syncs before
// returning — honest about transfer cost; device-resident tensors are the next
// optimization (§V14 keeps the Go interface stable so that lands without a break).
//
// Device buffers are POOLED, not malloc'd per call: cudaMalloc/cudaFree cost
// ~10-100us each and the profile is alloc+transfer-bound (cuBLAS itself is a
// small fraction of a 512^3 call), so three malloc + three free per matmul
// dominated small and medium GEMMs. gA/gB/gC persist across calls and grow to
// the largest (M*K, K*N, M*N) seen; a steady training loop with fixed shapes
// allocates once. A single mutex serializes the whole matmul, so the shared
// handle AND the shared buffers are safe under concurrent callers (the old code
// left the global handle unguarded — a latent race the pool would only widen).

#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <pthread.h>

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static cublasHandle_t gHandle = NULL;

static float *gA = NULL, *gB = NULL, *gC = NULL;
static size_t gACap = 0, gBCap = 0, gCCap = 0; // capacities in bytes

static int ensure_init(void) {
    if (gHandle != NULL) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    if (cublasCreate(&gHandle) != CUBLAS_STATUS_SUCCESS) { gHandle = NULL; return -1; }
    return 0;
}

// ensure_cap grows *buf to at least need bytes (grow-only; reuses on repeat).
static int ensure_cap(float **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) cudaFree(*buf);
    *buf = NULL;
    *cap = 0;
    if (cudaMalloc((void **)buf, need) != cudaSuccess) return -1;
    *cap = need;
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
    size_t aLen = (size_t)M * K * sizeof(float);
    size_t bLen = (size_t)K * N * sizeof(float);
    size_t cLen = (size_t)M * N * sizeof(float);

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    int rc = -2;

    pthread_mutex_lock(&gLock);

    if (ensure_init() != 0) { rc = -1; goto done; }
    if (ensure_cap(&gA, &gACap, aLen) != 0) { rc = -2; goto done; }
    if (ensure_cap(&gB, &gBCap, bLen) != 0) { rc = -2; goto done; }
    if (ensure_cap(&gC, &gCCap, cLen) != 0) { rc = -2; goto done; }

    if (cudaMemcpy(gA, A, aLen, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -3; goto done; }
    if (cudaMemcpy(gB, B, bLen, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -3; goto done; }

    // Row-major C[M,N]=A·B via column-major C^T = B^T·A^T (see comment above).
    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     gB, N,
                     gA, K,
                     &beta,
                     gC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = -5; goto done; }
    if (cudaMemcpy(C, gC, cLen, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -6; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// Resident-B (§V14 Phase-1). cu_upload_f32 owns a standalone device buffer (NOT
// the gA/gB/gC pool) so it can outlive individual matmuls; the caller frees it.
void* cu_upload_f32(const float* src, int n) {
    void* d = NULL;
    size_t sz = (size_t)n * sizeof(float);
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { goto done; }
    if (cudaMalloc(&d, sz) != cudaSuccess) { d = NULL; goto done; }
    if (cudaMemcpy(d, src, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        d = NULL;
    }
done:
    pthread_mutex_unlock(&gLock);
    return d;
}

void cu_free_f32(void* dptr) {
    if (!dptr) return;
    pthread_mutex_lock(&gLock);
    cudaFree(dptr);
    pthread_mutex_unlock(&gLock);
}

// cu_matmul_f32_bres: C = A·dB with dB already resident. A/C use the pool; B is
// not uploaded (the whole point). Same column-major idiom as cu_matmul_f32.
int cu_matmul_f32_bres(const float* A, const void* dB, float* C, int M, int K, int N) {
    size_t aLen = (size_t)M * K * sizeof(float);
    size_t cLen = (size_t)M * N * sizeof(float);

    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    int rc = -2;

    pthread_mutex_lock(&gLock);

    if (ensure_init() != 0) { rc = -1; goto done; }
    if (ensure_cap(&gA, &gACap, aLen) != 0) { rc = -2; goto done; }
    if (ensure_cap(&gC, &gCCap, cLen) != 0) { rc = -2; goto done; }

    if (cudaMemcpy(gA, A, aLen, cudaMemcpyHostToDevice) != cudaSuccess) { rc = -3; goto done; }

    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     (const float*)dB, N,
                     gA, K,
                     &beta,
                     gC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    if (cudaDeviceSynchronize() != cudaSuccess) { rc = -5; goto done; }
    if (cudaMemcpy(C, gC, cLen, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = -6; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}
