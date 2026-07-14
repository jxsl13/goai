//go:build cuda && cgo && (linux || windows)

// CUDA/cuBLAS side of the cuda backend (§T42). Compiled only under `-tags cuda`.
// One process-wide cublasHandle + one CUDA stream, lazily created. All GPU work
// (copies, Sgemm, alloc, free) is queued on that stream; the host only blocks at
// the points that return data (cu_matmul_f32*, cu_download_f32) via one
// cudaStreamSynchronize. A device-resident matmul CHAIN therefore pipelines on
// the GPU with ~2 host barriers total (initial upload + final download) instead
// of a CPU round-trip per link (§V14 Phase-2 async).
//
// Caller-owned device buffers (cu_upload_f32 / cu_alloc_f32 / cu_free_f32) use
// the STREAM-ORDERED allocator (cudaMallocAsync/cudaFreeAsync): a freed chain
// intermediate returns to the pool and is reused by the next alloc without a
// device sync, and the ordering is the stream's (free is queued after the matmul
// that read the buffer). The gA/gB/gC pool (host-facing single matmuls) stays on
// plain cudaMalloc — grow-only, reallocated rarely.
//
// A single mutex serializes enqueue, so the shared handle, stream and pool are
// safe under concurrent callers.

#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>    // driver API (cuLaunchKernel), for nvrtc-compiled kernels
#include <nvrtc.h>   // runtime CUDA-C→PTX compilation (no nvcc needed)
#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static cublasHandle_t gHandle = NULL;
static cudaStream_t gStream = NULL;
static CUcontext gCtx = NULL;    // runtime's primary context, retained for driver-API launches
static CUfunction gGelu = NULL;  // lazily nvrtc-compiled

static float *gA = NULL, *gB = NULL, *gC = NULL;
static size_t gACap = 0, gBCap = 0, gCCap = 0; // capacities in bytes

static int ensure_init(void) {
    if (gHandle != NULL) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    if (cudaStreamCreate(&gStream) != cudaSuccess) { gStream = NULL; return -1; }
    if (cublasCreate(&gHandle) != CUBLAS_STATUS_SUCCESS) { gHandle = NULL; return -1; }
    if (cublasSetStream(gHandle, gStream) != CUBLAS_STATUS_SUCCESS) { return -1; }
    // Retain the runtime's primary context so driver-API kernel launches
    // (nvrtc-compiled) share it with the runtime allocations/stream. The driver
    // API needs cuInit + an explicit current context per thread (see cu_gelu_f32).
    {
        CUdevice dev;
        if (cuInit(0) != CUDA_SUCCESS) return -1;
        if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return -1;
        if (cuDevicePrimaryCtxRetain(&gCtx, dev) != CUDA_SUCCESS) return -1;
    }
    return 0;
}

// ensure_gelu_kernel lazily compiles the GELU kernel from CUDA-C source with
// nvrtc (targeting the device's own compute capability) and loads it via the
// driver API. gCtx must be current on the calling thread.
static int ensure_gelu_kernel(void) {
    if (gGelu) return 0;
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    char arch[40];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
    // exact GELU (matches backend/ref: 0.5·x·(1+erf(x/√2))); erff is a CUDA builtin.
    const char* src =
        "extern \"C\" __global__ void gelu_f32(float* x, int n){\n"
        "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "  if (i < n){ float v = x[i]; x[i] = 0.5f*v*(1.0f+erff(v*0.7071067811865476f)); }\n"
        "}\n";
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, "gelu.cu", 0, NULL, NULL) != NVRTC_SUCCESS) return -1;
    const char* opts[1] = { arch };
    if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return -2; }
    size_t ptxSize = 0;
    if (nvrtcGetPTXSize(prog, &ptxSize) != NVRTC_SUCCESS) { nvrtcDestroyProgram(&prog); return -3; }
    char* ptx = (char*)malloc(ptxSize);
    if (!ptx) { nvrtcDestroyProgram(&prog); return -4; }
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);
    CUmodule mod;
    CUresult lr = cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL);
    free(ptx);
    if (lr != CUDA_SUCCESS) return -6;
    if (cuModuleGetFunction(&gGelu, mod, "gelu_f32") != CUDA_SUCCESS) return -7;
    return 0;
}

int cu_gelu_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; } // bind ctx to this thread
    if (ensure_gelu_kernel() != 0) { rc = -2; goto done; }
    {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        void *args[2];
        args[0] = &d;
        args[1] = &n;
        // launch on gStream (a cudaStream_t is a CUstream) → ordered with the
        // matmuls, no sync; the terminal cu_download_f32 is the barrier.
        CUresult klr = cuLaunchKernel(gGelu, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL);
        rc = (klr == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
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

    if (cudaMemcpyAsync(gA, A, aLen, cudaMemcpyHostToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }
    if (cudaMemcpyAsync(gB, B, bLen, cudaMemcpyHostToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }

    // Row-major C[M,N]=A·B via column-major C^T = B^T·A^T (see comment above).
    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     gB, N,
                     gA, K,
                     &beta,
                     gC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    if (cudaMemcpyAsync(C, gC, cLen, cudaMemcpyDeviceToHost, gStream) != cudaSuccess) { rc = -6; goto done; }
    if (cudaStreamSynchronize(gStream) != cudaSuccess) { rc = -5; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// Resident-B (§V14 Phase-1). cu_upload_f32 owns a stream-ordered device buffer
// (NOT the gA/gB/gC pool) so it can outlive individual matmuls; caller frees it.
void* cu_upload_f32(const float* src, int n) {
    void* d = NULL;
    size_t sz = (size_t)n * sizeof(float);
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { goto done; }
    if (cudaMallocAsync(&d, sz, gStream) != cudaSuccess) { d = NULL; goto done; }
    if (cudaMemcpyAsync(d, src, sz, cudaMemcpyHostToDevice, gStream) != cudaSuccess ||
        cudaStreamSynchronize(gStream) != cudaSuccess) {
        cudaFreeAsync(d, gStream);
        d = NULL;
    }
done:
    pthread_mutex_unlock(&gLock);
    return d;
}

void cu_free_f32(void* dptr) {
    if (!dptr) return;
    pthread_mutex_lock(&gLock);
    cudaFreeAsync(dptr, gStream);
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

    if (cudaMemcpyAsync(gA, A, aLen, cudaMemcpyHostToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }

    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     (const float*)dB, N,
                     gA, K,
                     &beta,
                     gC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    if (cudaMemcpyAsync(C, gC, cLen, cudaMemcpyDeviceToHost, gStream) != cudaSuccess) { rc = -6; goto done; }
    if (cudaStreamSynchronize(gStream) != cudaSuccess) { rc = -5; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// Fully-device path (§V14 Phase-2). cu_alloc_f32 / cu_download_f32 use the same
// stream-ordered allocator + stream as the chain, so allocs/frees pipeline.
void* cu_alloc_f32(int n) {
    void* d = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0) {
        if (cudaMallocAsync(&d, (size_t)n * sizeof(float), gStream) != cudaSuccess) d = NULL;
    }
    pthread_mutex_unlock(&gLock);
    return d;
}

int cu_download_f32(const void* dsrc, float* dst, int n) {
    int rc = -6;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    // D2H on the stream + one sync = the chain's single terminal barrier; any
    // queued Sgemm error also surfaces here.
    if (cudaMemcpyAsync(dst, dsrc, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost, gStream) != cudaSuccess) { goto done; }
    rc = (cudaStreamSynchronize(gStream) == cudaSuccess) ? 0 : -5;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_f32_ddd: dC = dA·dB, all resident, queued on the stream with NO sync —
// a chain of these pipelines end to end; cu_download_f32 is the barrier.
int cu_matmul_f32_ddd(const void* dA, const void* dB, void* dC, int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    int rc = -2;

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }

    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     (const float*)dB, N,
                     (const float*)dA, K,
                     &beta,
                     (float*)dC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}
