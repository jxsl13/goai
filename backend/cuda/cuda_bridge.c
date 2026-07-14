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
static CUcontext gCtx = NULL; // runtime's primary context, retained for driver-API launches
static CUfunction gGelu = NULL, gSilu = NULL, gAdd = NULL, gMul = NULL, gRms = NULL, gSoftmax = NULL, gRope = NULL, gCausal = NULL, gCausalMH = NULL, gEmbed = NULL, gSwiglu = NULL; // lazily nvrtc-compiled

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

// compile_kernel compiles one CUDA-C source with nvrtc (targeting the device's
// own compute capability), loads it via the driver API, and returns the named
// entry point in *out. gCtx must be current on the calling thread.
static int compile_kernel(const char* src, const char* name, const char* entry, CUfunction* out) {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0);
    char arch[40];
    snprintf(arch, sizeof(arch), "--gpu-architecture=compute_%d%d", major, minor);
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, name, 0, NULL, NULL) != NVRTC_SUCCESS) return -1;
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
    if (cuModuleGetFunction(out, mod, entry) != CUDA_SUCCESS) return -7;
    return 0;
}

// launch_unary runs an in-place elementwise kernel (float* x, int n) on gStream.
static int launch_unary(CUfunction fn, void* d, int n) {
    int threads = 256, blocks = (n + threads - 1) / threads;
    void* args[2];
    args[0] = &d;
    args[1] = &n;
    return (cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
}

int cu_gelu_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // exact GELU (matches backend/ref: 0.5·x·(1+erf(x/√2))); erff is a CUDA builtin.
    if (!gGelu && compile_kernel(
                      "extern \"C\" __global__ void gelu_f32(float* x, int n){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (i < n){ float v = x[i]; x[i] = 0.5f*v*(1.0f+erff(v*0.7071067811865476f)); }\n"
                      "}\n",
                      "gelu.cu", "gelu_f32", &gGelu) != 0) { rc = -2; goto done; }
    rc = launch_unary(gGelu, d, n);
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_silu_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // x·sigmoid(x); stable sigmoid (branch at 0) matching backend/ref's sigmoid.
    if (!gSilu && compile_kernel(
                      "extern \"C\" __global__ void silu_f32(float* x, int n){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (i < n){ float v = x[i];\n"
                      "    float s = v>=0.0f ? 1.0f/(1.0f+expf(-v)) : expf(v)/(1.0f+expf(v));\n"
                      "    x[i] = v*s; }\n"
                      "}\n",
                      "silu.cu", "silu_f32", &gSilu) != 0) { rc = -2; goto done; }
    rc = launch_unary(gSilu, d, n);
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_add_f32: dst += src elementwise (residual connection), on gStream.
int cu_add_f32(void* dst, const void* src, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAdd && compile_kernel(
                     "extern \"C\" __global__ void add_f32(float* dst, const float* src, int n){\n"
                     "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                     "  if (i < n){ dst[i] += src[i]; }\n"
                     "}\n",
                     "add.cu", "add_f32", &gAdd) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (n + threads - 1) / threads;
        void* args[3];
        args[0] = &dst;
        args[1] = &src;
        args[2] = &n;
        rc = (cuLaunchKernel(gAdd, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_swiglu_f32: gate[i] = SiLU(gate[i]) * up[i], fused — the SwiGLU nonlinearity
// in ONE pass/launch instead of SiLU then Mul (halves the elementwise memory
// traffic + saves a kernel launch). Stable sigmoid matching backend/ref.
int cu_swiglu_f32(void* gate, const void* up, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gSwiglu && compile_kernel(
                        "extern \"C\" __global__ void swiglu_f32(float* gate, const float* up, int n){\n"
                        "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                        "  if (i < n){ float v = gate[i];\n"
                        "    float s = v>=0.0f ? 1.0f/(1.0f+expf(-v)) : expf(v)/(1.0f+expf(v));\n"
                        "    gate[i] = v*s*up[i]; }\n"
                        "}\n",
                        "swiglu.cu", "swiglu_f32", &gSwiglu) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (n + threads - 1) / threads;
        void* args[3];
        args[0] = &gate;
        args[1] = &up;
        args[2] = &n;
        rc = (cuLaunchKernel(gSwiglu, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_mul_f32: dst *= src elementwise (e.g. the SwiGLU gate·up product), on gStream.
int cu_mul_f32(void* dst, const void* src, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gMul && compile_kernel(
                     "extern \"C\" __global__ void mul_f32(float* dst, const float* src, int n){\n"
                     "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                     "  if (i < n){ dst[i] *= src[i]; }\n"
                     "}\n",
                     "mul.cu", "mul_f32", &gMul) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (n + threads - 1) / threads;
        void* args[3];
        args[0] = &dst;
        args[1] = &src;
        args[2] = &n;
        rc = (cuLaunchKernel(gMul, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_rmsnorm_f32: in-place RMSNorm over the last axis, y = x/√(mean(x²)+eps)·w,
// one thread-block per row with a shared-memory reduction. Sum-of-squares is
// accumulated in DOUBLE to match backend/ref's f64 accumulation (§V10) closely.
int cu_rmsnorm_f32(void* x, const void* gamma, int rows, int cols, float eps) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRms && compile_kernel(
                     "extern \"C\" __global__ void rmsnorm_f32(float* x, const float* w, int rows, int cols, float eps){\n"
                     "  int row = blockIdx.x; if (row>=rows) return;\n"
                     "  extern __shared__ double sh[];\n"
                     "  int t=threadIdx.x, nt=blockDim.x;\n"
                     "  float* xr = x + (size_t)row*cols;\n"
                     "  double local=0.0;\n"
                     "  for (int j=t;j<cols;j+=nt){ double v=xr[j]; local+=v*v; }\n"
                     "  sh[t]=local; __syncthreads();\n"
                     "  for (int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                     "  double ms = sh[0]/(double)cols;\n"
                     "  float inv = (float)(1.0/sqrt(ms+(double)eps));\n"
                     "  for (int j=t;j<cols;j+=nt){ xr[j] = xr[j]*inv*w[j]; }\n"
                     "}\n",
                     "rmsnorm.cu", "rmsnorm_f32", &gRms) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[5];
        args[0] = &x;
        args[1] = &gamma;
        args[2] = &rows;
        args[3] = &cols;
        args[4] = &eps;
        rc = (cuLaunchKernel(gRms, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_softmax_f32: in-place stable softmax over the last axis, y=exp(x−max)/Σexp,
// one block per row with two shared-memory reductions (max, then sum-exp). The
// exp sum is accumulated in DOUBLE (matches backend/ref's f64, §V10).
int cu_softmax_f32(void* x, int rows, int cols) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gSoftmax && compile_kernel(
                         "extern \"C\" __global__ void softmax_f32(float* x, int rows, int cols){\n"
                         "  int row=blockIdx.x; if(row>=rows) return;\n"
                         "  extern __shared__ double sh[];\n"
                         "  int t=threadIdx.x, nt=blockDim.x;\n"
                         "  float* xr = x + (size_t)row*cols;\n"
                         "  double m=-1e300;\n"
                         "  for(int j=t;j<cols;j+=nt){ double v=xr[j]; if(v>m)m=v; }\n"
                         "  sh[t]=m; __syncthreads();\n"
                         "  for(int s=nt/2;s>0;s>>=1){ if(t<s && sh[t+s]>sh[t]) sh[t]=sh[t+s]; __syncthreads(); }\n"
                         "  double rowmax=sh[0]; __syncthreads();\n"
                         "  double local=0.0;\n"
                         "  for(int j=t;j<cols;j+=nt){ double e=exp((double)xr[j]-rowmax); xr[j]=(float)e; local+=e; }\n"
                         "  sh[t]=local; __syncthreads();\n"
                         "  for(int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                         "  double inv=1.0/sh[0];\n"
                         "  for(int j=t;j<cols;j+=nt){ xr[j]=(float)(xr[j]*inv); }\n"
                         "}\n",
                         "softmax.cu", "softmax_f32", &gSoftmax) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[3];
        args[0] = &x;
        args[1] = &rows;
        args[2] = &cols;
        rc = (cuLaunchKernel(gSoftmax, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_causal_scale_f32: for attention scores x[qRows, kCols], multiply by scale
// and apply a causal mask — element (i,j) is set to −inf when key j is in the
// future of query i, i.e. j > i + offset. offset = kCols − qRows aligns a short
// query window to the end of the key sequence (prefill of the last qRows rows).
int cu_causal_scale_f32(void* x, int qRows, int kCols, float scale, int offset) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gCausal && compile_kernel(
                        "extern \"C\" __global__ void causal_scale_f32(float* x, int rows, int cols, float scale, int offset){\n"
                        "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                        "  long total = (long)rows*cols;\n"
                        "  if (gid >= total) return;\n"
                        "  int i = (int)(gid / cols), j = (int)(gid % cols);\n"
                        "  x[gid] = (j > i + offset) ? __int_as_float(0xff800000) : x[gid]*scale;\n"
                        "}\n",
                        "causal.cu", "causal_scale_f32", &gCausal) != 0) { rc = -2; goto done; }
    {
        long total = (long)qRows * kCols;
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[5];
        args[0] = &x;
        args[1] = &qRows;
        args[2] = &kCols;
        args[3] = &scale;
        args[4] = &offset;
        rc = (cuLaunchKernel(gCausal, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_rope_f32: in-place rotary position embeddings on x[seq, heads*hd] (HF
// rotate_half; §R28), one thread per (position, head, dim-pair). inv is the
// resident [hd/2] frequency table and posDiv the position divisor — both from
// backend.RoPEFreqs on the host, so PI/YaRN scaling matches backend/ref exactly.
// Angles are computed in double.
int cu_rope_f32(void* x, const void* inv, int seq, int heads, int hd, int posOffset, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRope && compile_kernel(
                      "extern \"C\" __global__ void rope_f32(float* x, const float* inv, int seq, int heads, int hd, int posOffset, double posDiv){\n"
                      "  int half = hd/2;\n"
                      "  long total = (long)seq*heads*half;\n"
                      "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (gid >= total) return;\n"
                      "  int i = (int)(gid % half);\n"
                      "  int h = (int)((gid / half) % heads);\n"
                      "  int p = (int)(gid / ((long)half*heads));\n"
                      "  double pos = (double)(posOffset + p) / posDiv;\n"
                      "  double ang = pos * (double)inv[i];\n"
                      "  double c = cos(ang), s = sin(ang);\n"
                      "  float* xr = x + (size_t)p*heads*hd + (size_t)h*hd;\n"
                      "  double qi = xr[i], qih = xr[i+half];\n"
                      "  xr[i] = (float)(qi*c - qih*s);\n"
                      "  xr[i+half] = (float)(qih*c + qi*s);\n"
                      "}\n",
                      "rope.cu", "rope_f32", &gRope) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (hd / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &x;
        args[1] = &inv;
        args[2] = &seq;
        args[3] = &heads;
        args[4] = &hd;
        args[5] = &posOffset;
        args[6] = &posDiv;
        rc = (cuLaunchKernel(gRope, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_upload_i32 copies n int32 to a fresh device buffer (token ids for embed).
void* cu_upload_i32(const int* src, int n) {
    void* d = NULL;
    size_t sz = (size_t)n * sizeof(int);
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

// cu_embed_f32: out[i,:] = table[ids[i],:] — the input embedding row gather. One
// thread per output element; table is [vocab,d] resident, ids [seq] resident.
int cu_embed_f32(const void* dTable, const void* dIds, void* dOut, int seq, int d) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gEmbed && compile_kernel(
                       "extern \"C\" __global__ void embed_f32(const float* table, const int* ids, float* out, int seq, int d){\n"
                       "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                       "  long total = (long)seq*d;\n"
                       "  if (gid >= total) return;\n"
                       "  int i = (int)(gid / d), dim = (int)(gid % d);\n"
                       "  out[gid] = table[(size_t)ids[i]*d + dim];\n"
                       "}\n",
                       "embed.cu", "embed_f32", &gEmbed) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * d;
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[5];
        args[0] = &dTable;
        args[1] = &dIds;
        args[2] = &dOut;
        args[3] = &seq;
        args[4] = &d;
        rc = (cuLaunchKernel(gEmbed, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_copy_rows device→device copies nElems floats from src into dst starting at
// float offset dstOffset — the KV-cache append: write a new token's key/value
// rows into the contiguous cache buffer just past the rows already stored.
int cu_copy_rows(void* dst, const void* src, int dstOffset, int nElems) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemcpyAsync((float*)dst + dstOffset, src, (size_t)nElems * sizeof(float),
                        cudaMemcpyDeviceToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_clone_f32 allocates a new device buffer and device→device copies n floats
// into it (for a residual branch: keep x while an in-place op runs on a copy).
void* cu_clone_f32(const void* src, int n) {
    void* d = NULL;
    size_t sz = (size_t)n * sizeof(float);
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { goto done; }
    if (cudaMallocAsync(&d, sz, gStream) != cudaSuccess) { d = NULL; goto done; }
    if (cudaMemcpyAsync(d, src, sz, cudaMemcpyDeviceToDevice, gStream) != cudaSuccess) {
        cudaFreeAsync(d, gStream);
        d = NULL;
    }
done:
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

// ---- Multi-head attention (batched, strided). Q/K/V are [seq, heads*hd]; each
// head's [seq,hd] slice starts at column h*hd with row stride W=heads*hd, so a
// single cublas strided-batched Sgemm with ld=W, stride=hd, batch=heads does all
// heads at once. Scores are [heads, seq, seq] contiguous (head-major).

// cu_mha_scores: scores[h] = Q[h]·K[h]ᵀ for every head. Row-major A·Bᵀ batched
// = column-major sgemm(OP_T,OP_N) with the QKᵀ idiom, ld=W (embedded head).
int cu_mha_scores(const void* dQ, const void* dK, void* dScores, int seq, int heads, int hd) {
    const float alpha = 1.0f, beta = 0.0f;
    int W = heads * hd, rc = -2;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cublasSgemmStridedBatched(gHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                  seq, seq, hd, &alpha,
                                  (const float*)dK, W, hd,
                                  (const float*)dQ, W, hd, &beta,
                                  (float*)dScores, seq, (long long)seq * seq,
                                  heads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_mha_out: out[h] = scores[h]·V[h] for every head, written back into the
// [seq, heads*hd] output (each head at column h*hd). Plain A·B batched.
int cu_mha_out(const void* dScores, const void* dV, void* dOut, int seq, int heads, int hd) {
    const float alpha = 1.0f, beta = 0.0f;
    int W = heads * hd, rc = -2;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cublasSgemmStridedBatched(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  hd, seq, seq, &alpha,
                                  (const float*)dV, W, hd,
                                  (const float*)dScores, seq, (long long)seq * seq, &beta,
                                  (float*)dOut, W, hd,
                                  heads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// GQA: qHeads query heads share kvHeads key/value heads (group = qHeads/kvHeads).
// Query head h uses kv head h/group — a non-constant batch stride on K/V, so the
// strided path can't express it. cublasSgemmBatched takes explicit device
// pointer arrays instead: build them (query h → its kv head), upload, call.
// cu_gqa_scores: scores[h] = Q[h]·K[h/group]ᵀ for every query head. Q is
// [seqQ, WQ], K is [seqKV, WKV]; scores[h] is [seqQ, seqKV]. Full prefill passes
// seqQ==seqKV; a KV-cache step passes seqQ (new tokens) < seqKV (cache length).
int cu_gqa_scores(const void* dQ, const void* dK, void* dScores, int seqQ, int seqKV, int qHeads, int kvHeads, int hd) {
    const float alpha = 1.0f, beta = 0.0f;
    int group = qHeads / kvHeads, WQ = qHeads * hd, WKV = kvHeads * hd, rc = -2;
    const float **hK = NULL, **hQ = NULL, **dKa = NULL, **dQa = NULL;
    float **hC = NULL, **dCa = NULL;
    size_t asz = (size_t)qHeads * sizeof(void*);

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    hK = (const float**)malloc(asz); hQ = (const float**)malloc(asz); hC = (float**)malloc(asz);
    if (!hK || !hQ || !hC) { rc = -9; goto done; }
    for (int h = 0; h < qHeads; h++) {
        hK[h] = (const float*)dK + (size_t)(h / group) * hd;
        hQ[h] = (const float*)dQ + (size_t)h * hd;
        hC[h] = (float*)dScores + (size_t)h * seqQ * seqKV;
    }
    if (cudaMalloc((void**)&dKa, asz) != cudaSuccess || cudaMalloc((void**)&dQa, asz) != cudaSuccess ||
        cudaMalloc((void**)&dCa, asz) != cudaSuccess) { rc = -9; goto done; }
    cudaMemcpy(dKa, hK, asz, cudaMemcpyHostToDevice);
    cudaMemcpy(dQa, hQ, asz, cudaMemcpyHostToDevice);
    cudaMemcpy(dCa, hC, asz, cudaMemcpyHostToDevice);
    if (cublasSgemmBatched(gHandle, CUBLAS_OP_T, CUBLAS_OP_N, seqKV, seqQ, hd, &alpha,
                           dKa, WKV, dQa, WQ, &beta, dCa, seqKV, qHeads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;
done:
    if (dKa) cudaFree((void*)dKa);
    if (dQa) cudaFree((void*)dQa);
    if (dCa) cudaFree((void*)dCa);
    free(hK); free(hQ); free(hC);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_gqa_out: out[h] = scores[h]·V[h/group] for every query head, into [seqQ,WQ].
// scores[h] is [seqQ, seqKV], V is [seqKV, WKV]. Full prefill passes seqQ==seqKV.
int cu_gqa_out(const void* dScores, const void* dV, void* dOut, int seqQ, int seqKV, int qHeads, int kvHeads, int hd) {
    const float alpha = 1.0f, beta = 0.0f;
    int group = qHeads / kvHeads, WQ = qHeads * hd, WKV = kvHeads * hd, rc = -2;
    const float **hV = NULL, **hS = NULL, **dVa = NULL, **dSa = NULL;
    float **hO = NULL, **dOa = NULL;
    size_t asz = (size_t)qHeads * sizeof(void*);

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    hV = (const float**)malloc(asz); hS = (const float**)malloc(asz); hO = (float**)malloc(asz);
    if (!hV || !hS || !hO) { rc = -9; goto done; }
    for (int h = 0; h < qHeads; h++) {
        hV[h] = (const float*)dV + (size_t)(h / group) * hd;
        hS[h] = (const float*)dScores + (size_t)h * seqQ * seqKV;
        hO[h] = (float*)dOut + (size_t)h * hd;
    }
    if (cudaMalloc((void**)&dVa, asz) != cudaSuccess || cudaMalloc((void**)&dSa, asz) != cudaSuccess ||
        cudaMalloc((void**)&dOa, asz) != cudaSuccess) { rc = -9; goto done; }
    cudaMemcpy(dVa, hV, asz, cudaMemcpyHostToDevice);
    cudaMemcpy(dSa, hS, asz, cudaMemcpyHostToDevice);
    cudaMemcpy(dOa, hO, asz, cudaMemcpyHostToDevice);
    if (cublasSgemmBatched(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, hd, seqQ, seqKV, &alpha,
                           dVa, WKV, dSa, seqKV, &beta, dOa, WQ, qHeads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;
done:
    if (dVa) cudaFree((void*)dVa);
    if (dSa) cudaFree((void*)dSa);
    if (dOa) cudaFree((void*)dOa);
    free(hV); free(hS); free(hO);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_causal_scale_mh: scale + causal mask on scores[heads, seqQ, seqKV]
// (head-major) — per head, mask key j for query row i when j > i + offset. Full
// prefill passes seqQ==seqKV; a KV-cache step passes seqQ<=seqKV with the query
// rows being the tail of the context, offset = seqKV-seqQ (query row i is at
// absolute position i+offset, so it attends keys 0..i+offset).
int cu_causal_scale_mh(void* x, int heads, int seqQ, int seqKV, float scale, int offset) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gCausalMH && compile_kernel(
                          "extern \"C\" __global__ void causal_scale_mh(float* x, int heads, int seqQ, int seqKV, float scale, int offset){\n"
                          "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                          "  long total = (long)heads*seqQ*seqKV;\n"
                          "  if (gid >= total) return;\n"
                          "  int j = (int)(gid % seqKV);\n"
                          "  int i = (int)((gid / seqKV) % seqQ);\n"
                          "  x[gid] = (j > i + offset) ? __int_as_float(0xff800000) : x[gid]*scale;\n"
                          "}\n",
                          "causal_mh.cu", "causal_scale_mh", &gCausalMH) != 0) { rc = -2; goto done; }
    {
        long total = (long)heads * seqQ * seqKV;
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[6];
        args[0] = &x;
        args[1] = &heads;
        args[2] = &seqQ;
        args[3] = &seqKV;
        args[4] = &scale;
        args[5] = &offset;
        rc = (cuLaunchKernel(gCausalMH, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_f32_ddd_bt: dC[M,N] = dA[M,K]·dB[N,K]ᵀ, all resident (attention QKᵀ:
// A=Q[seq,hd], B=K[seq,hd] → scores[seq,seq]). Row-major A·Bᵀ maps to the
// column-major call sgemm(OP_T,OP_N, N,M,K, B(ld=K), A(ld=K), C(ld=N)) — the
// transpose of A·B's idiom with the first operand transposed (§R43).
int cu_matmul_f32_ddd_bt(const void* dA, const void* dB, void* dC, int M, int K, int N) {
    const float alpha = 1.0f, beta = 0.0f;
    cublasStatus_t st;
    int rc = -2;

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }

    st = cublasSgemm(gHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     (const float*)dB, K,
                     (const float*)dA, K,
                     &beta,
                     (float*)dC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}
