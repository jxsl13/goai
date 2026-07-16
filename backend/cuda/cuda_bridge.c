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
#include <string.h> // strstr — GPU-class name check (cu_gpu_is_geforce)
#include <stdio.h>

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
static cublasHandle_t gHandle = NULL;
static float *gOne = NULL, *gZero = NULL; // device 1.0f/0.0f — cuBLAS DEVICE pointer mode (graph-capture-safe alpha/beta)
static cudaStream_t gStream = NULL;
static CUcontext gCtx = NULL; // runtime's primary context, retained for driver-API launches
static CUfunction gGelu = NULL, gSilu = NULL, gAdd = NULL, gMul = NULL, gRms = NULL, gSoftmax = NULL, gRope = NULL, gCausal = NULL, gCausalMH = NULL, gEmbed = NULL, gSwiglu = NULL, gAttnSoftmax = NULL, gQgemv = NULL, gQgemv4 = NULL, gQgemv4k = NULL, gQgemv5k = NULL, gQgemv6k = NULL, gQgemv3k = NULL, gQgemv2k = NULL, gQgemv40 = NULL, gQgemvI4nl = NULL, gQgemvI4xs = NULL, gQgemvMxfp4 = NULL, gCvtF16 = NULL, gCvtFrom16 = NULL; // lazily nvrtc-compiled
static CUfunction gRopeDpos = NULL, gAttnSoftmaxDpos = NULL, gAppendDpos = NULL; // device-position (graph-capturable) twins
static CUfunction gGqaFlashPart = NULL, gGqaFlashMerge = NULL; // flash decode: GQA K/V-shared split-K partials + merge
static CUfunction gGqaFlashPartF16 = NULL, gAppendDposF16 = NULL; // f16 KV-cache twins (u16 storage, f32 compute)
static CUfunction gArgmax = NULL, gLayerNorm = NULL, gAddBias = NULL; // greedy argmax; layernorm; broadcast bias-add
static CUfunction gCopy2d = NULL; // strided 2D copy (fused-QKV band extraction, llamagpu adapter)
static int ensure_init(void); // fwd decls (defined later)
static int compile_kernel(const char* src, const char* name, const char* entry, CUfunction* out);

// cu_blit: contiguous device→device copy of n floats, src[srcOff:] → dst[dstOff:]
// (the llamagpu recorder Blit — offset band moves in the fused-QKV path).
int cu_blit(void* dst, int dstOff, const void* src, int srcOff, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemcpyAsync((float*)dst + dstOff, (const float*)src + srcOff, (size_t)n * sizeof(float),
                        cudaMemcpyDeviceToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_copy2d: copy a rows×rowFloats sub-matrix with independent src/dst row strides
// (the llamagpu recorder Copy2D — extracts the q/k/v bands from a fused-QKV output).
int cu_copy2d(void* dst, int dstOff, int dstStride, const void* src, int srcOff, int srcStride, int rows, int rowFloats) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gCopy2d && compile_kernel(
                        "extern \"C\" __global__ void copy2d(float* dst, int dstOff, int dstStride, const float* src, int srcOff, int srcStride, int rows, int rowFloats){\n"
                        "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                        "  long total = (long)rows*rowFloats; if (gid >= total) return;\n"
                        "  int r = (int)(gid / rowFloats), c = (int)(gid % rowFloats);\n"
                        "  dst[(size_t)dstOff + (size_t)r*dstStride + c] = src[(size_t)srcOff + (size_t)r*srcStride + c];\n"
                        "}\n", "copy2d.cu", "copy2d", &gCopy2d) != 0) { rc = -2; goto done; }
    {
        long total = (long)rows * rowFloats;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8] = {&dst, &dstOff, &dstStride, &src, &srcOff, &srcStride, &rows, &rowFloats};
        rc = (cuLaunchKernel(gCopy2d, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}
static int ensure_init(void);
static int compile_kernel(const char* src, const char* name, const char* entry, CUfunction* out);

// cu_argmax_f32 returns argmax_i x[i] (greedy token) — one block reduction over
// the [n] logits on device, downloading only the 4-byte index instead of the full
// [1,vocab] logit vector (128 KB for vocab=32000) every decode token.
int cu_argmax_f32(const void* x, int n) {
    static int* dIdx = NULL;
    int rc = -1, host = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { goto done; }
    if (dIdx == NULL && cudaMalloc((void**)&dIdx, sizeof(int)) != cudaSuccess) { goto done; }
    if (!gArgmax && compile_kernel(
            "extern \"C\" __global__ void argmax_f32(const float* x, int n, int* outIdx){\n"
            "  extern __shared__ char sh[];\n"
            "  float* sv = (float*)sh; int* si = (int*)(sv + blockDim.x);\n"
            "  int t = threadIdx.x, nt = blockDim.x;\n"
            "  float bv = -1e30f; int bi = 0;\n"
            "  for (int i = t; i < n; i += nt){ float v = x[i]; if (v > bv){ bv = v; bi = i; } }\n"
            "  sv[t] = bv; si[t] = bi; __syncthreads();\n"
            "  for (int s = nt/2; s > 0; s >>= 1){ if (t < s && sv[t+s] > sv[t]){ sv[t] = sv[t+s]; si[t] = si[t+s]; } __syncthreads(); }\n"
            "  if (t == 0) *outIdx = si[0];\n"
            "}\n", "argmax.cu", "argmax_f32", &gArgmax) != 0) { goto done; }
    {
        int threads = 256;
        size_t shmem = (size_t)threads * (sizeof(float) + sizeof(int));
        void* args[3] = {&x, &n, &dIdx};
        if (cuLaunchKernel(gArgmax, 1, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) != CUDA_SUCCESS) { goto done; }
        if (cudaMemcpyAsync(&host, dIdx, sizeof(int), cudaMemcpyDeviceToHost, gStream) != cudaSuccess) { goto done; }
        if (cudaStreamSynchronize(gStream) != cudaSuccess) { goto done; }
        rc = 0;
    }
done:
    pthread_mutex_unlock(&gLock);
    return (rc == 0) ? host : -1;
}
static CUfunction gBuildPtrs = NULL; // device-side batched pointer-array builder (graph-capturable GQA)
static int compile_kernel(const char* src, const char* name, const char* entry, CUfunction* out); // fwd decl

// cu_build_batch_ptrs computes the cublas*Batched pointer arrays ON DEVICE from the
// base buffer pointers, so there is no host→device pointer memcpy. That memcpy was
// NOT graph-capture-safe: the shared host source array is overwritten by later
// layers during capture, so every replayed layer read the last layer's pointers.
// A grouped (h/group), B/C per-head. Caller holds gLock and has set gCtx current.
static int cu_build_batch_ptrs(const void* A, const void* B, const void* C,
                               void* sA, void* sB, void* sC,
                               int n, int group, long strideA, long strideB, long strideC) {
    if (!gBuildPtrs && compile_kernel(
            "extern \"C\" __global__ void build_batch_ptrs(const float* A, const float* B, const float* C, const float** sA, const float** sB, const float** sC, int n, int group, long strideA, long strideB, long strideC){\n"
            "  int h = blockIdx.x*blockDim.x + threadIdx.x;\n"
            "  if (h >= n) return;\n"
            "  sA[h] = A + (size_t)(h/group)*strideA;\n"
            "  sB[h] = B + (size_t)h*strideB;\n"
            "  sC[h] = C + (size_t)h*strideC;\n"
            "}\n", "build_ptrs.cu", "build_batch_ptrs", &gBuildPtrs) != 0) return -2;
    int threads = 64, blocks = (n + threads - 1) / threads; if (blocks < 1) blocks = 1;
    void* args[11] = {&A, &B, &C, &sA, &sB, &sC, &n, &group, &strideA, &strideB, &strideC};
    return (cuLaunchKernel(gBuildPtrs, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
}

static float *gA = NULL, *gB = NULL, *gC = NULL;
static size_t gACap = 0, gBCap = 0, gCCap = 0; // capacities in bytes

static int ensure_init(void) {
    if (gHandle != NULL) return 0;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    if (cudaStreamCreate(&gStream) != cudaSuccess) { gStream = NULL; return -1; }
    if (cublasCreate(&gHandle) != CUBLAS_STATUS_SUCCESS) { gHandle = NULL; return -1; }
    if (cublasSetStream(gHandle, gStream) != CUBLAS_STATUS_SUCCESS) { return -1; }
    // Give cuBLAS a persistent user workspace so it does NOT allocate a workspace
    // lazily on first use — a lazy alloc inside a CUDA-graph stream capture breaks
    // the captured graph. 4 MB matches cuBLAS's own default (§PERF graph decode).
    {
        static void* gCublasWs = NULL;
        const size_t wsSize = 4u * 1024u * 1024u;
        if (gCublasWs == NULL && cudaMalloc(&gCublasWs, wsSize) != cudaSuccess) return -1;
        if (cublasSetWorkspace(gHandle, gCublasWs, wsSize) != CUBLAS_STATUS_SUCCESS) return -1;
    }
    // DEVICE pointer mode: alpha/beta are read from device memory at kernel
    // execution — required for cuBLAS calls to be captured into a CUDA graph
    // (host-pointer alpha/beta on the stack aren't valid at graph replay time).
    {
        float h1 = 1.0f, h0 = 0.0f;
        if (gOne == NULL && cudaMalloc((void**)&gOne, sizeof(float)) != cudaSuccess) return -1;
        if (gZero == NULL && cudaMalloc((void**)&gZero, sizeof(float)) != cudaSuccess) return -1;
        cudaMemcpy(gOne, &h1, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(gZero, &h0, sizeof(float), cudaMemcpyHostToDevice);
        if (cublasSetPointerMode(gHandle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS) return -1;
    }
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
// cu_rmsnorm_f32: out[row] = in[row]/rms(in[row]) · gamma. in==out is the valid
// in-place form; passing a distinct out normalizes without a preceding clone
// (the residual buffer stays intact) — one fewer launch + alloc per norm on the
// decode hot path (§PERF launch-bound).
int cu_rmsnorm_f32(const void* in, void* out, const void* gamma, int rows, int cols, float eps) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRms && compile_kernel(
                     "extern \"C\" __global__ void rmsnorm_f32(const float* in, float* out, const float* w, int rows, int cols, float eps){\n"
                     "  int row = blockIdx.x; if (row>=rows) return;\n"
                     "  extern __shared__ double sh[];\n"
                     "  int t=threadIdx.x, nt=blockDim.x;\n"
                     "  const float* xr = in + (size_t)row*cols;\n"
                     "  float* yr = out + (size_t)row*cols;\n"
                     "  double local=0.0;\n"
                     "  for (int j=t;j<cols;j+=nt){ double v=xr[j]; local+=v*v; }\n"
                     "  sh[t]=local; __syncthreads();\n"
                     "  for (int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                     "  double ms = sh[0]/(double)cols;\n"
                     "  float inv = (float)(1.0/sqrt(ms+(double)eps));\n"
                     "  for (int j=t;j<cols;j+=nt){ yr[j] = xr[j]*inv*w[j]; }\n"
                     "}\n",
                     "rmsnorm.cu", "rmsnorm_f32", &gRms) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[6];
        args[0] = &in;
        args[1] = &out;
        args[2] = &gamma;
        args[3] = &rows;
        args[4] = &cols;
        args[5] = &eps;
        rc = (cuLaunchKernel(gRms, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_layernorm_f32: out[r,j] = (x[r,j]−mean_r)·inv_r·γ_j + β_j over the last axis
// (torch LayerNorm, backend OpLayerNorm), mean/var accumulated in DOUBLE per row
// (§V10). One block per row, two shared-mem reductions (mean, then variance).
int cu_layernorm_f32(const void* in, void* out, const void* gamma, const void* beta, int rows, int cols, float eps) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gLayerNorm && compile_kernel(
                           "extern \"C\" __global__ void layernorm_f32(const float* in, float* out, const float* g, const float* b, int rows, int cols, float eps){\n"
                           "  int row = blockIdx.x; if (row>=rows) return;\n"
                           "  extern __shared__ double sh[];\n"
                           "  int t=threadIdx.x, nt=blockDim.x;\n"
                           "  const float* xr = in + (size_t)row*cols;\n"
                           "  float* yr = out + (size_t)row*cols;\n"
                           "  double s=0.0; for(int j=t;j<cols;j+=nt) s+=(double)xr[j];\n"
                           "  sh[t]=s; __syncthreads();\n"
                           "  for(int k=nt/2;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); }\n"
                           "  double mean=sh[0]/(double)cols; __syncthreads();\n"
                           "  double v=0.0; for(int j=t;j<cols;j+=nt){ double d=(double)xr[j]-mean; v+=d*d; }\n"
                           "  sh[t]=v; __syncthreads();\n"
                           "  for(int k=nt/2;k>0;k>>=1){ if(t<k) sh[t]+=sh[t+k]; __syncthreads(); }\n"
                           "  double var=sh[0]/(double)cols;\n"
                           "  float inv=(float)(1.0/sqrt(var+(double)eps));\n"
                           "  for(int j=t;j<cols;j+=nt){ yr[j]=(float)(((double)xr[j]-mean)*inv*(double)g[j]+(double)b[j]); }\n"
                           "}\n",
                           "layernorm.cu", "layernorm_f32", &gLayerNorm) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows; if (blocks < 1) blocks = 1;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[7] = {&in, &out, &gamma, &beta, &rows, &cols, &eps};
        rc = (cuLaunchKernel(gLayerNorm, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_addbias_f32: out[r,j] = x[r,j] + bias[j] (row-broadcast bias, backend
// OpAddBias) — the Qwen QKV-projection bias and GPT linear biases.
int cu_addbias_f32(const void* x, const void* bias, void* out, int rows, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAddBias && compile_kernel(
                         "extern \"C\" __global__ void addbias_f32(const float* x, const float* bias, float* out, int rows, int n){\n"
                         "  long gid=(long)blockIdx.x*blockDim.x+threadIdx.x;\n"
                         "  long total=(long)rows*n; if(gid>=total) return;\n"
                         "  int j=(int)(gid % n);\n"
                         "  out[gid]=x[gid]+bias[j];\n"
                         "}\n",
                         "addbias.cu", "addbias_f32", &gAddBias) != 0) { rc = -2; goto done; }
    {
        long total=(long)rows*n;
        int threads=256, blocks=(int)((total+threads-1)/threads); if(blocks<1) blocks=1;
        void* args[5] = {&x, &bias, &out, &rows, &n};
        rc = (cuLaunchKernel(gAddBias, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_attn_softmax: fused scale + causal-mask + stable softmax over attention
// scores[heads·seqQ, seqKV] (one block per query row). Folds what were three
// launches (cu_causal_scale_mh + cu_softmax_f32) into one on the attention hot
// path. Row r belongs to query position i = r % seqQ within its head; key j is
// masked when j > i + offset (offset = seqKV−seqQ for a KV window; offset ≥ seqKV
// disables masking). Valid scores are ×scale then softmaxed (double sum, §V10);
// masked entries become 0.
int cu_attn_softmax(void* x, int rows, int cols, float scale, int offset, int seqQ) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAttnSoftmax && compile_kernel(
                             "extern \"C\" __global__ void attn_softmax(float* x, int rows, int cols, float scale, int offset, int seqQ){\n"
                             "  int row=blockIdx.x; if(row>=rows) return;\n"
                             "  int lim = (row % seqQ) + offset; if(lim>=cols) lim=cols-1;\n"
                             "  extern __shared__ double sh[];\n"
                             "  int t=threadIdx.x, nt=blockDim.x;\n"
                             "  float* xr = x + (size_t)row*cols;\n"
                             "  double m=-1e300;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double v=(double)xr[j]*scale; if(v>m)m=v; } }\n"
                             "  sh[t]=m; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s && sh[t+s]>sh[t]) sh[t]=sh[t+s]; __syncthreads(); }\n"
                             "  double rowmax=sh[0]; __syncthreads();\n"
                             "  double local=0.0;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double e=exp((double)xr[j]*scale-rowmax); xr[j]=(float)e; local+=e; } else { xr[j]=0.0f; } }\n"
                             "  sh[t]=local; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                             "  double inv=1.0/sh[0];\n"
                             "  for(int j=t;j<=lim;j+=nt){ xr[j]=(float)(xr[j]*inv); }\n"
                             "}\n",
                             "attn_softmax.cu", "attn_softmax", &gAttnSoftmax) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows; if (blocks < 1) blocks = 1;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[6];
        args[0] = &x;
        args[1] = &rows;
        args[2] = &cols;
        args[3] = &scale;
        args[4] = &offset;
        args[5] = &seqQ;
        rc = (cuLaunchKernel(gAttnSoftmax, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_rope_f32_band: strided-band RoPE. Rotates `heads` heads (hd wide each) that
// start at float-element column `off` inside rows of stride `stride` floats, in
// place (HF rotate_half; §T613 fused-QKV path). This is cu_rope_f32 generalised
// with a row stride and a column offset, so the q and k bands of a single fused
// [seq, stride] QKV buffer rotate without being copied out first. Angles in double.
static CUfunction gRopeBand = NULL;
int cu_rope_f32_band(void* x, const void* inv, int seq, int stride, int off, int heads, int hd, int posOffset, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRopeBand && compile_kernel(
                      "extern \"C\" __global__ void rope_band(float* x, const float* inv, int seq, int stride, int off, int heads, int hd, int posOffset, double posDiv){\n"
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
                      "  float* xr = x + (size_t)p*stride + off + (size_t)h*hd;\n"
                      "  double qi = xr[i], qih = xr[i+half];\n"
                      "  xr[i] = (float)(qi*c - qih*s);\n"
                      "  xr[i+half] = (float)(qih*c + qi*s);\n"
                      "}\n",
                      "rope_band.cu", "rope_band", &gRopeBand) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (hd / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[9];
        args[0] = &x;
        args[1] = &inv;
        args[2] = &seq;
        args[3] = &stride;
        args[4] = &off;
        args[5] = &heads;
        args[6] = &hd;
        args[7] = &posOffset;
        args[8] = &posDiv;
        rc = (cuLaunchKernel(gRopeBand, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_mem_info reports the device free/total VRAM in bytes (cudaMemGetInfo) — the
// VRAM-budget probe for the T631 offload policy (how many resident layers fit).
int cu_mem_info(unsigned long long* freeB, unsigned long long* totalB) {
    size_t fr = 0, to = 0;
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0 && cudaMemGetInfo(&fr, &to) == cudaSuccess) {
        *freeB = (unsigned long long)fr;
        *totalB = (unsigned long long)to;
        rc = 0;
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_gpu_is_geforce: 1 if device 0 is a GeForce/consumer card (name contains "GeForce" or
// "TITAN"), else 0. GeForce/GA10x run FP32-accumulate tensor ops at HALF rate, so f16
// accumulate is a ~1.5-2× prefill win there (Tw61); datacenter cards (A100/A10/L4/T4/…) run
// f32 accumulate at full rate, where f16 accumulate would only cost precision. The default
// f16-accumulate gate keys on this; GOAI_CUDA_F16ACC overrides it either way.
int cu_gpu_is_geforce(void) {
    int rc = 0;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0) {
        struct cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, 0) == cudaSuccess) {
            rc = (strstr(p.name, "GeForce") != NULL || strstr(p.name, "TITAN") != NULL) ? 1 : 0;
        }
    }
    pthread_mutex_unlock(&gLock);
    return rc;
}

// ---- CUDA graph capture (§PERF: decode is launch-bound; a captured graph replays
// the whole per-token op sequence with ONE launch, eliminating per-op host cost).
// The caller must LockOSThread across begin→ops→end so ThreadLocal capture records
// every launch on gStream from the same thread (Go goroutines migrate threads).

int cu_capture_begin(void) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (cudaStreamBeginCapture((cudaStream_t)gStream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) { rc = -2; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_capture_end ends capture on gStream and instantiates an executable graph,
// returned as an opaque handle (NULL on failure). Free with cu_graph_free.
void* cu_capture_end(void) {
    cudaGraph_t graph = NULL;
    cudaGraphExec_t exec = NULL;
    pthread_mutex_lock(&gLock);
    if (cudaStreamEndCapture((cudaStream_t)gStream, &graph) != cudaSuccess) { exec = NULL; goto done; }
    if (cudaGraphInstantiate(&exec, graph, 0) != cudaSuccess) { exec = NULL; }
done:
    if (graph) cudaGraphDestroy(graph);
    pthread_mutex_unlock(&gLock);
    return (void*)exec;
}

int cu_graph_launch(void* exec) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (cudaGraphLaunch((cudaGraphExec_t)exec, (cudaStream_t)gStream) != cudaSuccess) { rc = -2; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_graph_sync(void) {
    pthread_mutex_lock(&gLock);
    int rc = (cudaStreamSynchronize((cudaStream_t)gStream) == cudaSuccess) ? 0 : -1;
    pthread_mutex_unlock(&gLock);
    return rc;
}

void cu_graph_free(void* exec) {
    if (!exec) return;
    pthread_mutex_lock(&gLock);
    cudaGraphExecDestroy((cudaGraphExec_t)exec);
    pthread_mutex_unlock(&gLock);
}

// ensure_devp grows a persistent device buffer (grow-only) used for the GQA
// batched-gemm pointer arrays, so the attention path reuses one allocation
// instead of a cudaMalloc+cudaFree per call (the decode hot path: 22 layers ×
// 6 arrays × N tokens). Guarded by gLock like every other bridge entry.
static int ensure_devp(void **buf, size_t *cap, size_t need) {
    if (*cap >= need) return 0;
    if (*buf) cudaFree(*buf);
    *buf = NULL;
    *cap = 0;
    if (cudaMalloc(buf, need) != cudaSuccess) return -1;
    *cap = need;
    return 0;
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
                     gOne,
                     gB, N,
                     gA, K,
                     gZero,
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

// cu_zero_f32 zeroes n floats on the stream (fixed-size KV cache init so masked
// 0·V terms are 0·0, never 0·NaN from uninitialized memory).
int cu_zero_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemsetAsync(d, 0, (size_t)n * sizeof(float), gStream) != cudaSuccess) { rc = -3; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_alloc_u16 allocates n u16 elements (f16 KV-cache storage, half of f32's bytes);
// cu_zero_u16 zeroes them (f16 zero == all-zero bytes). Freed with cu_free_f32
// (both are raw stream-ordered device allocations).
void* cu_alloc_u16(int n) {
    void* d = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0) {
        if (cudaMallocAsync(&d, (size_t)n * sizeof(unsigned short), gStream) != cudaSuccess) d = NULL;
    }
    pthread_mutex_unlock(&gLock);
    return d;
}

int cu_zero_u16(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemsetAsync(d, 0, (size_t)n * sizeof(unsigned short), gStream) != cudaSuccess) { rc = -3; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
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
                     gOne,
                     (const float*)dB, N,
                     gA, K,
                     gZero,
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

// cu_upload_i8 copies n signed bytes to a fresh device buffer (Q8 quantized
// weights). Caller frees via cu_free_f32 (cudaFreeAsync is dtype-agnostic).
void* cu_upload_i8(const signed char* src, int n) {
    void* d = NULL;
    size_t sz = (size_t)n;
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

// cu_qmatmul_q8: out[M,N] = a[M,K] · dequant(W), where W is stored TRANSPOSED and
// Q8_0-quantized — q[n,k] (int8, row-contiguous over K) with a per-32-block scale
// scales[n, k/32], so W[k,n] ≈ scales[n,k/32] · q[n,k]. Reading int8 weights is 4×
// less bandwidth than f32 — the decode (M=1, memory-bound GEMV) win. One thread
// per output element loops its K contraction over blocks. §PERF quantization arc.
static int q8_gemv_launch(const void* dA, const void* dQ, const void* dScales, void* dOut,
                          int M, int K, int N, int nb, float beta, const void* dGate) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv && compile_kernel(
                       // One WARP per output element: the 32 lanes split the K
                       // contraction. Two paths, chosen by K%128:
                       //  - VECTORISED (K%128==0): each lane loads an int32 = 4
                       //    packed int8 (one 16-byte-coalesced 128B warp transaction
                       //    per step vs 32B scalar), and a float4 activation, so a
                       //    step covers 128 k. 4× fewer, 4× wider weight loads →
                       //    far better DRAM bandwidth utilisation (the decode is
                       //    weight-bandwidth-bound; §PERF-SCALEBENCH). The 128-k
                       //    window spans 4 of the per-32 scale blocks, so lane l
                       //    (owning k=l*4..l*4+3, all inside one block) uses scale
                       //    sr[base + l/8].
                       //  - SCALAR (else): lane l → k=b*32+l, coalesced int8.
                       "extern \"C\" __global__ void qmatmul_q8(const float* a, const signed char* q, const float* scales, float* out, int M, int K, int N, int nb, float beta, const float* gate){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  long total = (long)M*N;\n"
                       "  if (warp >= total) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  const signed char* qr = q + (size_t)n*K;\n"
                       "  const float* sr = scales + (size_t)n*nb;\n"
                       "  float acc = 0.0f;\n"
                       "  if ((K & 511) == 0){\n"          // int4 (16B/lane = 512 k/step): more memory requests in flight, same warp count
                       "    const int4* qr4 = (const int4*)qr;\n"
                       "    int steps = K >> 9;\n"
                       "    for (int w = 0; w < steps; w++){\n"
                       "      int4 pk = qr4[w*32 + lane];\n"
                       "      int kb = w*512 + lane*16;\n"
                       "      float s = sr[w*16 + (lane >> 1)];\n"   // lane's 16 k lie in one per-32 block
                       "      int P[4]; P[0]=pk.x; P[1]=pk.y; P[2]=pk.z; P[3]=pk.w;\n"
                       "      #pragma unroll\n"
                       "      for (int j = 0; j < 4; j++){\n"
                       "        float4 av = *(const float4*)(ar + kb + j*4);\n"
                       "        int pj = P[j];\n"
                       "        acc += s*(av.x*(float)(signed char)(pj&0xff) + av.y*(float)(signed char)((pj>>8)&0xff) + av.z*(float)(signed char)((pj>>16)&0xff) + av.w*(float)(signed char)((pj>>24)&0xff));\n"
                       "      }\n"
                       "    }\n"
                       "  } else if ((K & 127) == 0){\n"
                       "    const int* qr32 = (const int*)qr;\n"
                       "    int steps = K >> 7;\n"          // K/128 windows of 128 k
                       "    for (int w = 0; w < steps; w++){\n"
                       "      int packed = qr32[w*32 + lane];\n"      // 4 int8 for this lane
                       "      int k = w*128 + lane*4;\n"
                       "      float4 av = *(const float4*)(ar + k);\n"
                       "      float s = sr[w*4 + (lane >> 3)];\n"     // per-32 block scale
                       "      float q0 = (float)(signed char)(packed & 0xff);\n"
                       "      float q1 = (float)(signed char)((packed >> 8) & 0xff);\n"
                       "      float q2 = (float)(signed char)((packed >> 16) & 0xff);\n"
                       "      float q3 = (float)(signed char)((packed >> 24) & 0xff);\n"
                       "      acc += s*(av.x*q0 + av.y*q1 + av.z*q2 + av.w*q3);\n"
                       "    }\n"
                       "  } else {\n"
                       "    for (int b = 0; b < nb; b++){\n"
                       "      int k = b*32 + lane;\n"
                       "      if (k < K){ acc += sr[b]*ar[k]*(float)qr[k]; }\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){\n"
                       "    float r = acc;\n"
                       "    if (gate){ float g = gate[warp]; float sg = g>=0.0f ? 1.0f/(1.0f+expf(-g)) : expf(g)/(1.0f+expf(g)); r = g*sg*acc; }\n" // SwiGLU epilogue, arithmetic == the standalone swiglu kernel (token parity)
                       "    out[warp] = beta*out[warp] + r;\n"                                        // beta=1 fuses the residual add
                       "  }\n"
                       "}\n",
                       "qmatmul_q8.cu", "qmatmul_q8", &gQgemv) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32; // one warp (32 threads) per output element
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[10];
        args[0] = &dA; args[1] = &dQ; args[2] = &dScales; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &nb; args[8] = &beta; args[9] = &dGate;
        rc = (cuLaunchKernel(gQgemv, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_qmatmul_q8(const void* dA, const void* dQ, const void* dScales, void* dOut,
                  int M, int K, int N, int nb, float beta) {
    return q8_gemv_launch(dA, dQ, dScales, dOut, M, K, N, nb, beta, NULL);
}

// cu_qmatmul_q8_swiglu: out = silu(gate) ⊙ (a·dequant(W)) — the up-projection GEMV with
// the SwiGLU applied in the epilogue (Tw55 fusion stack): kills the separate SwiGLU
// launch and a full hidden-vector round-trip. gate has out's [M,N] layout.
int cu_qmatmul_q8_swiglu(const void* dA, const void* dQ, const void* dScales, const void* dGate, void* dOut,
                         int M, int K, int N, int nb) {
    return q8_gemv_launch(dA, dQ, dScales, dOut, M, K, N, nb, 0.0f, dGate);
}

// cu_qmatmul_q4: out[M,N] = a[M,K]·dequant(W4), W4 = ASYMMETRIC Q4 stored TRANSPOSED —
// q[N,K/2] packed 4-bit nibbles + per-32-block f32 scale + f32 min, dequant
// w = min_b + nibble·scale_b (nibble∈[0,15]). Asymmetric (scale+min) is far more accurate
// than symmetric Q4_0. Reads ≈0.67× the bytes of the Q8 GEMV (0.75 vs 1.125 B/weight) —
// the weight-bandwidth lever for decode, where the GEMV is bandwidth-bound at ≈peak. One
// warp per output; lane l loads an int32 (8 nibbles) and 8 activations, 256 k/step (K%256==0).
int cu_qmatmul_q4(const void* dA, const void* dQ, const void* dScales, const void* dMins, void* dOut,
                  int M, int K, int N, int nb, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv4 && compile_kernel(
                       "extern \"C\" __global__ void qmatmul_q4(const float* a, const unsigned char* q, const float* scales, const float* mins, float* out, int M, int K, int N, int nb, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  const int* qr = (const int*)(q + (size_t)n*(K/2));\n"  // K/2 packed bytes/row
                       "  const float* sr = scales + (size_t)n*nb;\n"
                       "  const float* mr = mins + (size_t)n*nb;\n"
                       "  float acc = 0.0f;\n"
                       "  int steps = K >> 8;\n"      // 256 k / step = 8 per-32 blocks
                       "  for (int w = 0; w < steps; w++){\n"
                       "    int p = qr[w*32 + lane];\n"          // 8 nibbles = 8 weights
                       "    int k = w*256 + lane*8;\n"
                       "    int blk = w*8 + (lane >> 2);\n"      // this lane's 8 k lie in one block
                       "    float s = sr[blk], mn = mr[blk];\n"
                       "    float4 a0 = *(const float4*)(ar + k);\n"
                       "    float4 a1 = *(const float4*)(ar + k + 4);\n"
                       "    acc += a0.x*(mn + (float)(p&0xf)*s) + a0.y*(mn + (float)((p>>4)&0xf)*s) + a0.z*(mn + (float)((p>>8)&0xf)*s) + a0.w*(mn + (float)((p>>12)&0xf)*s);\n"
                       "    acc += a1.x*(mn + (float)((p>>16)&0xf)*s) + a1.y*(mn + (float)((p>>20)&0xf)*s) + a1.z*(mn + (float)((p>>24)&0xf)*s) + a1.w*(mn + (float)((p>>28)&0xf)*s);\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q4.cu", "qmatmul_q4", &gQgemv4) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[10];
        args[0] = &dA; args[1] = &dQ; args[2] = &dScales; args[3] = &dMins; args[4] = &dOut;
        args[5] = &M; args[6] = &K; args[7] = &N; args[8] = &nb; args[9] = &beta;
        rc = (cuLaunchKernel(gQgemv4, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_q4k: out[M,N] = a[M,K]·dequant(W), W = ggml Q4_K (§R100) stored per OUTPUT row:
// row n = K/256 super-blocks of 144 bytes (f16 d, f16 dmin, scales[12] = 8 six-bit scales +
// 8 six-bit mins bit-packed, qs[128] nibbles; per 64-elem pair the low nibbles are sub-block
// 2p, the high nibbles 2p+1). Dequant y = d*sc6*q - dmin*min6. SAME warp-per-output GEMV shape
// as cu_qmatmul_q4; 0.5625 B/weight = 25% fewer bytes than the asymmetric Q4 (0.75) at much
// higher accuracy (super-block-scaled 6-bit sub-scales vs one f32 scale+min per 32). K%256==0.
static int q4k_gemv_launch(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta, const void* dGate) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv4k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n" // *2^112 exponent rebias; exact for normals AND subnormals
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "__device__ __forceinline__ void get_sm(int j, const unsigned char* q, float* sc, float* mn){\n"
                       "  if (j<4){ *sc=(float)(q[j]&63); *mn=(float)(q[j+4]&63); }\n"
                       "  else { *sc=(float)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); *mn=(float)((q[j+4]>>4)|((q[j]>>6)<<4)); }\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q4k(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta, const float* gate){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*144;\n"
                       "  float acc = 0.0f;\n"
                       "  int p = lane >> 3, i0 = (lane & 7) * 4;\n"      // lane's pair-chunk + byte offset
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*144;\n"
                       "    unsigned int qw = *(const unsigned int*)(blk + 16 + lane*4);\n" // 4 qs bytes = 8 elems
                       "    float d = f16f(*(const unsigned short*)blk);\n"      // uniform across the warp (broadcast load)
                       "    float dmin = f16f(*(const unsigned short*)(blk+2));\n"
                       "    float sc, mn; get_sm(lane & 7, blk+4, &sc, &mn);\n"      // branch-free: every lane decodes one of the 8 pairs
                       "    float c1 = d*sc, c2 = dmin*mn;\n"
                       "    int kb = w*256 + p*64 + i0;\n"
                       "    float4 al = *(const float4*)(ar + kb);\n"      // low-nibble elems
                       "    float4 ah = *(const float4*)(ar + kb + 32);\n" // high-nibble elems (+32 in the pair)
                       "    float sql = al.x*(float)(qw&0xFu) + al.y*(float)((qw>>8)&0xFu) + al.z*(float)((qw>>16)&0xFu) + al.w*(float)((qw>>24)&0xFu);\n"
                       "    float sqh = ah.x*(float)((qw>>4)&0xFu) + ah.y*(float)((qw>>12)&0xFu) + ah.z*(float)((qw>>20)&0xFu) + ah.w*(float)((qw>>28)&0xFu);\n"
                       "    float sal = (al.x+al.y)+(al.z+al.w), sah = (ah.x+ah.y)+(ah.z+ah.w);\n"
                       "    float dl = __shfl_sync(0xffffffff, c1, 2*p),   o1 = __shfl_sync(0xffffffff, c2, 2*p);\n"
                       "    float dh = __shfl_sync(0xffffffff, c1, 2*p+1), o2 = __shfl_sync(0xffffffff, c2, 2*p+1);\n"
                       "    acc += dl*sql - o1*sal + dh*sqh - o2*sah;\n"   // y=d*sc*q-dmin*m folded per SUB-BLOCK, not per element
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){\n"
                       "    float r = acc;\n"
                       "    if (gate){ float g = gate[warp]; float sg = g>=0.0f ? 1.0f/(1.0f+expf(-g)) : expf(g)/(1.0f+expf(g)); r = g*sg*acc; }\n" // SwiGLU epilogue, arithmetic == the standalone swiglu kernel
                       "    out[warp] = beta*out[warp] + r;\n"
                       "  }\n"
                       "}\n",
                       "qmatmul_q4k.cu", "qmatmul_q4k", &gQgemv4k) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta; args[7] = &dGate;
        rc = (cuLaunchKernel(gQgemv4k, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_qmatmul_q4k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    return q4k_gemv_launch(dA, dQ, dOut, M, K, N, beta, NULL);
}

// cu_qmatmul_q4k_swiglu: out = silu(gate) ⊙ (a·dequant(W)) — see cu_qmatmul_q8_swiglu.
int cu_qmatmul_q4k_swiglu(const void* dA, const void* dQ, const void* dGate, void* dOut,
                          int M, int K, int N) {
    return q4k_gemv_launch(dA, dQ, dOut, M, K, N, 0.0f, dGate);
}

// cu_qmatmul_q5k: out[M,N] = a[M,K]·dequant(W), W = ggml Q5_K (§R102) stored per OUTPUT row:
// K/256 super-blocks of 176 bytes (f16 d, f16 dmin, scales[12] = the SAME get_scale_min_k4
// 6-bit scale+min packing as Q4_K, qh[32] one high bit per quant, qs[128] low nibbles).
// Dequant y = d·sc6·q5 − dmin·min6 with q5 = nibble | (highbit<<4). Same warp-per-output GEMV
// as cu_qmatmul_q4k — identical scale/min decode and per-sub-block min folding — the only
// change is the 5th bit: each lane reads its 4 qh bytes and, per pair p, uses qh bit 2p for the
// low nibble and 2p+1 for the high nibble. 0.6875 B/weight (Q4_K 0.5625, Q8 1.0). K%256==0.
int cu_qmatmul_q5k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv5k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "__device__ __forceinline__ void get_sm(int j, const unsigned char* q, float* sc, float* mn){\n"
                       "  if (j<4){ *sc=(float)(q[j]&63); *mn=(float)(q[j+4]&63); }\n"
                       "  else { *sc=(float)((q[j+4]&0xF)|((q[j-4]>>6)<<4)); *mn=(float)((q[j+4]>>4)|((q[j]>>6)<<4)); }\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q5k(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*176;\n"
                       "  float acc = 0.0f;\n"
                       "  int p = lane >> 3, i0 = (lane & 7) * 4;\n"
                       "  int slo = 2*p, shi = 2*p + 1;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*176;\n"
                       "    unsigned int qw  = *(const unsigned int*)(blk + 48 + lane*4);\n" // 4 qs bytes = 8 low/high nibbles
                       "    unsigned int qhw = *(const unsigned int*)(blk + 16 + i0);\n"     // 4 qh bytes for elems i0..i0+3
                       "    float d = f16f(*(const unsigned short*)blk);\n"
                       "    float dmin = f16f(*(const unsigned short*)(blk+2));\n"
                       "    float sc, mn; get_sm(lane & 7, blk+4, &sc, &mn);\n"
                       "    float c1 = d*sc, c2 = dmin*mn;\n"
                       "    int kb = w*256 + p*64 + i0;\n"
                       "    float4 al = *(const float4*)(ar + kb);\n"
                       "    float4 ah = *(const float4*)(ar + kb + 32);\n"
                       "    unsigned qb0=qhw&0xFFu, qb1=(qhw>>8)&0xFFu, qb2=(qhw>>16)&0xFFu, qb3=(qhw>>24)&0xFFu;\n"
                       "    float lo0=(float)((qw&0xFu)      |(((qb0>>slo)&1u)<<4));\n"
                       "    float lo1=(float)(((qw>>8)&0xFu) |(((qb1>>slo)&1u)<<4));\n"
                       "    float lo2=(float)(((qw>>16)&0xFu)|(((qb2>>slo)&1u)<<4));\n"
                       "    float lo3=(float)(((qw>>24)&0xFu)|(((qb3>>slo)&1u)<<4));\n"
                       "    float hi0=(float)(((qw>>4)&0xFu) |(((qb0>>shi)&1u)<<4));\n"
                       "    float hi1=(float)(((qw>>12)&0xFu)|(((qb1>>shi)&1u)<<4));\n"
                       "    float hi2=(float)(((qw>>20)&0xFu)|(((qb2>>shi)&1u)<<4));\n"
                       "    float hi3=(float)(((qw>>28)&0xFu)|(((qb3>>shi)&1u)<<4));\n"
                       "    float sql = al.x*lo0 + al.y*lo1 + al.z*lo2 + al.w*lo3;\n"
                       "    float sqh = ah.x*hi0 + ah.y*hi1 + ah.z*hi2 + ah.w*hi3;\n"
                       "    float sal = (al.x+al.y)+(al.z+al.w), sah = (ah.x+ah.y)+(ah.z+ah.w);\n"
                       "    float dl = __shfl_sync(0xffffffff, c1, 2*p),   o1 = __shfl_sync(0xffffffff, c2, 2*p);\n"
                       "    float dh = __shfl_sync(0xffffffff, c1, 2*p+1), o2 = __shfl_sync(0xffffffff, c2, 2*p+1);\n"
                       "    acc += dl*sql - o1*sal + dh*sqh - o2*sah;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q5k.cu", "qmatmul_q5k", &gQgemv5k) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemv5k, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_q3k: out[M,N] = a[M,K]·dequant(W), W = ggml Q3_K (§R103) stored per OUTPUT row:
// K/256 super-blocks of 110 bytes (hmask[32] one high bit/quant, qs[64] two low bits/quant,
// scales[12] = 16 six-bit scales via the aux/kmask splice, then f16 d LAST at offset 108).
// SYMMETRIC (no min): y = d·(sc6−32)·(q3−4), q3 = 2 low bits | (high bit)<<2, computed as
// d·(sc6−32)·(lowbits − (highbitSet ? 0 : 4)) (the inverted-hmask arithmetic). Warp-per-output
// GEMV: lane owns 8 contiguous elements = half of one 16-element sub-block (is=lane>>1), so one
// signed scale per lane. 0.4297 B/weight — the bulk tensors of Q3_K_M/_L/_S (fit big models in
// limited VRAM). K%256==0.
int cu_qmatmul_q3k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv3k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q3k(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*110;\n"
                       "  int is = lane >> 1, half = lane & 1, l0 = half*8;\n"     // sub-block, half, element offset
                       "  int nb = is >> 3, jj = (is & 7) >> 1, gsel = is & 1, g = gsel*16;\n"
                       "  int shift = 2*jj, hbit = nb*4 + jj;\n"
                       "  int auxIdx = is >> 2, byteIdx = is & 3;\n"
                       "  int yi = nb*128 + jj*32 + g + l0;\n"                      // K-offset within a super-block
                       "  int qsoff = 32 + nb*32 + g + l0, hmoff = g + l0;\n"
                       "  const unsigned km1 = 0x03030303u, km2 = 0x0f0f0f0fu;\n"
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*110;\n"
                       "    float d = f16f((unsigned short)(blk[108] | (blk[109]<<8)));\n" // 110-byte block: blk not 4-aligned, read bytes
                       "    const unsigned char* sp = blk + 96;\n"
                       "    unsigned a0 = sp[0]|(sp[1]<<8)|(sp[2]<<16)|(sp[3]<<24);\n"
                       "    unsigned a1 = sp[4]|(sp[5]<<8)|(sp[6]<<16)|(sp[7]<<24);\n"
                       "    unsigned a2v = sp[8]|(sp[9]<<8)|(sp[10]<<16)|(sp[11]<<24), tmp = a2v;\n"
                       "    unsigned A0 = (a0 & km2) | (((tmp>>0)&km1)<<4);\n"
                       "    unsigned A1 = (a1 & km2) | (((tmp>>2)&km1)<<4);\n"
                       "    unsigned A2 = ((a0>>4)&km2) | (((tmp>>4)&km1)<<4);\n"
                       "    unsigned A3 = ((a1>>4)&km2) | (((tmp>>6)&km1)<<4);\n"
                       "    unsigned aux = auxIdx==0?A0 : auxIdx==1?A1 : auxIdx==2?A2 : A3;\n"
                       "    int sc = (int)((aux >> (byteIdx*8)) & 0x3Fu);\n"
                       "    float dl = d * (float)(sc - 32);\n"
                       "    const float* av = ar + (size_t)w*256 + yi;\n"
                       "    float4 alo = *(const float4*)av;\n"
                       "    float4 ahi = *(const float4*)(av + 4);\n"
                       "    const unsigned char* qs = blk + qsoff;\n"
                       "    const unsigned char* hm = blk + hmoff;\n"
                       "    float av8[8] = { alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int t = 0; t < 8; t++){\n"
                       "      int lowb = (qs[t] >> shift) & 3;\n"
                       "      int hset = (hm[t] >> hbit) & 1;\n"
                       "      s += av8[t] * (float)(lowb - (hset ? 0 : 4));\n"
                       "    }\n"
                       "    acc += dl * s;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q3k.cu", "qmatmul_q3k", &gQgemv3k) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemv3k, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_q2k: out[M,N] = a[M,K]·dequant(W), W = ggml Q2_K (§R104) stored per OUTPUT row:
// K/256 super-blocks of 84 bytes (scales[16]: low nibble = 4-bit scale, high nibble = 4-bit
// min; qs[64] two-bit quants; f16 d at 80, f16 dmin at 82). ASYMMETRIC AFFINE like Q4_K but
// coarser: y = d·sc4·q2 − dmin·min4, q2 = (qs>>2j)&3. Same element order + lane mapping as the
// Q3_K GEMV (is=lane>>1 owns 8 contiguous elems = half a 16-sub-block), minus the hmask/splice:
// per lane acc += dl·Σaᵢqᵢ − ml·Σaᵢ. 0.3281 B/weight — the smallest quant (fit 70B+ in tight
// VRAM). K%256==0.
int cu_qmatmul_q2k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv2k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q2k(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*84;\n"
                       "  int is = lane >> 1, half = lane & 1, l0 = half*8;\n"
                       "  int nb = is >> 3, jj = (is & 7) >> 1, gsel = is & 1, g = gsel*16;\n"
                       "  int shift = 2*jj;\n"
                       "  int yi = nb*128 + jj*32 + g + l0;\n"
                       "  int qsoff = 16 + nb*32 + g + l0;\n"
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*84;\n"
                       "    float d = f16f((unsigned short)(blk[80] | (blk[81]<<8)));\n"     // 84B block is 4-aligned but read bytes uniformly
                       "    float dmin = f16f((unsigned short)(blk[82] | (blk[83]<<8)));\n"
                       "    unsigned sc = blk[is];\n"
                       "    float dl = d * (float)(sc & 0xF), ml = dmin * (float)(sc >> 4);\n"
                       "    const float* av = ar + (size_t)w*256 + yi;\n"
                       "    float4 alo = *(const float4*)av;\n"
                       "    float4 ahi = *(const float4*)(av + 4);\n"
                       "    const unsigned char* qs = blk + qsoff;\n"
                       "    float av8[8] = { alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w };\n"
                       "    float sq = 0.0f, sa = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int t = 0; t < 8; t++){\n"
                       "      float q2 = (float)((qs[t] >> shift) & 3);\n"
                       "      sq += av8[t] * q2; sa += av8[t];\n"
                       "    }\n"
                       "    acc += dl*sq - ml*sa;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q2k.cu", "qmatmul_q2k", &gQgemv2k) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemv2k, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_q40: out[M,N] = a[M,K]·dequant(W), W = ggml Q4_0 (SYMMETRIC round, y = d·(nibble−8)),
// REPACKED at upload into two 4/16-aligned regions per output row — dScale (nblk f16 block scales,
// contiguous) and dNib (nblk×16 nibble bytes, block-major, 16-aligned). The native 18-byte gguf
// block is not 4-aligned, forcing byte reads and ~8× the memory transactions of a Q4_K super-block
// at IDENTICAL bytes (transactions-not-bytes, §Tw54). The repack lets a warp process 8 blocks per
// iteration with 4 lanes/block doing COALESCED uint nibble reads (8 iters for K=2048, matching
// Q4_K) — ~2× the naive per-block kernel. K%32==0.
int cu_qmatmul_q40(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv40 && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q40(const float* a, const unsigned char* sc, const unsigned char* nb, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int nblk = K >> 5;\n"
                       "  int g = lane >> 2, sub = lane & 3;\n"      // block-in-chunk (0..7), uint-in-block (0..3)
                       "  float acc = 0.0f;\n"
                       "  for (int bb = 0; bb < nblk; bb += 8){\n"
                       "    int b = bb + g;\n"
                       "    if (b < nblk){\n"
                       "      const unsigned char* sp = sc + (size_t)(n*nblk + b)*2;\n"
                       "      float d = f16f((unsigned short)(sp[0] | (sp[1]<<8)));\n"
                       "      unsigned qw = *(const unsigned int*)(nb + (size_t)(n*nblk + b)*16 + sub*4);\n" // 4 nibble bytes, coalesced
                       "      const float* arb = ar + (size_t)b*32 + sub*4;\n"
                       "      float4 al = *(const float4*)arb;\n"        // 4 low-nibble activations, coalesced float4
                       "      float4 ah = *(const float4*)(arb + 16);\n" // 4 high-nibble activations
                       "      unsigned b0=qw&0xFFu, b1=(qw>>8)&0xFFu, b2=(qw>>16)&0xFFu, b3=(qw>>24)&0xFFu;\n"
                       "      float sl = al.x*(float)((int)(b0&0xF)-8) + al.y*(float)((int)(b1&0xF)-8) + al.z*(float)((int)(b2&0xF)-8) + al.w*(float)((int)(b3&0xF)-8);\n"
                       "      float sh = ah.x*(float)((int)(b0>>4)-8)  + ah.y*(float)((int)(b1>>4)-8)  + ah.z*(float)((int)(b2>>4)-8)  + ah.w*(float)((int)(b3>>4)-8);\n"
                       "      acc += d * (sl + sh);\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q40.cu", "qmatmul_q40", &gQgemv40) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dScale; args[2] = &dNib; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemv40, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// IQ4 nonlinear codebook (kvalues_iq4nl) — shared by IQ4_NL and IQ4_XS; nibble → value.
#define IQ4_KVALS "  const float kv[16] = {-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};\n"

// cu_qmatmul_iq4nl: out[M,N] = a[M,K]·dequant(W), W = ggml IQ4_NL stored per OUTPUT row:
// K/32 blocks of 18 bytes (f16 d + 16 nibble bytes). Like Q4_0 (byte i → element i low nibble,
// i+16 high) but the nibble indexes a NONLINEAR 16-value codebook instead of (nibble−8):
// y = d·kvals[nibble]. Warp-per-output GEMV, lane l owns element l of each 32-block. 18-byte
// blocks not 4-aligned → d byte-read. 0.5625 B/weight. K%32==0.
int cu_qmatmul_iq4nl(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI4nl && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq4nl(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int nblk = K >> 5;\n"
                       "  const unsigned char* qr = q + (size_t)n*nblk*18;\n"
                       IQ4_KVALS
                       "  int bi = lane & 15, hi = lane >> 4;\n"
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 4\n"
                       "  for (int b = 0; b < nblk; b++){\n"
                       "    const unsigned char* blk = qr + (size_t)b*18;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    unsigned qb = blk[2 + bi];\n"
                       "    int nib = hi ? (qb >> 4) : (qb & 0xF);\n"
                       "    acc += ar[b*32 + lane] * d * kv[nib];\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq4nl.cu", "qmatmul_iq4nl", &gQgemvI4nl) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemvI4nl, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_iq4xs: out[M,N] = a[M,K]·dequant(W), W = ggml IQ4_XS stored per OUTPUT row:
// K/256 super-blocks of 136 bytes (f16 d, u16 scales_h, 4 scales_l bytes, 128 qs). 8 sub-blocks
// of 32; sub-block sb scale = int8((scales_l[sb/2]>>(4·(sb&1))&0xF) | ((scales_h>>(2·sb))&3)<<4)
// − 32, effective d·scale; nibbles index the same IQ4 codebook. Warp-per-output GEMV: lane l
// owns element l of each 32-elem sub-block across the 8 sub-blocks. 136-byte blocks 4-aligned
// but d/scales byte-read uniformly. 0.5 B/weight (super-block sub-scales). K%256==0.
int cu_qmatmul_iq4xs(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI4xs && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq4xs(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*136;\n"
                       IQ4_KVALS
                       "  int bi = lane & 15, hi = lane >> 4;\n"
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*136;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    unsigned sch = (unsigned)(blk[2] | (blk[3]<<8));\n"     // scales_h u16
                       "    const unsigned char* scl = blk + 4;\n"                 // scales_l[4]
                       "    const unsigned char* qs = blk + 8;\n"
                       "    #pragma unroll\n"
                       "    for (int sb = 0; sb < 8; sb++){\n"
                       "      int sl = (scl[sb>>1] >> (4*(sb&1))) & 0xF;\n"
                       "      int sh = (sch >> (2*sb)) & 3;\n"
                       "      int s6 = (sl | (sh<<4)) - 32;\n"                      // signed 6-bit sub-scale
                       "      float ds = d * (float)s6;\n"
                       "      unsigned qb = qs[sb*16 + bi];\n"
                       "      int nib = hi ? (qb >> 4) : (qb & 0xF);\n"
                       "      acc += ar[w*256 + sb*32 + lane] * ds * kv[nib];\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq4xs.cu", "qmatmul_iq4xs", &gQgemvI4xs) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemvI4xs, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_mxfp4: out[M,N] = a[M,K]·dequant(MXFP4, the OCP Microscaling FP4 that gpt-oss ships
// in). REPACKED at upload like Q4_0 into dScale (nblk E8M0 scale bytes/row) + dNib (nblk×16 nibble
// bytes/row, 16-aligned) so a warp processes 8 blocks/iteration, 4 lanes/block, with coalesced uint
// nibble reads + float4 activations. Nibble indexes the FP4 E2M1 codebook: y = e8m0(scale)·kv[nibble].
// K%32==0.
int cu_qmatmul_mxfp4(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvMxfp4 && compile_kernel(
                       "__device__ const float mxfp4kv[16] = {0.f,1.f,2.f,3.f,4.f,6.f,8.f,12.f,0.f,-1.f,-2.f,-3.f,-4.f,-6.f,-8.f,-12.f};\n"
                       "extern \"C\" __global__ void qmatmul_mxfp4(const float* a, const unsigned char* sc, const unsigned char* nb, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int nblk = K >> 5;\n"
                       "  int g = lane >> 2, sub = lane & 3;\n"
                       "  float acc = 0.0f;\n"
                       "  for (int bb = 0; bb < nblk; bb += 8){\n"
                       "    int b = bb + g;\n"
                       "    if (b < nblk){\n"
                       "      unsigned e = sc[(size_t)(n*nblk + b)];\n"                 // E8M0 scale byte
                       "      unsigned bits = (e < 2) ? (0x00200000u << e) : ((e-1) << 23);\n"
                       "      float d = __uint_as_float(bits);\n"
                       "      unsigned qw = *(const unsigned int*)(nb + (size_t)(n*nblk + b)*16 + sub*4);\n"
                       "      const float* arb = ar + (size_t)b*32 + sub*4;\n"
                       "      float4 al = *(const float4*)arb;\n"
                       "      float4 ah = *(const float4*)(arb + 16);\n"
                       "      unsigned b0=qw&0xFFu, b1=(qw>>8)&0xFFu, b2=(qw>>16)&0xFFu, b3=(qw>>24)&0xFFu;\n"
                       "      float sl = al.x*mxfp4kv[b0&0xF] + al.y*mxfp4kv[b1&0xF] + al.z*mxfp4kv[b2&0xF] + al.w*mxfp4kv[b3&0xF];\n"
                       "      float sh = ah.x*mxfp4kv[b0>>4] + ah.y*mxfp4kv[b1>>4] + ah.z*mxfp4kv[b2>>4] + ah.w*mxfp4kv[b3>>4];\n"
                       "      acc += d * (sl + sh);\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_mxfp4.cu", "qmatmul_mxfp4", &gQgemvMxfp4) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dScale; args[2] = &dNib; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvMxfp4, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_q6k: out[M,N] = a[M,K]·dequant(W), W = ggml Q6_K (§R99) stored per OUTPUT
// row: K/256 super-blocks of 210 bytes (ql[128] low nibbles, qh[64] 2-bit highs 4/byte,
// scales[16] int8 per 16-elem sub-block, f16 d at 208). Dequant y = d·sc·(q6−32), q6 =
// ql-nibble | qh-2bits<<4, element quadrants l, l+32, l+64, l+96 per group of 128.
// Warp-per-output GEMV like cu_qmatmul_q4k: lane g=lane>>4 picks the 128-group, each
// lane covers two l positions × 4 quadrants = 8 elements/block. 0.8203 B/weight — the
// native path for Q4_K_M's Q6_K minority (v/down on some layers + output head), vs
// re-encoding them to Q8 (1.0625 B/w). K%256==0.
int cu_qmatmul_q6k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv6k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q6k(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*210;\n"
                       "  float acc = 0.0f;\n"
                       "  int g = lane >> 4, i = (lane & 15) * 2;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*210;\n"
                       "    const unsigned char* ql = blk + g*64;\n"
                       "    const unsigned char* qh = blk + 128 + g*32;\n"
                       "    const signed char* sc = (const signed char*)(blk + 192) + g*8;\n"
                       "    float d = f16f(*(const unsigned short*)(blk + 208));\n"
                       "    const float* ay = ar + w*256 + g*128;\n"
                       "    int is = i >> 4;\n"                                        // i and i+1 share the sub-block
                       "    float s0 = d*(float)sc[is],   s2 = d*(float)sc[is+2];\n"
                       "    float s4 = d*(float)sc[is+4], s6 = d*(float)sc[is+6];\n"
                       "    unsigned int b1 = *(const unsigned short*)(ql + i);\n"      // {ql[i], ql[i+1]} in one load
                       "    unsigned int b2 = *(const unsigned short*)(ql + i + 32);\n"
                       "    unsigned int hh = *(const unsigned short*)(qh + i);\n"
                       "    float2 a1v = *(const float2*)(ay + i),      a2v = *(const float2*)(ay + i + 32);\n"
                       "    float2 a3v = *(const float2*)(ay + i + 64), a4v = *(const float2*)(ay + i + 96);\n"
                       "    #pragma unroll 2\n"
                       "    for (int t = 0; t < 2; t++){\n"
                       "      unsigned int lo = (b1 >> (8*t)) & 0xffu, hi = (b2 >> (8*t)) & 0xffu, h = (hh >> (8*t)) & 0xffu;\n"
                       "      float q1 = (float)((int)((lo & 15u) | ((h & 3u) << 4)) - 32);\n"
                       "      float q2 = (float)((int)((hi & 15u) | (((h >> 2) & 3u) << 4)) - 32);\n"
                       "      float q3 = (float)((int)((lo >> 4) | (((h >> 4) & 3u) << 4)) - 32);\n"
                       "      float q4 = (float)((int)((hi >> 4) | (((h >> 6) & 3u) << 4)) - 32);\n"
                       "      float w1 = t ? a1v.y : a1v.x, w2 = t ? a2v.y : a2v.x;\n"
                       "      float w3 = t ? a3v.y : a3v.x, w4 = t ? a4v.y : a4v.x;\n"
                       "      acc += s0*q1*w1 + s2*q2*w2 + s4*q3*w3 + s6*q4*w4;\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q6k.cu", "qmatmul_q6k", &gQgemv6k) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemv6k, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// ---- f16 tensor-core prefill path (lever b1): weights resident as f16, activations
// converted per call, cublasGemmEx with f32 accumulation. Prefill is compute-bound at
// the cuBLAS f32 ceiling (PERF-PREFILL-PROFILE: ffn-gemm 54%); Ampere runs f16 tensor
// cores at ~2x the f32 FMA rate, which is how llama.cpp's prefill pulls ahead.

static const char* kCvtF16Src =
    "extern \"C\" __global__ void cvt_f32_f16(const float* src, unsigned short* dst, long n){\n"
    "  long i = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
    "  if (i >= n) return;\n"
    "  float v = src[i];\n"
    "  dst[i] = __float2half_rn_bits(v);\n"
    "}\n";

// __float2half intrinsics need cuda_fp16.h under nvrtc; a bit-exact manual RN
// conversion avoids the header (round-to-nearest-even, handles subnormals/inf).
static const char* kCvtF16ManualSrc =
    "extern \"C\" __global__ void cvt_f32_f16(const float* src, unsigned short* dst, long n){\n"
    "  long i = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
    "  if (i >= n) return;\n"
    "  unsigned int x = __float_as_uint(src[i]);\n"
    "  unsigned int sign = (x >> 16) & 0x8000u;\n"
    "  unsigned int em = x & 0x7fffffffu;\n"
    "  unsigned short h;\n"
    "  if (em >= 0x47800000u) { h = (unsigned short)(sign | (em > 0x7f800000u ? 0x7e00u : 0x7c00u)); }\n" // inf/nan/overflow
    "  else if (em < 0x38800000u) {\n"                                                                    // subnormal/zero
    "    unsigned int m = (em & 0x007fffffu) | 0x00800000u;\n"
    "    int shift = 126 - (int)(em >> 23);\n"
    "    unsigned int v = (shift < 32) ? (m >> shift) : 0u;\n"
    "    unsigned int r = (shift < 33 && shift > 0) ? ((m >> (shift-1)) & 1u) : 0u;\n"
    "    unsigned int s = (shift < 32 && (m & ((1u << (shift-1)) - 1u))) ? 1u : 0u;\n"
    "    v += (r & (s | (v & 1u)));\n"
    "    h = (unsigned short)(sign | v);\n"
    "  } else {\n"
    "    unsigned int e = ((em >> 23) - 112u) << 10;\n"
    "    unsigned int m = (em >> 13) & 0x3ffu;\n"
    "    unsigned int r = (em >> 12) & 1u;\n"
    "    unsigned int s = (em & 0xfffu) ? 1u : 0u;\n"
    "    unsigned int v = (sign | e | m) + (r & (s | (m & 1u)));\n"
    "    h = (unsigned short)v;\n"
    "  }\n"
    "  dst[i] = h;\n"
    "}\n";

static int ensure_cvt_f16(void) {
    if (gCvtF16) return 0;
    (void)kCvtF16Src; // intrinsic variant kept for reference; manual RN is header-free
    return compile_kernel(kCvtF16ManualSrc, "cvt_f16.cu", "cvt_f32_f16", &gCvtF16);
}

static int launch_cvt_f16(const void* src, void* dst, long n) {
    if (ensure_cvt_f16() != 0) return -2;
    int threads = 256;
    long blocks = (n + threads - 1) / threads;
    void* args[3] = { &src, &dst, &n };
    return (cuLaunchKernel(gCvtF16, (unsigned)blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
}

void* cu_upload_f16(const float* src, long n) {
    void* d32 = NULL; void* d16 = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) goto fail;
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) goto fail;
    if (cudaMallocAsync(&d32, n * sizeof(float), gStream) != cudaSuccess) goto fail;
    if (cudaMallocAsync(&d16, n * sizeof(unsigned short), gStream) != cudaSuccess) goto fail;
    if (cudaMemcpyAsync(d32, src, n * sizeof(float), cudaMemcpyHostToDevice, gStream) != cudaSuccess) goto fail;
    if (launch_cvt_f16(d32, d16, n) != 0) goto fail;
    cudaFreeAsync(d32, gStream); d32 = NULL;
    if (cudaStreamSynchronize(gStream) != cudaSuccess) goto fail; // src is caller host memory
    pthread_mutex_unlock(&gLock);
    return d16;
fail:
    if (d32) cudaFreeAsync(d32, gStream);
    if (d16) cudaFreeAsync(d16, gStream);
    pthread_mutex_unlock(&gLock);
    return NULL;
}

int cu_matmul_f16w(const void* dA32, const void* dW16, void* dC32, int M, int K, int N, float beta) {
    int rc = -2;
    void* a16 = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (cudaMallocAsync(&a16, (size_t)M * K * sizeof(unsigned short), gStream) != cudaSuccess) { rc = -5; goto done; }
    if (launch_cvt_f16(dA32, a16, (long)M * K) != 0) { rc = -6; goto done; }
    {
        // the handle is in CUBLAS_POINTER_MODE_DEVICE (graph-capture-safe): alpha/beta
        // MUST be the resident gOne/gZero constants, never host stack pointers (the
        // silent-garbage trap: cublas reads a bogus device address without erroring)
        cublasStatus_t st = cublasGemmEx(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         N, M, K,
                                         gOne,
                                         dW16, CUDA_R_16F, N,
                                         a16, CUDA_R_16F, K,
                                         beta == 0.0f ? gZero : gOne,
                                         dC32, CUDA_R_32F, N,
                                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    }
    rc = 0;
done:
    if (a16) cudaFreeAsync(a16, gStream);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_f16acc16: like cu_matmul_f16w but with f16 ACCUMULATE (CUBLAS_COMPUTE_16F) — on
// GeForce/GA106 the f32-accumulate tensor path runs at HALF rate, so f16 accumulate ≈1.5-2×
// (the prefill lever, Tw61). f16 accumulator, f32 output C. APPROXIMATE: f16 accumulation
// over K loses precision → gate on accuracy.
int cu_matmul_f16acc16(const void* dA32, const void* dW16, void* dC16, int M, int K, int N) {
    int rc = -2;
    void* a16 = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (cudaMallocAsync(&a16, (size_t)M * K * sizeof(unsigned short), gStream) != cudaSuccess) { rc = -5; goto done; }
    if (launch_cvt_f16(dA32, a16, (long)M * K) != 0) { rc = -6; goto done; }
    {
        static const unsigned short h1 = 0x3C00, h0 = 0x0000; // f16 alpha=1/beta=0 (COMPUTE_16F scale type)
        cublasStatus_t st = cublasSetPointerMode(gHandle, CUBLAS_POINTER_MODE_HOST);
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasGemmEx(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                              &h1, dW16, CUDA_R_16F, N, a16, CUDA_R_16F, K,
                              &h0, dC16, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasSetPointerMode(gHandle, CUBLAS_POINTER_MODE_DEVICE); // restore for the decode graph path
        if (st != CUBLAS_STATUS_SUCCESS) { rc = -(4000 + (int)st); goto done; }
    }
    rc = 0;
done:
    if (a16) cudaFreeAsync(a16, gStream);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// launch_cvt_from_f16: dst[i] = f16→f32(src[i]) (add=0) or dst[i] += f16→f32(src[i]) (add=1).
// The residual add supports the beta=1 (o/down projection) path of the f16-accumulate GEMM.
static int launch_cvt_from_f16(const void* src16, void* dst32, long n, int add) {
    if (!gCvtFrom16 && compile_kernel(
            "__device__ __forceinline__ float f16f(unsigned short h){\n"
            "  unsigned s = (h & 0x8000u) << 16;\n"
            "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
            "  return __uint_as_float(__float_as_uint(v) | s);\n"
            "}\n"
            "extern \"C\" __global__ void cvt_f16_f32(const unsigned short* src, float* dst, long n, int add){\n"
            "  long i = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
            "  if (i >= n) return;\n"
            "  float v = f16f(src[i]);\n"
            "  dst[i] = add ? dst[i] + v : v;\n"
            "}\n",
            "cvt_f16_f32.cu", "cvt_f16_f32", &gCvtFrom16) != 0) return -2;
    int threads = 256; long blocks = (n + threads - 1) / threads;
    void* args[4] = { &src16, &dst32, &n, &add };
    return (cuLaunchKernel(gCvtFrom16, (unsigned)blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
}

// cu_matmul_f16w_acc16: drop-in twin of cu_matmul_f16w but with f16 ACCUMULATE (≈1.5-2× on
// GeForce). GEMM into an f16 scratch (COMPUTE_16F), then convert back to the f32 output C
// (beta=1 → residual add). Same signature/semantics as cu_matmul_f16w — the gated prefill
// path (GOAI_CUDA_F16ACC=1). APPROXIMATE: f16 accumulation (norm-rel ~2-5e-3, gate on a model).
int cu_matmul_f16w_acc16(const void* dA32, const void* dW16, void* dC32, int M, int K, int N, float beta) {
    int rc = -2;
    void* a16 = NULL; void* c16 = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (cudaMallocAsync(&a16, (size_t)M * K * sizeof(unsigned short), gStream) != cudaSuccess) { rc = -5; goto done; }
    if (cudaMallocAsync(&c16, (size_t)M * N * sizeof(unsigned short), gStream) != cudaSuccess) { rc = -5; goto done; }
    if (launch_cvt_f16(dA32, a16, (long)M * K) != 0) { rc = -6; goto done; }
    {
        static const unsigned short h1 = 0x3C00, h0 = 0x0000;
        cublasStatus_t st = cublasSetPointerMode(gHandle, CUBLAS_POINTER_MODE_HOST);
        if (st == CUBLAS_STATUS_SUCCESS)
            st = cublasGemmEx(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                              &h1, dW16, CUDA_R_16F, N, a16, CUDA_R_16F, K,
                              &h0, c16, CUDA_R_16F, N, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
        cublasSetPointerMode(gHandle, CUBLAS_POINTER_MODE_DEVICE);
        if (st != CUBLAS_STATUS_SUCCESS) { rc = -(4000 + (int)st); goto done; }
    }
    if (launch_cvt_from_f16(c16, dC32, (long)M * N, beta == 0.0f ? 0 : 1) != 0) { rc = -7; goto done; }
    rc = 0;
done:
    if (a16) cudaFreeAsync(a16, gStream);
    if (c16) cudaFreeAsync(c16, gStream);
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_download_u16: copy n u16 (f16) elements device→host (for validating f16-output GEMMs).
int cu_download_u16(const void* dsrc, unsigned short* dst, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0 && cuCtxSetCurrent(gCtx) == CUDA_SUCCESS)
        rc = (cudaMemcpy(dst, dsrc, (size_t)n * sizeof(unsigned short), cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -2;
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_alloc_i8: uninitialized device buffer of n bytes.
void* cu_alloc_i8(int n) {
    void* d = NULL;
    pthread_mutex_lock(&gLock);
    if (ensure_init() == 0 && cuCtxSetCurrent(gCtx) == CUDA_SUCCESS) {
        if (cudaMalloc(&d, (size_t)n) != cudaSuccess) d = NULL;
    }
    pthread_mutex_unlock(&gLock);
    return d;
}

// ---- Device-position (graph-capturable) op twins. The decode graph must be
// structurally constant across tokens, so the per-token position lives in a
// device int (updated between graph replays via cu_set_i32, not a launch param).

// cu_set_i32 writes one int to device buffer d (the shared decode position).
int cu_set_i32(void* d, int val) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemcpyAsync(d, &val, sizeof(int), cudaMemcpyHostToDevice, gStream) != cudaSuccess) { rc = -3; goto done; }
    rc = 0;
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_rope_f32_dpos: RoPE reading posOffset from device int *dPos (else identical
// to cu_rope_f32) — so a captured decode graph rotates at the token's true
// position without re-capture.
int cu_rope_f32_dpos(void* x, const void* inv, int seq, int heads, int hd, const void* dPos, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRopeDpos && compile_kernel(
                          "extern \"C\" __global__ void rope_f32_dpos(float* x, const float* inv, int seq, int heads, int hd, const int* dPos, double posDiv){\n"
                          "  int half = hd/2;\n"
                          "  long total = (long)seq*heads*half;\n"
                          "  long gid = (long)blockIdx.x*blockDim.x + threadIdx.x;\n"
                          "  if (gid >= total) return;\n"
                          "  int i = (int)(gid % half);\n"
                          "  int h = (int)((gid / half) % heads);\n"
                          "  int p = (int)(gid / ((long)half*heads));\n"
                          "  double pos = (double)(*dPos + p) / posDiv;\n"
                          "  double ang = pos * (double)inv[i];\n"
                          "  double c = cos(ang), s = sin(ang);\n"
                          "  float* xr = x + (size_t)p*heads*hd + (size_t)h*hd;\n"
                          "  double qi = xr[i], qih = xr[i+half];\n"
                          "  xr[i] = (float)(qi*c - qih*s);\n"
                          "  xr[i+half] = (float)(qih*c + qi*s);\n"
                          "}\n",
                          "rope_dpos.cu", "rope_f32_dpos", &gRopeDpos) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (hd / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &x; args[1] = &inv; args[2] = &seq; args[3] = &heads; args[4] = &hd; args[5] = &dPos; args[6] = &posDiv;
        rc = (cuLaunchKernel(gRopeDpos, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_attn_softmax_dpos: fused scale+causal-mask+softmax with the mask offset read
// from device int *dOff (else identical to cu_attn_softmax).
int cu_attn_softmax_dpos(void* x, int rows, int cols, float scale, const void* dOff, int seqQ) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAttnSoftmaxDpos && compile_kernel(
                                "extern \"C\" __global__ void attn_softmax_dpos(float* x, int rows, int cols, float scale, const int* dOff, int seqQ){\n"
                                "  int row=blockIdx.x; if(row>=rows) return;\n"
                                "  int lim = (row % seqQ) + *dOff; if(lim>=cols) lim=cols-1;\n"
                                "  extern __shared__ double sh[];\n"
                                "  int t=threadIdx.x, nt=blockDim.x;\n"
                                "  float* xr = x + (size_t)row*cols;\n"
                                "  double m=-1e300;\n"
                                "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double v=(double)xr[j]*scale; if(v>m)m=v; } }\n"
                                "  sh[t]=m; __syncthreads();\n"
                                "  for(int s=nt/2;s>0;s>>=1){ if(t<s && sh[t+s]>sh[t]) sh[t]=sh[t+s]; __syncthreads(); }\n"
                                "  double rowmax=sh[0]; __syncthreads();\n"
                                "  double local=0.0;\n"
                                "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double e=exp((double)xr[j]*scale-rowmax); xr[j]=(float)e; local+=e; } else { xr[j]=0.0f; } }\n"
                                "  sh[t]=local; __syncthreads();\n"
                                "  for(int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                                "  double inv=1.0/sh[0];\n"
                                "  for(int j=t;j<=lim;j+=nt){ xr[j]=(float)(xr[j]*inv); }\n"
                                "}\n",
                                "attn_softmax_dpos.cu", "attn_softmax_dpos", &gAttnSoftmaxDpos) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows; if (blocks < 1) blocks = 1;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[6];
        args[0] = &x; args[1] = &rows; args[2] = &cols; args[3] = &scale; args[4] = &dOff; args[5] = &seqQ;
        rc = (cuLaunchKernel(gAttnSoftmaxDpos, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_gqa_flash_dpos: FLASH decode attention (seqQ==1) — the structural win over
// the cuBLAS chain is GQA K/V SHARING: one block per (kv head, key chunk)
// stages K/V tiles into shared memory ONCE and serves ALL `group` query heads
// of that kv head (the chain reads K/V `group`× — 8× on TinyLlama). Online
// softmax (Milakov-Gimelshein) with split-K partials (m, l, unnormalized acc)
// written to a device scratch, then a small merge kernel combines the splitK
// partials per query head. Two launches, no [heads,seqKV] score matrix, K/V
// traffic cut by `group`. The causal limit comes from device int *dOff and the
// chunking from the fixed cache size — both graph-capturable.
int cu_gqa_flash_dpos(const void* dQ, const void* dK, const void* dV, void* dOut,
                      int seqKV, int qHeads, int kvHeads, int hd, float scale, const void* dOff) {
    static void* sPart = NULL; // grow-only [qHeads*splitK*(hd+2)] f32 partials
    static size_t cPart = 0;
    int rc = -1;
    int group = qHeads / kvHeads;
    if (hd > 128 || group > 8) return -4; // register/warp budget (all local models fit)
    int splitK = seqKV / 32; // ≥64 keys per chunk after tiling; capture-constant
    if (splitK < 1) splitK = 1;
    if (splitK > 32) splitK = 32;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gGqaFlashPart && compile_kernel(
                              "extern \"C\" __global__ void gqa_flash_partial(const float* q, const float* k, const float* v, float* part,\n"
                              "    int seqKV, int qHeads, int kvHeads, int hd, float scale, const int* dOff, int splitK){\n"
                              "  int kvh = blockIdx.x / splitK, c = blockIdx.x % splitK;\n"
                              "  int group = qHeads / kvHeads, WKV = kvHeads * hd;\n"
                              "  int lim = *dOff; if (lim >= seqKV) lim = seqKV - 1;\n"
                              "  int total = lim + 1;\n"
                              "  int chunk = (seqKV + splitK - 1) / splitK;\n"
                              "  int begin = c * chunk, end = begin + chunk;\n"
                              "  if (end > total) end = total;\n"
                              "  int t = threadIdx.x, nt = blockDim.x, lane = t & 31, w = t >> 5;\n"
                              "  int hp = hd + 1; // shared row padding against bank conflicts\n"
                              "  extern __shared__ float sh[];\n"
                              "  float* shq = sh;               // [group*hd] the group's query heads\n"
                              "  float* shk = shq + group * hd; // [32*hp] K tile\n"
                              "  float* shv = shk + 32 * hp;    // [32*hp] V tile\n"
                              "  for (int i = t; i < group * hd; i += nt) shq[i] = q[(size_t)kvh * group * hd + i];\n"
                              "  float NEGINF = __int_as_float(0xff800000);\n"
                              "  float m = NEGINF, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;\n"
                              "  for (int base = begin; base < end; base += 32) {\n"
                              "    int nk = end - base; if (nk > 32) nk = 32;\n"
                              "    __syncthreads();\n"
                              "    for (int i = t; i < nk * hd; i += nt) {\n"
                              "      int j = i / hd, d = i - j * hd;\n"
                              "      size_t src = (size_t)(base + j) * WKV + (size_t)kvh * hd + d;\n"
                              "      shk[j * hp + d] = k[src];\n"
                              "      shv[j * hp + d] = v[src];\n"
                              "    }\n"
                              "    __syncthreads();\n"
                              "    if (w < group) {\n"
                              "      float s = NEGINF;\n"
                              "      if (lane < nk) {\n"
                              "        const float* kr = shk + lane * hp;\n"
                              "        const float* qh = shq + w * hd;\n"
                              "        float dot = 0.f;\n"
                              "        for (int d = 0; d < hd; d++) dot += qh[d] * kr[d];\n"
                              "        s = dot * scale;\n"
                              "      }\n"
                              "      float bm = s;\n"
                              "      for (int o = 16; o > 0; o >>= 1) { float z = __shfl_down_sync(0xffffffffu, bm, o); if (z > bm) bm = z; }\n"
                              "      bm = __shfl_sync(0xffffffffu, bm, 0);\n"
                              "      float newm = m > bm ? m : bm;\n"
                              "      float corr = __expf(m - newm); // m==-INF -> 0\n"
                              "      float p = (lane < nk) ? __expf(s - newm) : 0.f;\n"
                              "      float bl = p;\n"
                              "      for (int o = 16; o > 0; o >>= 1) bl += __shfl_down_sync(0xffffffffu, bl, o);\n"
                              "      bl = __shfl_sync(0xffffffffu, bl, 0);\n"
                              "      l = l * corr + bl; m = newm;\n"
                              "      a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr;\n"
                              "      for (int j = 0; j < nk; j++) {\n"
                              "        float pj = __shfl_sync(0xffffffffu, p, j);\n"
                              "        const float* vr = shv + j * hp;\n"
                              "        if (lane < hd)      a0 += pj * vr[lane];\n"
                              "        if (lane + 32 < hd) a1 += pj * vr[lane + 32];\n"
                              "        if (lane + 64 < hd) a2 += pj * vr[lane + 64];\n"
                              "        if (lane + 96 < hd) a3 += pj * vr[lane + 96];\n"
                              "      }\n"
                              "    }\n"
                              "  }\n"
                              "  if (w < group) {\n"
                              "    int qh = kvh * group + w;\n"
                              "    float* o = part + (size_t)(qh * splitK + c) * (hd + 2);\n"
                              "    if (lane == 0) { o[0] = m; o[1] = l; }\n"
                              "    if (lane < hd)      o[2 + lane] = a0;\n"
                              "    if (lane + 32 < hd) o[2 + lane + 32] = a1;\n"
                              "    if (lane + 64 < hd) o[2 + lane + 64] = a2;\n"
                              "    if (lane + 96 < hd) o[2 + lane + 96] = a3;\n"
                              "  }\n"
                              "}\n",
                              "gqa_flash_partial.cu", "gqa_flash_partial", &gGqaFlashPart) != 0) { rc = -2; goto done; }
    if (!gGqaFlashMerge && compile_kernel(
                               "extern \"C\" __global__ void gqa_flash_merge(const float* part, float* out, int qHeads, int hd, int splitK){\n"
                               "  int qh = blockIdx.x; if (qh >= qHeads) return;\n"
                               "  int t = threadIdx.x, nt = blockDim.x;\n"
                               "  float NEGINF = __int_as_float(0xff800000);\n"
                               "  float M = NEGINF;\n"
                               "  for (int s = 0; s < splitK; s++) {\n"
                               "    const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "    if (p[1] > 0.f && p[0] > M) M = p[0];\n"
                               "  }\n"
                               "  float L = 0.f;\n"
                               "  for (int s = 0; s < splitK; s++) {\n"
                               "    const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "    if (p[1] > 0.f) L += p[1] * __expf(p[0] - M);\n"
                               "  }\n"
                               "  float invL = 1.f / L;\n"
                               "  for (int d = t; d < hd; d += nt) {\n"
                               "    float a = 0.f;\n"
                               "    for (int s = 0; s < splitK; s++) {\n"
                               "      const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "      if (p[1] > 0.f) a += __expf(p[0] - M) * p[2 + d];\n"
                               "    }\n"
                               "    out[qh * hd + d] = a * invL;\n"
                               "  }\n"
                               "}\n",
                               "gqa_flash_merge.cu", "gqa_flash_merge", &gGqaFlashMerge) != 0) { rc = -2; goto done; }
    if (ensure_devp(&sPart, &cPart, (size_t)qHeads * splitK * (hd + 2) * sizeof(float)) != 0) { rc = -9; goto done; }
    {
        int threads = 256;
        int blocks = kvHeads * splitK;
        size_t shmem = ((size_t)group * hd + 2u * 32u * (hd + 1)) * sizeof(float);
        void* args[11];
        args[0] = &dQ; args[1] = &dK; args[2] = &dV; args[3] = &sPart;
        args[4] = &seqKV; args[5] = &qHeads; args[6] = &kvHeads; args[7] = &hd;
        args[8] = &scale; args[9] = &dOff; args[10] = &splitK;
        if (cuLaunchKernel(gGqaFlashPart, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) != CUDA_SUCCESS) { rc = -3; goto done; }
        int mthreads = 128;
        void* margs[5];
        margs[0] = &sPart; margs[1] = &dOut; margs[2] = &qHeads; margs[3] = &hd; margs[4] = &splitK;
        rc = (cuLaunchKernel(gGqaFlashMerge, qHeads, 1, 1, mthreads, 1, 1, 0, (CUstream)gStream, margs, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_gqa_flash_f16_dpos: cu_gqa_flash_dpos over an f16 (u16-stored) KV cache —
// K/V global reads are HALF the bytes; tiles are converted to f32 in shared
// during staging, so the conversion cost is amortized over all `group` query
// heads and the math (dots, softmax, accumulation) is bit-identical to the f32
// kernel given the same tile values. Header-free h2f (nvrtc has no cuda_fp16.h
// here — mirrors the manual RN f32→f16 in kCvtF16ManualSrc).
int cu_gqa_flash_f16_dpos(const void* dQ, const void* dK16, const void* dV16, void* dOut,
                          int seqKV, int qHeads, int kvHeads, int hd, float scale, const void* dOff) {
    static void* sPart = NULL; // grow-only [qHeads*splitK*(hd+2)] f32 partials (own scratch — may interleave with the f32 path)
    static size_t cPart = 0;
    int rc = -1;
    int group = qHeads / kvHeads;
    if (hd > 128 || group > 8) return -4;
    int splitK = seqKV / 32;
    if (splitK < 1) splitK = 1;
    if (splitK > 32) splitK = 32;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gGqaFlashPartF16 && compile_kernel(
                                 "__device__ __forceinline__ float h2f(unsigned short h){\n"
                                 "  unsigned int sign = ((unsigned int)h & 0x8000u) << 16;\n"
                                 "  unsigned int em = h & 0x7fffu;\n"
                                 "  unsigned int f;\n"
                                 "  if (em >= 0x7c00u) f = sign | 0x7f800000u | ((em & 0x3ffu) << 13);\n"
                                 "  else if (em >= 0x0400u) f = sign | ((em + 0x1c000u) << 13);\n"
                                 "  else if (em == 0u) f = sign;\n"
                                 "  else {\n"
                                 "    int e = 113; unsigned int m = em;\n"
                                 "    while (!(m & 0x400u)) { m <<= 1; e--; }\n"
                                 "    f = sign | ((unsigned int)e << 23) | ((m & 0x3ffu) << 13);\n"
                                 "  }\n"
                                 "  return __uint_as_float(f);\n"
                                 "}\n"
                                 "extern \"C\" __global__ void gqa_flash_partial_f16(const float* q, const unsigned short* k, const unsigned short* v, float* part,\n"
                                 "    int seqKV, int qHeads, int kvHeads, int hd, float scale, const int* dOff, int splitK){\n"
                                 "  int kvh = blockIdx.x / splitK, c = blockIdx.x % splitK;\n"
                                 "  int group = qHeads / kvHeads, WKV = kvHeads * hd;\n"
                                 "  int lim = *dOff; if (lim >= seqKV) lim = seqKV - 1;\n"
                                 "  int total = lim + 1;\n"
                                 "  int chunk = (seqKV + splitK - 1) / splitK;\n"
                                 "  int begin = c * chunk, end = begin + chunk;\n"
                                 "  if (end > total) end = total;\n"
                                 "  int t = threadIdx.x, nt = blockDim.x, lane = t & 31, w = t >> 5;\n"
                                 "  int hp = hd + 1;\n"
                                 "  extern __shared__ float sh[];\n"
                                 "  float* shq = sh;\n"
                                 "  float* shk = shq + group * hd;\n"
                                 "  float* shv = shk + 32 * hp;\n"
                                 "  for (int i = t; i < group * hd; i += nt) shq[i] = q[(size_t)kvh * group * hd + i];\n"
                                 "  float NEGINF = __int_as_float(0xff800000);\n"
                                 "  float m = NEGINF, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;\n"
                                 "  for (int base = begin; base < end; base += 32) {\n"
                                 "    int nk = end - base; if (nk > 32) nk = 32;\n"
                                 "    __syncthreads();\n"
                                 "    for (int i = t; i < nk * hd; i += nt) {\n"
                                 "      int j = i / hd, d = i - j * hd;\n"
                                 "      size_t src = (size_t)(base + j) * WKV + (size_t)kvh * hd + d;\n"
                                 "      shk[j * hp + d] = h2f(k[src]);\n"
                                 "      shv[j * hp + d] = h2f(v[src]);\n"
                                 "    }\n"
                                 "    __syncthreads();\n"
                                 "    if (w < group) {\n"
                                 "      float s = NEGINF;\n"
                                 "      if (lane < nk) {\n"
                                 "        const float* kr = shk + lane * hp;\n"
                                 "        const float* qh = shq + w * hd;\n"
                                 "        float dot = 0.f;\n"
                                 "        for (int d = 0; d < hd; d++) dot += qh[d] * kr[d];\n"
                                 "        s = dot * scale;\n"
                                 "      }\n"
                                 "      float bm = s;\n"
                                 "      for (int o = 16; o > 0; o >>= 1) { float z = __shfl_down_sync(0xffffffffu, bm, o); if (z > bm) bm = z; }\n"
                                 "      bm = __shfl_sync(0xffffffffu, bm, 0);\n"
                                 "      float newm = m > bm ? m : bm;\n"
                                 "      float corr = __expf(m - newm);\n"
                                 "      float p = (lane < nk) ? __expf(s - newm) : 0.f;\n"
                                 "      float bl = p;\n"
                                 "      for (int o = 16; o > 0; o >>= 1) bl += __shfl_down_sync(0xffffffffu, bl, o);\n"
                                 "      bl = __shfl_sync(0xffffffffu, bl, 0);\n"
                                 "      l = l * corr + bl; m = newm;\n"
                                 "      a0 *= corr; a1 *= corr; a2 *= corr; a3 *= corr;\n"
                                 "      for (int j = 0; j < nk; j++) {\n"
                                 "        float pj = __shfl_sync(0xffffffffu, p, j);\n"
                                 "        const float* vr = shv + j * hp;\n"
                                 "        if (lane < hd)      a0 += pj * vr[lane];\n"
                                 "        if (lane + 32 < hd) a1 += pj * vr[lane + 32];\n"
                                 "        if (lane + 64 < hd) a2 += pj * vr[lane + 64];\n"
                                 "        if (lane + 96 < hd) a3 += pj * vr[lane + 96];\n"
                                 "      }\n"
                                 "    }\n"
                                 "  }\n"
                                 "  if (w < group) {\n"
                                 "    int qh = kvh * group + w;\n"
                                 "    float* o = part + (size_t)(qh * splitK + c) * (hd + 2);\n"
                                 "    if (lane == 0) { o[0] = m; o[1] = l; }\n"
                                 "    if (lane < hd)      o[2 + lane] = a0;\n"
                                 "    if (lane + 32 < hd) o[2 + lane + 32] = a1;\n"
                                 "    if (lane + 64 < hd) o[2 + lane + 64] = a2;\n"
                                 "    if (lane + 96 < hd) o[2 + lane + 96] = a3;\n"
                                 "  }\n"
                                 "}\n",
                                 "gqa_flash_partial_f16.cu", "gqa_flash_partial_f16", &gGqaFlashPartF16) != 0) { rc = -2; goto done; }
    if (!gGqaFlashMerge && compile_kernel(
                               "extern \"C\" __global__ void gqa_flash_merge(const float* part, float* out, int qHeads, int hd, int splitK){\n"
                               "  int qh = blockIdx.x; if (qh >= qHeads) return;\n"
                               "  int t = threadIdx.x, nt = blockDim.x;\n"
                               "  float NEGINF = __int_as_float(0xff800000);\n"
                               "  float M = NEGINF;\n"
                               "  for (int s = 0; s < splitK; s++) {\n"
                               "    const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "    if (p[1] > 0.f && p[0] > M) M = p[0];\n"
                               "  }\n"
                               "  float L = 0.f;\n"
                               "  for (int s = 0; s < splitK; s++) {\n"
                               "    const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "    if (p[1] > 0.f) L += p[1] * __expf(p[0] - M);\n"
                               "  }\n"
                               "  float invL = 1.f / L;\n"
                               "  for (int d = t; d < hd; d += nt) {\n"
                               "    float a = 0.f;\n"
                               "    for (int s = 0; s < splitK; s++) {\n"
                               "      const float* p = part + (size_t)(qh * splitK + s) * (hd + 2);\n"
                               "      if (p[1] > 0.f) a += __expf(p[0] - M) * p[2 + d];\n"
                               "    }\n"
                               "    out[qh * hd + d] = a * invL;\n"
                               "  }\n"
                               "}\n",
                               "gqa_flash_merge.cu", "gqa_flash_merge", &gGqaFlashMerge) != 0) { rc = -2; goto done; }
    if (ensure_devp(&sPart, &cPart, (size_t)qHeads * splitK * (hd + 2) * sizeof(float)) != 0) { rc = -9; goto done; }
    {
        int threads = 256;
        int blocks = kvHeads * splitK;
        size_t shmem = ((size_t)group * hd + 2u * 32u * (hd + 1)) * sizeof(float);
        void* args[11];
        args[0] = &dQ; args[1] = &dK16; args[2] = &dV16; args[3] = &sPart;
        args[4] = &seqKV; args[5] = &qHeads; args[6] = &kvHeads; args[7] = &hd;
        args[8] = &scale; args[9] = &dOff; args[10] = &splitK;
        if (cuLaunchKernel(gGqaFlashPartF16, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) != CUDA_SUCCESS) { rc = -3; goto done; }
        int mthreads = 128;
        void* margs[5];
        margs[0] = &sPart; margs[1] = &dOut; margs[2] = &qHeads; margs[3] = &hd; margs[4] = &splitK;
        rc = (cuLaunchKernel(gGqaFlashMerge, qHeads, 1, 1, mthreads, 1, 1, 0, (CUstream)gStream, margs, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_append_dpos_f16 converts one token's wkv f32 values to f16 (round-to-
// nearest-even, mirrors kCvtF16ManualSrc) and writes them into the u16 cache at
// row *dPos — the f16 KV-cache append (graph-capturable).
int cu_append_dpos_f16(void* dst16, const void* src, const void* dPos, int wkv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAppendDposF16 && compile_kernel(
                               "extern \"C\" __global__ void append_dpos_f16(unsigned short* dst, const float* src, const int* dPos, int wkv){\n"
                               "  int j = blockIdx.x*blockDim.x + threadIdx.x;\n"
                               "  if (j >= wkv) return;\n"
                               "  unsigned int x = __float_as_uint(src[j]);\n"
                               "  unsigned int sign = (x >> 16) & 0x8000u;\n"
                               "  unsigned int em = x & 0x7fffffffu;\n"
                               "  unsigned short h;\n"
                               "  if (em >= 0x47800000u) { h = (unsigned short)(sign | (em > 0x7f800000u ? 0x7e00u : 0x7c00u)); }\n"
                               "  else if (em < 0x38800000u) {\n"
                               "    unsigned int m = (em & 0x007fffffu) | 0x00800000u;\n"
                               "    int shift = 126 - (int)(em >> 23);\n"
                               "    unsigned int v = (shift < 32) ? (m >> shift) : 0u;\n"
                               "    unsigned int r = (shift < 33 && shift > 0) ? ((m >> (shift-1)) & 1u) : 0u;\n"
                               "    unsigned int s = (shift < 32 && shift > 0 && (m & ((1u << (shift-1)) - 1u))) ? 1u : 0u;\n"
                               "    v += (r & (s | (v & 1u)));\n"
                               "    h = (unsigned short)(sign | v);\n"
                               "  } else {\n"
                               "    unsigned int e = ((em >> 23) - 112u) << 10;\n"
                               "    unsigned int m = (em >> 13) & 0x3ffu;\n"
                               "    unsigned int r = (em >> 12) & 1u;\n"
                               "    unsigned int s = (em & 0xfffu) ? 1u : 0u;\n"
                               "    unsigned int v = (e | m) + (r & (s | (m & 1u)));\n"
                               "    h = (unsigned short)(sign | v);\n"
                               "  }\n"
                               "  dst[(size_t)(*dPos)*wkv + j] = h;\n"
                               "}\n",
                               "append_dpos_f16.cu", "append_dpos_f16", &gAppendDposF16) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (wkv + threads - 1) / threads; if (blocks < 1) blocks = 1;
        void* args[4];
        args[0] = &dst16; args[1] = &src; args[2] = &dPos; args[3] = &wkv;
        rc = (cuLaunchKernel(gAppendDposF16, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_append_dpos writes wkv floats from src into dst at row offset *dPos (device
// int) — the KV-cache append with a device-side position (graph-capturable).
int cu_append_dpos(void* dst, const void* src, const void* dPos, int wkv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAppendDpos && compile_kernel(
                            "extern \"C\" __global__ void append_dpos(float* dst, const float* src, const int* dPos, int wkv){\n"
                            "  int j = blockIdx.x*blockDim.x + threadIdx.x;\n"
                            "  if (j >= wkv) return;\n"
                            "  dst[(size_t)(*dPos)*wkv + j] = src[j];\n"
                            "}\n",
                            "append_dpos.cu", "append_dpos", &gAppendDpos) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (wkv + threads - 1) / threads; if (blocks < 1) blocks = 1;
        void* args[4];
        args[0] = &dst; args[1] = &src; args[2] = &dPos; args[3] = &wkv;
        rc = (cuLaunchKernel(gAppendDpos, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_upload_into: H2D copy of n floats into an EXISTING device buffer (dst keeps its
// pointer, so references stay valid) — the buffer.UploadF32 the llamagpu recorder
// needs to (re)fill a resident buffer. Synchronous (host src must stay live only
// for the call).
int cu_upload_into(void* dst, const float* src, int n) {
    int rc = -6;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cudaMemcpyAsync(dst, src, (size_t)n * sizeof(float), cudaMemcpyHostToDevice, gStream) != cudaSuccess) { goto done; }
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
                     gOne,
                     (const float*)dB, N,
                     (const float*)dA, K,
                     gZero,
                     (float*)dC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_f32_ddd_acc: dC = dA·dB + dC (beta=1) — fuses the transformer residual
// add into the projection matmul. dC must already hold the residual; the gemm
// accumulates in place, saving a separate elementwise-add kernel launch + its
// output allocation on the decode hot path. Same column-major idiom as ddd.
int cu_matmul_f32_ddd_acc(const void* dA, const void* dB, void* dC, int M, int K, int N) {
    const float alpha = 1.0f, beta = 1.0f;
    cublasStatus_t st;
    int rc = -2;

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }

    st = cublasSgemm(gHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                     N, M, K,
                     gOne,
                     (const float*)dB, N,
                     (const float*)dA, K,
                     gOne,
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
                                  seq, seq, hd, gOne,
                                  (const float*)dK, W, hd,
                                  (const float*)dQ, W, hd, gZero,
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
                                  hd, seq, seq, gOne,
                                  (const float*)dV, W, hd,
                                  (const float*)dScores, seq, (long long)seq * seq, gZero,
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
int cu_gqa_scores(const void* dQ, const void* dK, void* dScores, int seqQ, int seqKV, int qHeads, int kvHeads, int hd, int tf32) {
    // Persistent (grow-only, gLock-guarded) host + device pointer arrays: the
    // attention pointer lists are rebuilt each call but reuse one allocation, and
    // the uploads ride gStream (async) so there is no per-call cudaMalloc/cudaFree
    // and no device-wide sync — the decode-latency win (§PERF).
    static void *sK = NULL, *sQ = NULL, *sC = NULL;
    static size_t cK = 0, cQ = 0, cC = 0;
    int group = qHeads / kvHeads, WQ = qHeads * hd, WKV = kvHeads * hd, rc = -2;
    size_t asz = (size_t)qHeads * sizeof(void*);

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (ensure_devp(&sK, &cK, asz) != 0 || ensure_devp(&sQ, &cQ, asz) != 0 ||
        ensure_devp(&sC, &cC, asz) != 0) { rc = -9; goto done; }
    // Build the pointer arrays on device (graph-capture-safe — no host memcpy):
    // K grouped (h/group)·hd, Q per-head h·hd, scores per-head h·(seqQ·seqKV).
    if (cu_build_batch_ptrs(dK, dQ, dScores, sK, sQ, sC, qHeads, group,
                            hd, hd, (long)seqQ * seqKV) != 0) { rc = -3; goto done; }
    if (tf32) cublasSetMathMode(gHandle, CUBLAS_TF32_TENSOR_OP_MATH); // opt-in: prefill only (§PERF-F16-PREFILL follow-up); decode parity paths pass 0
    if (cublasSgemmBatched(gHandle, CUBLAS_OP_T, CUBLAS_OP_N, seqKV, seqQ, hd, gOne,
                           (const float* const*)sK, WKV, (const float* const*)sQ, WQ, gZero,
                           (float* const*)sC, seqKV, qHeads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto tf32done; }
    rc = 0;
tf32done:
    if (tf32) cublasSetMathMode(gHandle, CUBLAS_DEFAULT_MATH);
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_gqa_out: out[h] = scores[h]·V[h/group] for every query head, into [seqQ,WQ].
// scores[h] is [seqQ, seqKV], V is [seqKV, WKV]. Full prefill passes seqQ==seqKV.
int cu_gqa_out(const void* dScores, const void* dV, void* dOut, int seqQ, int seqKV, int qHeads, int kvHeads, int hd, int tf32) {
    // Persistent grow-only pointer arrays (see cu_gqa_scores) — no per-call
    // cudaMalloc/cudaFree, uploads on gStream.
    static void *sV = NULL, *sS = NULL, *sO = NULL;
    static size_t cV = 0, cS = 0, cO = 0;
    int group = qHeads / kvHeads, WQ = qHeads * hd, WKV = kvHeads * hd, rc = -2;
    size_t asz = (size_t)qHeads * sizeof(void*);

    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (ensure_devp(&sV, &cV, asz) != 0 || ensure_devp(&sS, &cS, asz) != 0 ||
        ensure_devp(&sO, &cO, asz) != 0) { rc = -9; goto done; }
    // Device-built pointer arrays (graph-safe): V grouped (h/group)·hd, scores
    // per-head h·(seqQ·seqKV), out per-head h·hd.
    if (cu_build_batch_ptrs(dV, dScores, dOut, sV, sS, sO, qHeads, group,
                            hd, (long)seqQ * seqKV, hd) != 0) { rc = -3; goto done; }
    if (tf32) cublasSetMathMode(gHandle, CUBLAS_TF32_TENSOR_OP_MATH);
    if (cublasSgemmBatched(gHandle, CUBLAS_OP_N, CUBLAS_OP_N, hd, seqQ, seqKV, gOne,
                           (const float* const*)sV, WKV, (const float* const*)sS, seqKV, gZero,
                           (float* const*)sO, WQ, qHeads) != CUBLAS_STATUS_SUCCESS) { rc = -4; goto tf32done2; }
    rc = 0;
tf32done2:
    if (tf32) cublasSetMathMode(gHandle, CUBLAS_DEFAULT_MATH);
done:
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
                     gOne,
                     (const float*)dB, K,
                     (const float*)dA, K,
                     gZero,
                     (float*)dC, N);
    if (st != CUBLAS_STATUS_SUCCESS) { rc = -4; goto done; }
    rc = 0;

done:
    pthread_mutex_unlock(&gLock);
    return rc;
}
