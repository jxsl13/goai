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
static CUfunction gGelu = NULL, gRelu2 = NULL, gMoeGate = NULL, gRowAxpy = NULL, gSsmStep = NULL, gSsdStep = NULL, gConv1dStep = NULL, gSilu = NULL, gSigmoid = NULL, gSoftplus = NULL, gAdd = NULL, gMul = NULL, gRms = NULL, gSoftmax = NULL, gRope = NULL, gRopePartial = NULL, gCausal = NULL, gCausalMH = NULL, gEmbed = NULL, gSwiglu = NULL, gAttnSoftmax = NULL, gAttnSoftmaxCap = NULL, gAttnSoftmaxAlibi = NULL, gQgemv = NULL, gQgemv4 = NULL, gQgemv4k = NULL, gQgemv4kPre = NULL, gQgemv5k = NULL, gQgemv6k = NULL, gQgemv3k = NULL, gQgemv2k = NULL, gQgemv40 = NULL, gQgemvI4nl = NULL, gQgemvI4xs = NULL, gQgemvMxfp4 = NULL, gQgemvI2xxs = NULL, gQgemvI2xs = NULL, gQgemvI3xxs = NULL, gQgemvI3s = NULL, gQgemvI1s = NULL, gQgemvI1m = NULL, gI8Mma = NULL, gI8MmaT = NULL, gI8MmaRb = NULL, gI8MmaDb = NULL, gI8MmaWt = NULL, gI8MmaWp = NULL, gI8Mmq = NULL, gI8MmqR = NULL, gQrowsI8 = NULL, gLdmProbe = NULL, gLdmProbe2 = NULL, gI8MmaLm = NULL, gCvtF16 = NULL, gCvtFrom16 = NULL; // lazily nvrtc-compiled
static CUfunction gRopeDpos = NULL, gRopePartialDpos = NULL, gAttnSoftmaxDpos = NULL, gAppendDpos = NULL; // device-position (graph-capturable) twins
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

// cu_row_axpy: dst[r,c] += arow[r]·src[r,c] — a per-ROW scalar AXPY. The MoE combine: scale an
// expert's [rows, cols] output by that token's routing weight (arow, one scalar per row) and
// accumulate into the running mixture output. arow is a [rows] device vector.
int cu_row_axpy(void* dst, const void* src, const void* arow, int rows, int cols) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRowAxpy && compile_kernel(
                      "extern \"C\" __global__ void row_axpy(float* dst, const float* src, const float* arow, int rows, int cols){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x; long n=(long)rows*cols;\n"
                      "  if (i < n){ dst[i] += arow[i/cols] * src[i]; }\n"
                      "}\n",
                      "row_axpy.cu", "row_axpy", &gRowAxpy) != 0) { rc = -2; goto done; }
    {
        long n = (long)rows * cols; int threads = 256; int blocks = (int)((n + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[5]; args[0]=&dst; args[1]=&src; args[2]=&arow; args[3]=&rows; args[4]=&cols;
        rc = (cuLaunchKernel(gRowAxpy, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_moe_gate: Mixtral-style top-k routing. For each of `rows` tokens, given E raw router logits,
// select the top-K, and write weights[E] = softmax over the SELECTED logits (exp(l)/Σ_topk exp) for
// the K chosen experts and 0 for the rest — the renormalized combine weights. One block per row,
// single-threaded (E and K are small). weights must be a separate [rows·E] buffer from logits.
int cu_moe_gate(const void* logits, void* weights, int rows, int E, int K, int raw, float scale) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // renorm (raw==0, Mixtral): weight[sel] = softmax over the SELECTED top-k logits, sums to 1.
    // raw (raw==1, DeepSeek-V2): weight[sel] = softmax over ALL E logits · scale, NOT renormalized.
    if (!gMoeGate && compile_kernel(
                      "extern \"C\" __global__ void moe_gate(const float* logits, float* weights, int rows, int E, int K, int raw, float scale){\n"
                      "  int r=blockIdx.x; if(r>=rows) return; if(threadIdx.x!=0) return;\n"
                      "  const float* lg = logits + (size_t)r*E;\n"
                      "  float* w = weights + (size_t)r*E;\n"
                      "  for(int e=0;e<E;e++) w[e]=0.0f;\n"
                      "  int kk = K<E ? K : E;\n"
                      "  for(int p=0;p<kk;p++){\n"           // K passes: mark the p-th largest with -1
                      "    int best=-1; double bv=-1e300;\n"
                      "    for(int e=0;e<E;e++){ if(w[e]==0.0f && (double)lg[e]>bv){ bv=(double)lg[e]; best=e; } }\n"
                      "    if(best>=0) w[best]=-1.0f;\n"
                      "  }\n"
                      "  double mx=-1e300;\n"                // max over ALL (raw) or SELECTED (renorm)
                      "  for(int e=0;e<E;e++){ if((raw || w[e]==-1.0f) && (double)lg[e]>mx) mx=(double)lg[e]; }\n"
                      "  double s=0.0;\n"
                      "  for(int e=0;e<E;e++){ if(raw || w[e]==-1.0f) s+=exp((double)lg[e]-mx); }\n"
                      "  for(int e=0;e<E;e++){ w[e] = (w[e]==-1.0f) ? (float)(exp((double)lg[e]-mx)/s*(raw?(double)scale:1.0)) : 0.0f; }\n"
                      "}\n",
                      "moe_gate.cu", "moe_gate", &gMoeGate) != 0) { rc = -2; goto done; }
    {
        void* args[7]; args[0]=&logits; args[1]=&weights; args[2]=&rows; args[3]=&E; args[4]=&K; args[5]=&raw; args[6]=&scale;
        rc = (cuLaunchKernel(gMoeGate, rows<1?1:rows, 1, 1, 32, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_ssm_step: one timestep of the Mamba selective-scan (S6) recurrence for DECODE — advances the
// per-channel state h[D,N] and emits y[D], matching backend/ref selectiveScanKernel one step at a
// time (zero-order-hold Ā=exp(Δ·A), simplified B̄=Δ·B; f64 accumulation):
//   h[d,n] = exp(Δ[d]·A[d,n])·h[d,n] + Δ[d]·B[n]·u[d];  y[d] = Σ_n C[d? or shared]·h[d,n] + Dskip[d]·u[d]
// B,C are [N] (shared across the D channels) for this token; h is updated in place. One thread per d.
int cu_ssm_step(const void* u, const void* delta, const void* A, const void* B, const void* C,
                const void* dskip, void* h, void* y, int D, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gSsmStep && compile_kernel(
                      "extern \"C\" __global__ void ssm_step(const float* u, const float* delta, const float* A, const float* B, const float* C, const float* dskip, float* h, float* y, int D, int N){\n"
                      "  int d = blockIdx.x*blockDim.x + threadIdx.x; if(d>=D) return;\n"
                      "  double dl=(double)delta[d], ud=(double)u[d], acc=0.0;\n"
                      "  const float* ad = A + (size_t)d*N; float* hd = h + (size_t)d*N;\n"
                      "  for(int n=0;n<N;n++){\n"
                      "    double dA = exp(dl*(double)ad[n]);\n"
                      "    double hv = dA*(double)hd[n] + dl*(double)B[n]*ud;\n"
                      "    hd[n] = (float)hv; acc += (double)C[n]*hv;\n"
                      "  }\n"
                      "  y[d] = (float)(acc + (double)dskip[d]*ud);\n"
                      "}\n",
                      "ssm_step.cu", "ssm_step", &gSsmStep) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (D + threads - 1) / threads; if (blocks < 1) blocks = 1;
        void* args[10]; args[0]=&u; args[1]=&delta; args[2]=&A; args[3]=&B; args[4]=&C; args[5]=&dskip; args[6]=&h; args[7]=&y; args[8]=&D; args[9]=&N;
        rc = (cuLaunchKernel(gSsmStep, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_conv1d_step: one decode step of Mamba's causal depthwise conv (backend/ref conv1dKernel:
// out[t,c] = Σ_k w[c,k]·x[t−(K−1)+k,c] + b[c]). state[D,K-1] holds this channel's last K−1 inputs
// (oldest first). out[c] = Σ_{k<K-1} w[c,k]·state[c,k] + w[c,K-1]·x[c] + b[c]; then state shifts left
// and appends x (state[c,K-2]=x[c]). state starts zeroed (the causal left-pad). One thread per channel.
int cu_conv1d_step(const void* x, const void* w, const void* b, void* state, void* out, int D, int K) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gConv1dStep && compile_kernel(
                      "extern \"C\" __global__ void conv1d_step(const float* x, const float* w, const float* b, float* state, float* out, int D, int K){\n"
                      "  int c = blockIdx.x*blockDim.x + threadIdx.x; if(c>=D) return;\n"
                      "  int S = K-1; const float* wc = w + (size_t)c*K; float* sc = state + (size_t)c*S;\n"
                      "  double acc = (double)b[c] + (double)wc[K-1]*(double)x[c];\n"
                      "  for(int k=0;k<S;k++) acc += (double)wc[k]*(double)sc[k];\n"
                      "  out[c] = (float)acc;\n"
                      "  for(int k=0;k<S-1;k++) sc[k]=sc[k+1];\n"  // shift the window left...
                      "  if(S>0) sc[S-1]=x[c];\n"                  // ...and append the new input
                      "}\n",
                      "conv1d_step.cu", "conv1d_step", &gConv1dStep) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = (D + threads - 1) / threads; if (blocks < 1) blocks = 1;
        void* args[7]; args[0]=&x; args[1]=&w; args[2]=&b; args[3]=&state; args[4]=&out; args[5]=&D; args[6]=&K;
        rc = (cuLaunchKernel(gConv1dStep, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_ssd_step: one timestep of the Mamba-2 SSD (state-space-duality) scan for DECODE. Where Mamba-1
// (cu_ssm_step) has a per-channel state-matrix decay, Mamba-2 restricts A to a SCALAR per head and
// shares B/C across a GROUP of heads. Per head h (group g = h/(H/G)): a = exp(Δ[h]·A[h]) (with A[h] =
// −exp(A_log)); the head's value is Δ-scaled; state[N,P] (P = head_dim):
//   state[i,j] = a·state[i,j] + B[g,i]·(Δ[h]·x[h,j]);  y[h,j] = Σ_i C[g,i]·state[i,j] + Dskip[h]·x[h,j]
// matching nn.SSDRecurrent applied per head + the D·x skip (f64 accumulation). x/y are [H·P] =
// intermediate; B,C are [G·N]; delta,A,Dskip are [H]; state is [H·N·P] (per head [N,P], i outer,
// stride P between successive n). One thread per (h,j) — H·P threads.
int cu_ssd_step(const void* x, const void* delta, const void* A, const void* B, const void* C,
                const void* dskip, void* state, void* y, int H, int P, int G, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gSsdStep && compile_kernel(
                      "extern \"C\" __global__ void ssd_step(const float* x, const float* delta, const float* A, const float* B, const float* C, const float* dskip, float* state, float* y, int H, int P, int G, int N){\n"
                      "  int idx = blockIdx.x*blockDim.x + threadIdx.x; int HP = H*P; if(idx>=HP) return;\n"
                      "  int h = idx / P; int g = h / (H / G);\n"
                      "  double dl = (double)delta[h];\n"
                      "  double a = exp(dl * (double)A[h]);\n"
                      "  double xv = (double)x[idx];\n"
                      "  double xin = dl * xv;\n"
                      "  const float* Bg = B + (size_t)g*N; const float* Cg = C + (size_t)g*N;\n"
                      "  float* st = state + (size_t)h*N*P + (idx % P);\n"  // state[h][i][j], stride P over i
                      "  double acc = 0.0;\n"
                      "  for(int i=0;i<N;i++){\n"
                      "    double s = a*(double)st[(size_t)i*P] + (double)Bg[i]*xin;\n"
                      "    st[(size_t)i*P] = (float)s; acc += (double)Cg[i]*s;\n"
                      "  }\n"
                      "  y[idx] = (float)(acc + (double)dskip[h]*xv);\n"
                      "}\n",
                      "ssd_step.cu", "ssd_step", &gSsdStep) != 0) { rc = -2; goto done; }
    {
        int HP = H*P, threads = 256, blocks = (HP + threads - 1) / threads; if (blocks < 1) blocks = 1;
        void* args[12]; args[0]=&x; args[1]=&delta; args[2]=&A; args[3]=&B; args[4]=&C; args[5]=&dskip; args[6]=&state; args[7]=&y; args[8]=&H; args[9]=&P; args[10]=&G; args[11]=&N;
        rc = (cuLaunchKernel(gSsdStep, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
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

int cu_relu2_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // squared ReLU (Nemotron hidden_act="relu2"): relu(x)² = x>0 ? x·x : 0, in place.
    if (!gRelu2 && compile_kernel(
                      "extern \"C\" __global__ void relu2_f32(float* x, int n){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (i < n){ float v = x[i]; x[i] = v > 0.0f ? v*v : 0.0f; }\n"
                      "}\n",
                      "relu2.cu", "relu2_f32", &gRelu2) != 0) { rc = -2; goto done; }
    rc = launch_unary(gRelu2, d, n);
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_softplus_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // softplus log(1+e^x), stable as max(x,0)+log(1+e^-|x|) (matches backend/ref); Mamba's Δ>0.
    if (!gSoftplus && compile_kernel(
                      "extern \"C\" __global__ void softplus_f32(float* x, int n){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (i < n){ float v = x[i]; x[i] = fmaxf(v,0.0f) + log1pf(expf(-fabsf(v))); }\n"
                      "}\n",
                      "softplus.cu", "softplus_f32", &gSoftplus) != 0) { rc = -2; goto done; }
    rc = launch_unary(gSoftplus, d, n);
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

int cu_sigmoid_f32(void* d, int n) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // sigmoid(x) = 1/(1+e^-x), stable branch at 0 — matches backend/ref (Qwen2-MoE shared-expert gate).
    if (!gSigmoid && compile_kernel(
                      "extern \"C\" __global__ void sigmoid_f32(float* x, int n){\n"
                      "  int i = blockIdx.x*blockDim.x + threadIdx.x;\n"
                      "  if (i < n){ float v = x[i];\n"
                      "    x[i] = v>=0.0f ? 1.0f/(1.0f+expf(-v)) : expf(v)/(1.0f+expf(v)); }\n"
                      "}\n",
                      "sigmoid.cu", "sigmoid_f32", &gSigmoid) != 0) { rc = -2; goto done; }
    rc = launch_unary(gSigmoid, d, n);
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
// cu_attn_softmax_cap: like cu_attn_softmax but applies Gemma 2's attention-logit soft-cap
// s = cap·tanh(score·scale / cap) to each SCALED score BEFORE the mask + softmax (matching HF
// Gemma2Attention: scores·scale → softcap → +mask → softmax). cap must be > 0.
int cu_attn_softmax_cap(void* x, int rows, int cols, float scale, int offset, int seqQ, float cap) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAttnSoftmaxCap && compile_kernel(
                             "extern \"C\" __global__ void attn_softmax_cap(float* x, int rows, int cols, float scale, int offset, int seqQ, float cap){\n"
                             "  int row=blockIdx.x; if(row>=rows) return;\n"
                             "  int lim = (row % seqQ) + offset; if(lim>=cols) lim=cols-1;\n"
                             "  extern __shared__ double sh[];\n"
                             "  int t=threadIdx.x, nt=blockDim.x; double dc=(double)cap;\n"
                             "  float* xr = x + (size_t)row*cols;\n"
                             "  double m=-1e300;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double v=dc*tanh((double)xr[j]*scale/dc); if(v>m)m=v; } }\n"
                             "  sh[t]=m; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s && sh[t+s]>sh[t]) sh[t]=sh[t+s]; __syncthreads(); }\n"
                             "  double rowmax=sh[0]; __syncthreads();\n"
                             "  double local=0.0;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double cs=dc*tanh((double)xr[j]*scale/dc); double e=exp(cs-rowmax); xr[j]=(float)e; local+=e; } else { xr[j]=0.0f; } }\n"
                             "  sh[t]=local; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                             "  double inv=1.0/sh[0];\n"
                             "  for(int j=t;j<=lim;j+=nt){ xr[j]=(float)(xr[j]*inv); }\n"
                             "}\n",
                             "attn_softmax_cap.cu", "attn_softmax_cap", &gAttnSoftmaxCap) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows; if (blocks < 1) blocks = 1;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[7];
        args[0] = &x; args[1] = &rows; args[2] = &cols; args[3] = &scale;
        args[4] = &offset; args[5] = &seqQ; args[6] = &cap;
        rc = (cuLaunchKernel(gAttnSoftmaxCap, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_attn_softmax_alibi: like cu_attn_softmax but adds an ALiBi position bias slopeₕ·(j − qabs) to
// each SCALED score before the mask+softmax (Press et al. 2021). Row r is query qi=r%seqQ in head
// h=r/seqQ; its absolute position qabs = offset+qi is also the causal limit. slopes is a device
// array of `heads` per-head slopes (backend.ALiBiSlopes). No RoPE is used with this path.
int cu_attn_softmax_alibi(void* x, int rows, int cols, float scale, int offset, int seqQ, const void* slopes) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gAttnSoftmaxAlibi && compile_kernel(
                             "extern \"C\" __global__ void attn_softmax_alibi(float* x, int rows, int cols, float scale, int offset, int seqQ, const float* slopes){\n"
                             "  int row=blockIdx.x; if(row>=rows) return;\n"
                             "  int qi=row%seqQ; int qabs=offset+qi; int lim=qabs; if(lim>=cols) lim=cols-1;\n"
                             "  double sl=(double)slopes[row/seqQ];\n"
                             "  extern __shared__ double sh[];\n"
                             "  int t=threadIdx.x, nt=blockDim.x;\n"
                             "  float* xr = x + (size_t)row*cols;\n"
                             "  double m=-1e300;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double v=(double)xr[j]*scale + sl*(double)(j-qabs); if(v>m)m=v; } }\n"
                             "  sh[t]=m; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s && sh[t+s]>sh[t]) sh[t]=sh[t+s]; __syncthreads(); }\n"
                             "  double rowmax=sh[0]; __syncthreads();\n"
                             "  double local=0.0;\n"
                             "  for(int j=t;j<cols;j+=nt){ if(j<=lim){ double v=(double)xr[j]*scale + sl*(double)(j-qabs); double e=exp(v-rowmax); xr[j]=(float)e; local+=e; } else { xr[j]=0.0f; } }\n"
                             "  sh[t]=local; __syncthreads();\n"
                             "  for(int s=nt/2;s>0;s>>=1){ if(t<s) sh[t]+=sh[t+s]; __syncthreads(); }\n"
                             "  double inv=1.0/sh[0];\n"
                             "  for(int j=t;j<=lim;j+=nt){ xr[j]=(float)(xr[j]*inv); }\n"
                             "}\n",
                             "attn_softmax_alibi.cu", "attn_softmax_alibi", &gAttnSoftmaxAlibi) != 0) { rc = -2; goto done; }
    {
        int threads = 256, blocks = rows; if (blocks < 1) blocks = 1;
        size_t shmem = (size_t)threads * sizeof(double);
        void* args[7];
        args[0] = &x; args[1] = &rows; args[2] = &cols; args[3] = &scale;
        args[4] = &offset; args[5] = &seqQ; args[6] = &slopes;
        rc = (cuLaunchKernel(gAttnSoftmaxAlibi, blocks, 1, 1, threads, 1, 1, shmem, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

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

// cu_rope_partial: PARTIAL rotary — rotates ONLY the first rotaryDim channels of each head
// (rotate_half within the rotaryDim block: pair (i, i+rotaryDim/2)), leaving [rotaryDim, hd)
// unchanged. GPT-NeoX/Phi/StableLM "partial rotary" (partial_rotary_factor<1). cu_rope_f32 is
// the rotaryDim==hd special case. inv = resident [rotaryDim/2] freq table (from RoPEFreqs on
// rotaryDim, so PI/YaRN scaling matches backend/ref). Angles in double.
int cu_rope_partial(void* x, const void* inv, int seq, int heads, int hd, int rotaryDim, int posOffset, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRopePartial && compile_kernel(
                      "extern \"C\" __global__ void rope_partial(float* x, const float* inv, int seq, int heads, int hd, int rotaryDim, int posOffset, double posDiv){\n"
                      "  int half = rotaryDim/2;\n"
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
                      "rope_partial.cu", "rope_partial", &gRopePartial) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (rotaryDim / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &x; args[1] = &inv; args[2] = &seq; args[3] = &heads;
        args[4] = &hd; args[5] = &rotaryDim; args[6] = &posOffset; args[7] = &posDiv;
        rc = (cuLaunchKernel(gRopePartial, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_rope_partial_band: PARTIAL strided-band RoPE — cu_rope_f32_band with only the first
// rotaryDim channels of each head rotated (half=rotaryDim/2), the rest passing through. The
// fused-QKV band path (§T613) for the partial-rotary architectures (GPT-NeoX/Phi/StableLM):
// rotate the q/k bands of a single [seq,stride] buffer in place, partial-rotary, no copy-out.
static CUfunction gRopePartialBand = NULL;
int cu_rope_partial_band(void* x, const void* inv, int seq, int stride, int off, int heads, int hd, int rotaryDim, int posOffset, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRopePartialBand && compile_kernel(
                      "extern \"C\" __global__ void rope_partial_band(float* x, const float* inv, int seq, int stride, int off, int heads, int hd, int rotaryDim, int posOffset, double posDiv){\n"
                      "  int half = rotaryDim/2;\n"
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
                      "rope_partial_band.cu", "rope_partial_band", &gRopePartialBand) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (rotaryDim / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        void* args[10];
        args[0] = &x; args[1] = &inv; args[2] = &seq; args[3] = &stride; args[4] = &off;
        args[5] = &heads; args[6] = &hd; args[7] = &rotaryDim; args[8] = &posOffset; args[9] = &posDiv;
        rc = (cuLaunchKernel(gRopePartialBand, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_qmatmul_q4k_pre: out[M,N] = a·dequant(W), W = Q4_K with PRE-DECODED sub-block scales
// (perf-notes-cuda.md R5). 192-byte blocks: a 64 B scale plane of 8 f32 pairs (c1[j]=d·sc6[j],
// c2[j]=dmin·min6[j]) + 128 B nibbles. Identical arithmetic to qmatmul_q4k but the per-block
// get_sm 6-bit unpack + f16 d/dmin decode + the two shfls are GONE — a lane reads its pair p's
// {c1[2p],c2[2p],c1[2p+1],c2[2p+1]} as one float4. Bit-exact vs qmatmul_q4k; trades +33% weight
// bytes for zero scale-decode ALU (measure-first A/B). K%256==0. DECODE-ONLY (GEMV).
int cu_qmatmul_q4k_pre(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemv4kPre && compile_kernel(
                       "extern \"C\" __global__ void qmatmul_q4k_pre(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*192;\n"
                       "  int p = lane >> 3, i0 = (lane & 7) * 4;\n"
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*192;\n"
                       "    float4 sm = *(const float4*)(blk + p*16);\n"                  // {c1[2p],c2[2p],c1[2p+1],c2[2p+1]}
                       "    unsigned int qw = *(const unsigned int*)(blk + 64 + lane*4);\n" // 4 qs bytes = 8 nibbles
                       "    int kb = w*256 + p*64 + i0;\n"
                       "    float4 al = *(const float4*)(ar + kb);\n"
                       "    float4 ah = *(const float4*)(ar + kb + 32);\n"
                       "    float sql = al.x*(float)(qw&0xFu) + al.y*(float)((qw>>8)&0xFu) + al.z*(float)((qw>>16)&0xFu) + al.w*(float)((qw>>24)&0xFu);\n"
                       "    float sqh = ah.x*(float)((qw>>4)&0xFu) + ah.y*(float)((qw>>12)&0xFu) + ah.z*(float)((qw>>20)&0xFu) + ah.w*(float)((qw>>28)&0xFu);\n"
                       "    float sal = (al.x+al.y)+(al.z+al.w), sah = (ah.x+ah.y)+(ah.z+ah.w);\n"
                       "    acc += sm.x*sql - sm.y*sal + sm.z*sqh - sm.w*sah;\n"           // c1[2p]*sql - c2[2p]*sal + c1[2p+1]*sqh - c2[2p+1]*sah
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_q4k_pre.cu", "qmatmul_q4k_pre", &gQgemv4kPre) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[7];
        args[0] = &dA; args[1] = &dQ; args[2] = &dOut;
        args[3] = &M; args[4] = &K; args[5] = &N; args[6] = &beta;
        rc = (cuLaunchKernel(gQgemv4kPre, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
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
int cu_qmatmul_q3k(const void* dA, const void* dMeta, const void* dQs, const void* dHm, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    // REPACKED (Tw72): the native 110-byte Q3_K block is not 4-aligned and its per-lane
    // aux/kmask scale splice is a redundant ~15-op decode per super-block. At upload we split
    // each super-block into meta (16 PRE-DECODED 6-bit scales + f16 d = 18 B) + qs (64 B,
    // aligned) + hmask (32 B, aligned), so the kernel reads a scale byte directly and pulls
    // qs/hmask as coalesced uints. Element mapping unchanged (lane owns 8 elems = half a 16-sub).
    if (!gQgemv3k && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_q3k(const float* a, const unsigned char* meta, const unsigned char* qsb, const unsigned char* hmb, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  int is = lane >> 1, half = lane & 1, l0 = half*8;\n"
                       "  int nb = is >> 3, jj = (is & 7) >> 1, gsel = is & 1, g = gsel*16;\n"
                       "  int shift = 2*jj, hbit = nb*4 + jj;\n"
                       "  int yi = nb*128 + jj*32 + g + l0;\n"
                       "  int qsrel = nb*32 + g + l0, hmrel = g + l0;\n"     // offsets within the 64B qs / 32B hmask regions (4-aligned)
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    size_t sb = (size_t)(n*sbs + w);\n"
                       "    const unsigned char* mp = meta + sb*18;\n"
                       "    float dl = f16f((unsigned short)(mp[16] | (mp[17]<<8))) * (float)((int)mp[is] - 32);\n"
                       "    const unsigned char* qp = qsb + sb*64 + qsrel;\n"
                       "    const unsigned char* hp = hmb + sb*32 + hmrel;\n"
                       "    unsigned qw0 = *(const unsigned int*)qp, qw1 = *(const unsigned int*)(qp+4);\n"
                       "    unsigned hw0 = *(const unsigned int*)hp, hw1 = *(const unsigned int*)(hp+4);\n"
                       "    const float* av = ar + (size_t)w*256 + yi;\n"
                       "    float4 alo = *(const float4*)av;\n"
                       "    float4 ahi = *(const float4*)(av + 4);\n"
                       "    float av8[8] = { alo.x, alo.y, alo.z, alo.w, ahi.x, ahi.y, ahi.z, ahi.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int t = 0; t < 8; t++){\n"
                       "      unsigned qb = (t<4) ? ((qw0>>(t*8))&0xFFu) : ((qw1>>((t-4)*8))&0xFFu);\n"
                       "      unsigned hb = (t<4) ? ((hw0>>(t*8))&0xFFu) : ((hw1>>((t-4)*8))&0xFFu);\n"
                       "      int lowb = (qb >> shift) & 3;\n"
                       "      int hset = (hb >> hbit) & 1;\n"
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
        void* args[9];
        args[0] = &dA; args[1] = &dMeta; args[2] = &dQs; args[3] = &dHm; args[4] = &dOut;
        args[5] = &M; args[6] = &K; args[7] = &N; args[8] = &beta;
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
// IQ4 codebook at file scope as __device__ const (L1-cached global) — an inside-kernel const
// array lands in per-thread local memory, and __constant__ serializes divergent per-lane indices.
#define IQ4_KVDECL "__device__ const float iq4kv[16] = {-127.f,-104.f,-83.f,-65.f,-49.f,-35.f,-22.f,-10.f,1.f,13.f,25.f,38.f,53.f,69.f,89.f,113.f};\n"

// cu_qmatmul_iq4nl: out[M,N] = a[M,K]·dequant(W), W = ggml IQ4_NL stored per OUTPUT row:
// K/32 blocks of 18 bytes (f16 d + 16 nibble bytes). Like Q4_0 (byte i → element i low nibble,
// i+16 high) but the nibble indexes a NONLINEAR 16-value codebook instead of (nibble−8):
// y = d·kvals[nibble]. Warp-per-output GEMV, lane l owns element l of each 32-block. 18-byte
// blocks not 4-aligned → d byte-read. 0.5625 B/weight. K%32==0.
int cu_qmatmul_iq4nl(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta) {
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
                       IQ4_KVDECL
                       "extern \"C\" __global__ void qmatmul_iq4nl(const float* a, const unsigned char* sc, const unsigned char* nb, float* out, int M, int K, int N, float beta){\n"
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
                       "      const unsigned char* sp = sc + (size_t)(n*nblk + b)*2;\n"
                       "      float d = f16f((unsigned short)(sp[0] | (sp[1]<<8)));\n"
                       "      unsigned qw = *(const unsigned int*)(nb + (size_t)(n*nblk + b)*16 + sub*4);\n"
                       "      const float* arb = ar + (size_t)b*32 + sub*4;\n"
                       "      float4 al = *(const float4*)arb;\n"
                       "      float4 ah = *(const float4*)(arb + 16);\n"
                       "      unsigned b0=qw&0xFFu, b1=(qw>>8)&0xFFu, b2=(qw>>16)&0xFFu, b3=(qw>>24)&0xFFu;\n"
                       "      float sl = al.x*iq4kv[b0&0xF] + al.y*iq4kv[b1&0xF] + al.z*iq4kv[b2&0xF] + al.w*iq4kv[b3&0xF];\n"
                       "      float sh = ah.x*iq4kv[b0>>4] + ah.y*iq4kv[b1>>4] + ah.z*iq4kv[b2>>4] + ah.w*iq4kv[b3>>4];\n"
                       "      acc += d * (sl + sh);\n"
                       "    }\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq4nl.cu", "qmatmul_iq4nl", &gQgemvI4nl) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dScale; args[2] = &dNib; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
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
                       IQ4_KVDECL
                       "extern \"C\" __global__ void qmatmul_iq4xs(const float* a, const unsigned char* q, float* out, int M, int K, int N, float beta){\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*136;\n"
                       "  int g = lane >> 2, sub = lane & 3;\n"     // sub-block (0..7), uint-in-sub-block (0..3)
                       "  float acc = 0.0f;\n"
                       "  #pragma unroll 2\n"
                       "  for (int w = 0; w < sbs; w++){\n"          // 136-byte super-block is 4-aligned → coalesced uint reads, no repack
                       "    const unsigned char* blk = qr + (size_t)w*136;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    unsigned sch = (unsigned)(blk[2] | (blk[3]<<8));\n"     // scales_h u16
                       "    const unsigned char* scl = blk + 4;\n"                 // scales_l[4]
                       "    int sl = (scl[g>>1] >> (4*(g&1))) & 0xF;\n"
                       "    int sh = (sch >> (2*g)) & 3;\n"
                       "    float ds = d * (float)((sl | (sh<<4)) - 32);\n"        // signed 6-bit sub-scale for this warp's sub-block g
                       "    unsigned qw = *(const unsigned int*)(blk + 8 + g*16 + sub*4);\n" // 4 nibble bytes, coalesced
                       "    const float* arb = ar + (size_t)w*256 + g*32 + sub*4;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 16);\n"
                       "    unsigned b0=qw&0xFFu, b1=(qw>>8)&0xFFu, b2=(qw>>16)&0xFFu, b3=(qw>>24)&0xFFu;\n"
                       "    float sl4 = al.x*iq4kv[b0&0xF] + al.y*iq4kv[b1&0xF] + al.z*iq4kv[b2&0xF] + al.w*iq4kv[b3&0xF];\n"
                       "    float sh4 = ah.x*iq4kv[b0>>4] + ah.y*iq4kv[b1>>4] + ah.z*iq4kv[b2>>4] + ah.w*iq4kv[b3>>4];\n"
                       "    acc += ds * (sl4 + sh4);\n"
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

// cu_qmatmul_iq2xxs: out[M,N] = a[M,K]·dequant(W), W = ggml IQ2_XXS (§T554), the first i-quant
// with a GRID codebook. K/256 super-blocks of 66 bytes: f16 d + 8 (qs0,qs1) u32 pairs; each pair
// decodes 32 values as 4 groups of 8 — qs0's 4 bytes index the 256×8 grid (device buffer dGrid,
// reconstructed host-side via gguf.Dequantize and shared across all IQ2_XXS weights), qs1 bits
// 7g..7g+6 index the 128-entry ksigns table (computed inline: ksigns[i] = i | (popcount(i)&1)<<7),
// bits 28..31 the 4-bit scale s (db = d·(0.5+s)·0.25). Warp-per-output GEMV: lane = (pair, group),
// 8 elements/lane. 66-byte blocks not 4-aligned → qs bytes assembled. K%256==0.
int cu_qmatmul_iq2xxs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI2xxs && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq2xxs(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ float sgrid[];\n"                                        // 256×8 grid in shared (Tw80)
                       "  for (int ii = threadIdx.x; ii < 2048; ii += blockDim.x) sgrid[ii] = grid[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*66;\n"
                       "  int p = lane >> 2, gg = lane & 3;\n"       // qs pair (0..7), group within pair (0..3)
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*66;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    const unsigned char* pp = blk + 2 + p*8;\n"                 // this lane's (qs0,qs1) pair
                       "    unsigned qs0 = pp[0]|(pp[1]<<8)|(pp[2]<<16)|(pp[3]<<24);\n"
                       "    unsigned qs1 = pp[4]|(pp[5]<<8)|(pp[6]<<16)|(pp[7]<<24);\n"
                       "    float db = d * (0.5f + (float)(qs1>>28)) * 0.25f;\n"
                       "    unsigned gi = (qs0 >> (8*gg)) & 0xFFu;\n"                    // grid index (0..255)
                       "    unsigned ksi = (qs1 >> (7*gg)) & 0x7Fu;\n"                   // ksigns index (0..127)
                       "    unsigned signs = ksi | ((unsigned)__popc(ksi) & 1u) << 7;\n" // ksigns[i] inline
                       "    const float* gv = sgrid + (size_t)gi*8;\n"
                       "    const float* arb = ar + (size_t)w*256 + p*32 + gg*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float av[8] = { al.x, al.y, al.z, al.w, ah.x, ah.y, ah.z, ah.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int k = 0; k < 8; k++){\n"
                       "      float g = ((signs >> k) & 1u) ? -gv[k] : gv[k];\n"
                       "      s += av[k] * g;\n"
                       "    }\n"
                       "    acc += db * s;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq2xxs.cu", "qmatmul_iq2xxs", &gQgemvI2xxs) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI2xxs, blocks, 1, 1, threads, 1, 1, 8192, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_iq3xxs: out[M,N] = a[M,K]·dequant(W), W = ggml IQ3_XXS (§T554), the 3.06-bit i-quant.
// K/256 super-blocks of 98 bytes: f16 d + 64 grid-index bytes + 8 uint32 sign/scale words. Each word
// decodes 32 values as 4 sub-groups of 8 — two grid bytes index the 256×4 grid (device dGrid,
// reconstructed host-side + shared across all IQ3_XXS weights), the word's 7j..7j+6 bits index the
// 128-entry ksigns table (computed inline), bits 28..31 the scale s (db = d·(0.5+s)·0.5). The 8th
// sign per group is the parity bit of the ksigns index. Warp-per-output GEMV: lane = (word g, group
// j), 8 elements/lane, r1 = av[0..3]·gv1, r2 = av[4..7]·gv2. 98-byte blocks not 4-aligned → bytes
// assembled. K%256==0.
int cu_qmatmul_iq3xxs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI3xxs && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq3xxs(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ float sgrid[];\n"                                        // 256×4 grid in shared (Tw80): the gather is the bottleneck, not the byte reads
                       "  for (int ii = threadIdx.x; ii < 1024; ii += blockDim.x) sgrid[ii] = grid[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*98;\n"
                       "  int g = lane >> 2, jj = lane & 3;\n"                          // word (0..7), sub-group (0..3)
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*98;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    const unsigned char* sp = blk + 66 + g*4;\n"                // this lane's sign/scale word
                       "    unsigned sg = sp[0]|(sp[1]<<8)|(sp[2]<<16)|(sp[3]<<24);\n"
                       "    float db = d * (0.5f + (float)(sg>>28)) * 0.5f;\n"
                       "    unsigned gb1 = blk[2 + g*8 + jj*2];\n"                       // two grid indices (256×4 rows)
                       "    unsigned gb2 = blk[2 + g*8 + jj*2 + 1];\n"
                       "    unsigned ksi = (sg >> (7*jj)) & 0x7Fu;\n"                    // ksigns index (0..127)
                       "    unsigned signs = ksi | ((unsigned)__popc(ksi) & 1u) << 7;\n" // ksigns[i] inline (8th = parity)
                       "    const float* gv1 = sgrid + (size_t)gb1*4;\n"
                       "    const float* gv2 = sgrid + (size_t)gb2*4;\n"
                       "    const float* arb = ar + (size_t)w*256 + g*32 + jj*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float av[8] = { al.x, al.y, al.z, al.w, ah.x, ah.y, ah.z, ah.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int k = 0; k < 4; k++){\n"
                       "      float g1 = ((signs >> k) & 1u) ? -gv1[k] : gv1[k];\n"
                       "      float g2 = ((signs >> (k+4)) & 1u) ? -gv2[k] : gv2[k];\n"
                       "      s += av[k]*g1 + av[k+4]*g2;\n"
                       "    }\n"
                       "    acc += db * s;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq3xxs.cu", "qmatmul_iq3xxs", &gQgemvI3xxs) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI3xxs, blocks, 1, 1, threads, 1, 1, 4096, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_iq3s: out[M,N] = a[M,K]·dequant(W), W = ggml IQ3_S (§T554), the 3.44-bit i-quant.
// K/256 super-blocks of 110 bytes: f16 d + 64 qs + 8 qh + 32 signs + 4 scale bytes. A 512×4 grid
// (device dGrid), 9-bit grid indices (qs byte + a qh high bit), DIRECT sign bytes (no ksigns), and
// explicit per-32 4-bit sub-scales (db = d·(1+2s)). Warp-per-output GEMV: lane = (group g, sub-group
// j), 8 elements/lane, r1 = av[0..3]·gv1, r2 = av[4..7]·gv2. K%256==0.
int cu_qmatmul_iq3s(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI3s && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq3s(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ float sgrid[];\n"                                        // 512×4 grid in shared (Tw80)
                       "  for (int ii = threadIdx.x; ii < 2048; ii += blockDim.x) sgrid[ii] = grid[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*110;\n"
                       "  int g = lane >> 2, jj = lane & 3;\n"                          // group (0..7), sub-group (0..3)
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*110;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    unsigned sc = (blk[106 + (g>>1)] >> (4*(g&1))) & 0x0Fu;\n"   // 4-bit sub-scale
                       "    float db = d * (1.0f + 2.0f*(float)sc);\n"
                       "    unsigned qh = blk[66 + g];\n"
                       "    unsigned i1 = g*8 + jj*2;\n"
                       "    unsigned u1 = blk[2+i1]     | ((qh >> (jj*2))   & 1u) << 8;\n" // 9-bit grid indices
                       "    unsigned u2 = blk[2+i1+1]   | ((qh >> (jj*2+1)) & 1u) << 8;\n"
                       "    unsigned sg = blk[74 + g*4 + jj];\n"                          // direct sign byte
                       "    const float* gv1 = sgrid + (size_t)u1*4;\n"
                       "    const float* gv2 = sgrid + (size_t)u2*4;\n"
                       "    const float* arb = ar + (size_t)w*256 + g*32 + jj*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float av[8] = { al.x, al.y, al.z, al.w, ah.x, ah.y, ah.z, ah.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int k = 0; k < 4; k++){\n"
                       "      float g1 = ((sg >> k) & 1u) ? -gv1[k] : gv1[k];\n"
                       "      float g2 = ((sg >> (k+4)) & 1u) ? -gv2[k] : gv2[k];\n"
                       "      s += av[k]*g1 + av[k+4]*g2;\n"
                       "    }\n"
                       "    acc += db * s;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq3s.cu", "qmatmul_iq3s", &gQgemvI3s) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI3s, blocks, 1, 1, threads, 1, 1, 8192, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_iq1s: out[M,N] = a[M,K]·dequant(W), W = ggml IQ1_S (§T554), the 1.56-bit TERNARY
// i-quant. K/256 super-blocks of 50 bytes: f16 d + 32 qs + 8 qh u16. A shared 2048×8 {−1,0,+1}
// grid (device dGrid) with a ±0.125 delta on every element and odd per-32 multipliers dl=d·(2s+1).
// Each qh u16 covers 32 elements: bits 0..11 = four 3-bit grid-index-high triples, bits 12..14 = s,
// bit 15 = delta sign. y[k]=dl·(grid[u][k]+δ), so out += dl·(Σ av·grid + δ·Σ av). Warp-per-output
// GEMV: lane = (qh index g, sub-group j), 8 elements/lane. K%256==0.
int cu_qmatmul_iq1s(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI1s && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq1s(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ unsigned short sgrid[];\n"                                    // 2048-uint16 packed ternary grid (4KB, Tw80 escape route)
                       "  const unsigned short* pg = (const unsigned short*)grid;\n"
                       "  for (int ii = threadIdx.x; ii < 2048; ii += blockDim.x) sgrid[ii] = pg[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*50;\n"
                       "  int g = lane >> 2, jj = lane & 3;\n"                          // qh index (0..7), sub-group (0..3)
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*50;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    unsigned qh = blk[34+g*2] | (blk[34+g*2+1]<<8);\n"
                       "    float dl = d * (float)(2u*((qh>>12)&7u) + 1u);\n"
                       "    float delta = (qh & 0x8000u) ? -0.125f : 0.125f;\n"
                       "    unsigned u = blk[2 + g*4 + jj] | (((qh >> (3*jj)) & 7u) << 8);\n" // 11-bit grid index
                       "    unsigned pk = sgrid[u];\n"                                          // 8 × 2-bit ternary codes from shared (2B read)
                       "    const float* arb = ar + (size_t)w*256 + g*32 + jj*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float sg = al.x*((float)(pk&3u)-1) + al.y*((float)((pk>>2)&3u)-1) + al.z*((float)((pk>>4)&3u)-1) + al.w*((float)((pk>>6)&3u)-1)\n"
                       "             + ah.x*((float)((pk>>8)&3u)-1) + ah.y*((float)((pk>>10)&3u)-1) + ah.z*((float)((pk>>12)&3u)-1) + ah.w*((float)((pk>>14)&3u)-1);\n"
                       "    float sa = (al.x+al.y)+(al.z+al.w) + (ah.x+ah.y)+(ah.z+ah.w);\n"
                       "    acc += dl * (sg + delta*sa);\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq1s.cu", "qmatmul_iq1s", &gQgemvI1s) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI1s, blocks, 1, 1, threads, 1, 1, 4096, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_qmatmul_iq1m: out[M,N] = a·dequant(W), W = ggml IQ1_M (§T554), the 1.75-bit ternary i-quant.
// K/256 super-blocks of 56 bytes: 32 qs + 16 qh + 8 scale bytes (4 u16). SAME 2048×8 grid + ±0.125
// delta as IQ1_S, but: (a) the f16 super-scale d is SPLIT across the top nibbles of the 4 scale words
// (dbits = w0>>12 | (w1>>12)<<4 | (w2>>12)<<8 | (w3>>12)<<12); (b) per-2-grid-row 3-bit sub-scales
// (dl=d·(2sc+1)); (c) each qh nibble carries a 3-bit grid-high triple + a delta sign. Warp-per-output
// GEMV: lane = grid-row i (0..31), 8 elements/lane. K%256==0.
int cu_qmatmul_iq1m(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI1m && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq1m(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ unsigned short sgrid[];\n"                                    // 2048-uint16 packed ternary grid (4KB, Tw80)
                       "  const unsigned short* pg = (const unsigned short*)grid;\n"
                       "  for (int ii = threadIdx.x; ii < 2048; ii += blockDim.x) sgrid[ii] = pg[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*56;\n"
                       "  int i = lane;\n"                                              // grid-row (0..31)
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*56;\n"
                       "    const unsigned char* sp = blk + 48;\n"
                       "    unsigned sw0=sp[0]|(sp[1]<<8), sw1=sp[2]|(sp[3]<<8), sw2=sp[4]|(sp[5]<<8), sw3=sp[6]|(sp[7]<<8);\n"
                       "    unsigned dbits = (sw0>>12) | ((sw1>>12)<<4) | ((sw2>>12)<<8) | ((sw3>>12)<<12);\n" // split f16 d
                       "    float d = f16f((unsigned short)dbits);\n"
                       "    unsigned swi = (i<8)?sw0:(i<16)?sw1:(i<24)?sw2:sw3;\n"
                       "    unsigned sc = (swi >> (3*((i>>1)&3))) & 7u;\n"              // per-2 3-bit sub-scale
                       "    float dl = d * (float)(2u*sc + 1u);\n"
                       "    unsigned nib = (blk[32 + (i>>1)] >> (4*(i&1))) & 0xFu;\n"    // grid-high triple + delta sign
                       "    unsigned u = blk[i] | ((nib & 7u) << 8);\n"
                       "    float delta = (nib & 8u) ? -0.125f : 0.125f;\n"
                       "    unsigned pk = sgrid[u];\n"                                          // 8 × 2-bit ternary codes from shared (2B read)
                       "    const float* arb = ar + (size_t)w*256 + i*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float sg = al.x*((float)(pk&3u)-1) + al.y*((float)((pk>>2)&3u)-1) + al.z*((float)((pk>>4)&3u)-1) + al.w*((float)((pk>>6)&3u)-1)\n"
                       "             + ah.x*((float)((pk>>8)&3u)-1) + ah.y*((float)((pk>>10)&3u)-1) + ah.z*((float)((pk>>12)&3u)-1) + ah.w*((float)((pk>>14)&3u)-1);\n"
                       "    float sa = (al.x+al.y)+(al.z+al.w) + (ah.x+ah.y)+(ah.z+ah.w);\n"
                       "    acc += dl * (sg + delta*sa);\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq1m.cu", "qmatmul_iq1m", &gQgemvI1m) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI1m, blocks, 1, 1, threads, 1, 1, 4096, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}


// cu_qmatmul_iq2xs: out[M,N] = a·dequant(W), W = ggml IQ2_XS (§T554) — the 2.31-bit sibling of
// IQ2_XXS with a 512-entry grid and EXPLICIT per-16-element 4-bit scales. K/256 super-blocks of
// 74 bytes: f16 d + 32 u16 qs (low 9 bits = grid index into the 512×8 dGrid, high 7 = ksigns
// index) + 8 bytes = 16 four-bit scales. Effective scale db = d·(0.5+sc4)·0.25. Warp-per-output:
// lane = qs word (0..31), each decodes 8 elements. K%256==0.
int cu_qmatmul_iq2xs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQgemvI2xs && compile_kernel(
                       "__device__ __forceinline__ float f16f(unsigned short h){\n"
                       "  unsigned s = (h & 0x8000u) << 16;\n"
                       "  float v = __uint_as_float((h & 0x7fffu) << 13) * __uint_as_float(0x77800000u);\n"
                       "  return __uint_as_float(__float_as_uint(v) | s);\n"
                       "}\n"
                       "extern \"C\" __global__ void qmatmul_iq2xs(const float* a, const unsigned char* q, const float* grid, float* out, int M, int K, int N, float beta){\n"
                       "  extern __shared__ float sgrid[];\n"                                        // 512×8 grid in shared (Tw80)
                       "  for (int ii = threadIdx.x; ii < 4096; ii += blockDim.x) sgrid[ii] = grid[ii];\n"
                       "  __syncthreads();\n"
                       "  long warp = ((long)blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"
                       "  int lane = threadIdx.x & 31;\n"
                       "  if (warp >= (long)M*N) return;\n"
                       "  int m = (int)(warp / N), n = (int)(warp % N);\n"
                       "  const float* ar = a + (size_t)m*K;\n"
                       "  int sbs = K >> 8;\n"
                       "  const unsigned char* qr = q + (size_t)n*sbs*74;\n"
                       "  float acc = 0.0f;\n"
                       "  for (int w = 0; w < sbs; w++){\n"
                       "    const unsigned char* blk = qr + (size_t)w*74;\n"
                       "    float d = f16f((unsigned short)(blk[0] | (blk[1]<<8)));\n"
                       "    const unsigned char* qp = blk + 2 + lane*2;\n"
                       "    unsigned qw = qp[0] | (qp[1]<<8);\n"                    // this lane's u16 qs word
                       "    unsigned gi = qw & 0x1FFu;\n"                           // 9-bit grid index (0..511)
                       "    unsigned ksi = qw >> 9;\n"                             // 7-bit ksigns index
                       "    int sub = lane >> 1;\n"                                // 16-element sub-block (0..15)
                       "    unsigned scb = blk[66 + (sub>>1)];\n"
                       "    unsigned sc4 = (scb >> (4*(sub&1))) & 0xFu;\n"
                       "    float db = d * (0.5f + (float)sc4) * 0.25f;\n"
                       "    unsigned signs = ksi | ((unsigned)__popc(ksi) & 1u) << 7;\n"
                       "    const float* gv = sgrid + (size_t)gi*8;\n"
                       "    const float* arb = ar + (size_t)w*256 + lane*8;\n"
                       "    float4 al = *(const float4*)arb;\n"
                       "    float4 ah = *(const float4*)(arb + 4);\n"
                       "    float av[8] = { al.x, al.y, al.z, al.w, ah.x, ah.y, ah.z, ah.w };\n"
                       "    float s = 0.0f;\n"
                       "    #pragma unroll\n"
                       "    for (int k = 0; k < 8; k++){\n"
                       "      float g = ((signs >> k) & 1u) ? -gv[k] : gv[k];\n"
                       "      s += av[k] * g;\n"
                       "    }\n"
                       "    acc += db * s;\n"
                       "  }\n"
                       "  for (int o = 16; o > 0; o >>= 1){ acc += __shfl_down_sync(0xffffffff, acc, o); }\n"
                       "  if (lane == 0){ out[warp] = beta*out[warp] + acc; }\n"
                       "}\n",
                       "qmatmul_iq2xs.cu", "qmatmul_iq2xs", &gQgemvI2xs) != 0) { rc = -2; goto done; }
    {
        long total = (long)M * N * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &dA; args[1] = &dQ; args[2] = &dGrid; args[3] = &dOut;
        args[4] = &M; args[5] = &K; args[6] = &N; args[7] = &beta;
        rc = (cuLaunchKernel(gQgemvI2xs, blocks, 1, 1, threads, 1, 1, 16384, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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

// cu_matmul_i8_mma: dC[M,N] (int32) = dA8[M,K] (int8 row-major) · dW8[K,N] (int8 row-major), via
// TILED tensor-core mma.sync.m16n8k32.s8 (inline PTX, no headers). One warp per 16×8 output tile,
// K looped in 32-wide MMA steps accumulating int32. The int8 tensor-core path llama.cpp's prefill
// lead rides on (cuBLAS underutilizes int8, Tw60). Correctness-first (global loads, no shared
// staging yet). M%16==0, N%8==0, K%32==0.
int cu_matmul_i8_mma(const void* dA8, const void* dW8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8Mma && compile_kernel(
            "extern \"C\" __global__ void i8mma(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  long tile = (long)blockIdx.x*blockDim.x + threadIdx.x; tile >>= 5;\n"  // warp id = output tile
            "  int lane = threadIdx.x & 31;\n"
            "  int ntiles = N >> 3;\n"
            "  int mt = (int)(tile / ntiles), nt = (int)(tile % ntiles);\n"
            "  if (mt*16 >= M) return;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int rowA = mt*16 + gid, rowA8 = mt*16 + gid + 8;\n"
            "  int colB = nt*8 + gid;\n"
            "  int acc0=0, acc1=0, acc2=0, acc3=0;\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    const signed char* a0 = A + (size_t)rowA*K + kt + c;\n"
            "    const signed char* a8 = A + (size_t)rowA8*K + kt + c;\n"
            "    unsigned ra0=(a0[0]&0xFF)|((a0[1]&0xFF)<<8)|((a0[2]&0xFF)<<16)|((a0[3]&0xFF)<<24);\n"
            "    unsigned ra1=(a8[0]&0xFF)|((a8[1]&0xFF)<<8)|((a8[2]&0xFF)<<16)|((a8[3]&0xFF)<<24);\n"
            "    unsigned ra2=(a0[16]&0xFF)|((a0[17]&0xFF)<<8)|((a0[18]&0xFF)<<16)|((a0[19]&0xFF)<<24);\n"
            "    unsigned ra3=(a8[16]&0xFF)|((a8[17]&0xFF)<<8)|((a8[18]&0xFF)<<16)|((a8[19]&0xFF)<<24);\n"
            "    const signed char* wp = W + (size_t)(kt+c)*N + colB;\n"     // W[(kt+c+j)*N + colB]
            "    unsigned rb0=(wp[0]&0xFF)|((wp[N]&0xFF)<<8)|((wp[2*N]&0xFF)<<16)|((wp[3*N]&0xFF)<<24);\n"
            "    const signed char* wp2 = W + (size_t)(kt+c+16)*N + colB;\n"
            "    unsigned rb1=(wp2[0]&0xFF)|((wp2[N]&0xFF)<<8)|((wp2[2*N]&0xFF)<<16)|((wp2[3*N]&0xFF)<<24);\n"
            "    asm volatile(\n"
            "      \"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \"\n"
            "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "      : \"+r\"(acc0),\"+r\"(acc1),\"+r\"(acc2),\"+r\"(acc3)\n"
            "      : \"r\"(ra0),\"r\"(ra1),\"r\"(ra2),\"r\"(ra3),\"r\"(rb0),\"r\"(rb1));\n"
            "  }\n"
            "  C[(size_t)rowA*N + nt*8 + tid*2] = acc0; C[(size_t)rowA*N + nt*8 + tid*2 + 1] = acc1;\n"
            "  C[(size_t)rowA8*N + nt*8 + tid*2] = acc2; C[(size_t)rowA8*N + nt*8 + tid*2 + 1] = acc3;\n"
            "}\n",
            "i8mma.cu", "i8mma", &gI8Mma) != 0) { rc = -2; goto done; }
    {
        long tiles = (long)(M/16) * (N/8);
        long total = tiles * 32;
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[6]; args[0] = &dA8; args[1] = &dW8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8Mma, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_t: SHARED-TILED int8 mma GEMM. Thread block (256 threads = 8 warps) computes a
// 16x64 output tile; warp w owns the 16x8 sub-tile at cols [w*8,+8). Per K-step (BK=32) the block
// cooperatively stages A[16x32] + W[32x64] into shared (coalesced), so all 8 warps read their mma
// fragments from shared — killing the naive kernel's redundant global loads. M%16==0,N%64==0,K%32==0.
int cu_matmul_i8_mma_t(const void* dA8, const void* dW8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaT && compile_kernel(
            "extern \"C\" __global__ void i8mmat(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ signed char sA[16*32];\n"
            "  __shared__ signed char sW[32*64];\n"
            "  int t = threadIdx.x;\n"                              // 0..255
            "  int warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"                                // 64-col blocks
            "  int bM = blockIdx.x / nblk, bN = blockIdx.x % nblk;\n"
            "  int rowBase = bM*16, colBase = bN*64;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int acc0=0, acc1=0, acc2=0, acc3=0;\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    sA[t & 511] = A[(size_t)(rowBase + ((t)>>5))*K + kt + (t&31)];\n"      // t<512? no: 256 threads, 512 elems → 2 loads
            "    sA[(t+256) & 511] = A[(size_t)(rowBase + ((t+256)>>5))*K + kt + ((t+256)&31)];\n"
            "    #pragma unroll\n"
            "    for (int i = 0; i < 8; i++){\n"                    // 2048 sW elems / 256 = 8
            "      int e = t + i*256; int kk = e >> 6, nn = e & 63;\n"
            "      sW[e] = W[(size_t)(kt + kk)*N + colBase + nn];\n"
            "    }\n"
            "    __syncthreads();\n"
            "    unsigned ra0=(sA[gid*32+c]&0xFF)|((sA[gid*32+c+1]&0xFF)<<8)|((sA[gid*32+c+2]&0xFF)<<16)|((sA[gid*32+c+3]&0xFF)<<24);\n"
            "    unsigned ra1=(sA[(gid+8)*32+c]&0xFF)|((sA[(gid+8)*32+c+1]&0xFF)<<8)|((sA[(gid+8)*32+c+2]&0xFF)<<16)|((sA[(gid+8)*32+c+3]&0xFF)<<24);\n"
            "    unsigned ra2=(sA[gid*32+c+16]&0xFF)|((sA[gid*32+c+17]&0xFF)<<8)|((sA[gid*32+c+18]&0xFF)<<16)|((sA[gid*32+c+19]&0xFF)<<24);\n"
            "    unsigned ra3=(sA[(gid+8)*32+c+16]&0xFF)|((sA[(gid+8)*32+c+17]&0xFF)<<8)|((sA[(gid+8)*32+c+18]&0xFF)<<16)|((sA[(gid+8)*32+c+19]&0xFF)<<24);\n"
            "    int col = warp*8 + gid;\n"                          // this warp's N-tile column within the 64-wide sW
            "    unsigned rb0=(sW[c*64+col]&0xFF)|((sW[(c+1)*64+col]&0xFF)<<8)|((sW[(c+2)*64+col]&0xFF)<<16)|((sW[(c+3)*64+col]&0xFF)<<24);\n"
            "    unsigned rb1=(sW[(c+16)*64+col]&0xFF)|((sW[(c+17)*64+col]&0xFF)<<8)|((sW[(c+18)*64+col]&0xFF)<<16)|((sW[(c+19)*64+col]&0xFF)<<24);\n"
            "    asm volatile(\n"
            "      \"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 \"\n"
            "      \"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "      : \"+r\"(acc0),\"+r\"(acc1),\"+r\"(acc2),\"+r\"(acc3)\n"
            "      : \"r\"(ra0),\"r\"(ra1),\"r\"(ra2),\"r\"(ra3),\"r\"(rb0),\"r\"(rb1));\n"
            "    __syncthreads();\n"
            "  }\n"
            "  int oc = colBase + warp*8 + tid*2;\n"
            "  C[(size_t)(rowBase+gid)*N + oc] = acc0; C[(size_t)(rowBase+gid)*N + oc + 1] = acc1;\n"
            "  C[(size_t)(rowBase+gid+8)*N + oc] = acc2; C[(size_t)(rowBase+gid+8)*N + oc + 1] = acc3;\n"
            "}\n",
            "i8mmat.cu", "i8mmat", &gI8MmaT) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/16) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dW8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaT, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_rb: REGISTER-BLOCKED int8 mma GEMM. 256-thread block (8 warps, 2x4 grid)
// computes a 64x64 output tile; each warp owns a 32x16 sub-tile = 2 M-tiles x 2 N-tiles = 4 MMAs.
// Per K-step (BK=32) the block stages A[64x32]+W[32x64] into shared, then each warp loads 2 A-frags
// + 2 B-frags and issues 4 mma.sync from them (2x register reuse per fragment → higher arithmetic
// intensity than the 16x64 tiled kernel). M%64==0, N%64==0, K%32==0.
int cu_matmul_i8_mma_rb(const void* dA8, const void* dW8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaRb && compile_kernel(
            "extern \"C\" __global__ void i8mmarb(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ signed char sA[64*32];\n"
            "  __shared__ signed char sW[32*64];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"                   // 2x4 warp grid
            "  int Mb = wr*32, Nb = wc*16;\n"                          // this warp's 32x16 region
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int acc[4][4]; for(int i=0;i<4;i++){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0;}\n"  // [mi*2+ni][0..3]
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    #pragma unroll\n"
            "    for (int i=0;i<8;i++){ int e=t+i*256; sA[e]=A[(size_t)(rowBase+(e>>5))*K + kt + (e&31)]; }\n"   // 2048 elems
            "    #pragma unroll\n"
            "    for (int i=0;i<8;i++){ int e=t+i*256; sW[e]=W[(size_t)(kt+(e>>6))*N + colBase + (e&63)]; }\n"
            "    __syncthreads();\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int r0=(Mb+mi*16+gid)*32, r1=(Mb+mi*16+gid+8)*32;\n"
            "      ra[mi][0]=(sA[r0+c]&0xFF)|((sA[r0+c+1]&0xFF)<<8)|((sA[r0+c+2]&0xFF)<<16)|((sA[r0+c+3]&0xFF)<<24);\n"
            "      ra[mi][1]=(sA[r1+c]&0xFF)|((sA[r1+c+1]&0xFF)<<8)|((sA[r1+c+2]&0xFF)<<16)|((sA[r1+c+3]&0xFF)<<24);\n"
            "      ra[mi][2]=(sA[r0+c+16]&0xFF)|((sA[r0+c+17]&0xFF)<<8)|((sA[r0+c+18]&0xFF)<<16)|((sA[r0+c+19]&0xFF)<<24);\n"
            "      ra[mi][3]=(sA[r1+c+16]&0xFF)|((sA[r1+c+17]&0xFF)<<8)|((sA[r1+c+18]&0xFF)<<16)|((sA[r1+c+19]&0xFF)<<24);\n"
            "    }\n"
            "    unsigned rb[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int col=Nb+ni*8+gid;\n"
            "      rb[ni][0]=(sW[c*64+col]&0xFF)|((sW[(c+1)*64+col]&0xFF)<<8)|((sW[(c+2)*64+col]&0xFF)<<16)|((sW[(c+3)*64+col]&0xFF)<<24);\n"
            "      rb[ni][1]=(sW[(c+16)*64+col]&0xFF)|((sW[(c+17)*64+col]&0xFF)<<8)|((sW[(c+18)*64+col]&0xFF)<<16)|((sW[(c+19)*64+col]&0xFF)<<24);\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "      }\n"
            "    __syncthreads();\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=d[0]; C[(size_t)r0*N+oc+1]=d[1];\n"
            "      C[(size_t)r1*N+oc]=d[2]; C[(size_t)r1*N+oc+1]=d[3];\n"
            "    }\n"
            "}\n",
            "i8mmarb.cu", "i8mmarb", &gI8MmaRb) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dW8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaRb, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_db: register-blocked int8 GEMM (same 64x64 tile / 4-MMA-per-warp compute as _rb)
// with DOUBLE-BUFFERED cp.async shared staging — the next K-step's A/W tiles are prefetched
// (cp.async.cg.shared.global, 16B/thread) while the current step's MMAs run, hiding shared-fill
// latency behind compute. M%64==0, N%64==0, K%32==0.
int cu_matmul_i8_mma_db(const void* dA8, const void* dW8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaDb && compile_kernel(
            "__device__ __forceinline__ void ld16(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ void loadk(signed char* sA, signed char* sW, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K, int N){\n"
            "  if (t < 128){ int off=t*16; size_t g=(size_t)(rowBase+(t>>1))*K + kt + ((t&1)*16); ld16(sA+off, A+g); }\n"
            "  else { int ci=t-128, off=ci*16; size_t g=(size_t)(kt+(ci>>2))*N + colBase + ((ci&3)*16); ld16(sW+off, W+g); }\n"
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmadb(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*32];\n"
            "  __shared__ __align__(16) signed char sW[2][32*64];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int acc[4][4]; for(int i=0;i<4;i++){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0;}\n"
            "  int buf = 0;\n"
            "  loadk(sA[0], sW[0], A, W, t, rowBase, colBase, 0, K, N);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    if (kt + 32 < K){ loadk(sA[buf^1], sW[buf^1], A, W, t, rowBase, colBase, kt+32, K, N); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pW = sW[buf];\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int r0=(Mb+mi*16+gid)*32, r1=(Mb+mi*16+gid+8)*32;\n"
            "      ra[mi][0]=(pA[r0+c]&0xFF)|((pA[r0+c+1]&0xFF)<<8)|((pA[r0+c+2]&0xFF)<<16)|((pA[r0+c+3]&0xFF)<<24);\n"
            "      ra[mi][1]=(pA[r1+c]&0xFF)|((pA[r1+c+1]&0xFF)<<8)|((pA[r1+c+2]&0xFF)<<16)|((pA[r1+c+3]&0xFF)<<24);\n"
            "      ra[mi][2]=(pA[r0+c+16]&0xFF)|((pA[r0+c+17]&0xFF)<<8)|((pA[r0+c+18]&0xFF)<<16)|((pA[r0+c+19]&0xFF)<<24);\n"
            "      ra[mi][3]=(pA[r1+c+16]&0xFF)|((pA[r1+c+17]&0xFF)<<8)|((pA[r1+c+18]&0xFF)<<16)|((pA[r1+c+19]&0xFF)<<24);\n"
            "    }\n"
            "    unsigned rb[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int col=Nb+ni*8+gid;\n"
            "      rb[ni][0]=(pW[c*64+col]&0xFF)|((pW[(c+1)*64+col]&0xFF)<<8)|((pW[(c+2)*64+col]&0xFF)<<16)|((pW[(c+3)*64+col]&0xFF)<<24);\n"
            "      rb[ni][1]=(pW[(c+16)*64+col]&0xFF)|((pW[(c+17)*64+col]&0xFF)<<8)|((pW[(c+18)*64+col]&0xFF)<<16)|((pW[(c+19)*64+col]&0xFF)<<24);\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=d[0]; C[(size_t)r0*N+oc+1]=d[1];\n"
            "      C[(size_t)r1*N+oc]=d[2]; C[(size_t)r1*N+oc+1]=d[3];\n"
            "    }\n"
            "}\n",
            "i8mmadb.cu", "i8mmadb", &gI8MmaDb) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dW8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaDb, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_wt: like _db (64x64, cp.async double-buffered) but W is [N][K] row-major (the
// NATIVE GGUF weight layout: out-features x in-features, blocks along K). Storing W transposed in
// shared (sWt[N][K]) makes the B-fragment reads CONTIGUOUS single-int32 loads — killing the ~16-way
// bank conflict of the [K][N] column-gather in _db. C[M][N] = A[M][K] * W[N][K]^T. M%64,N%64,K%32==0.
int cu_matmul_i8_mma_wt(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaWt && compile_kernel(
            "__device__ __forceinline__ void ld16w(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ void loadkw(signed char* sA, signed char* sWt, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K){\n"
            "  if (t < 128){ int off=t*16; size_t g=(size_t)(rowBase+(t>>1))*K + kt + ((t&1)*16); ld16w(sA+off, A+g); }\n"
            "  else { int ci=t-128, off=ci*16; size_t g=(size_t)(colBase+(ci>>1))*K + kt + ((ci&1)*16); ld16w(sWt+off, W+g); }\n"  // W[N][K] row-major
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmawt(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*32];\n"
            "  __shared__ __align__(16) signed char sWt[2][64*32];\n"                 // [N=64][K=32] transposed
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int acc[4][4]; for(int i=0;i<4;i++){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0;}\n"
            "  int buf = 0;\n"
            "  loadkw(sA[0], sWt[0], A, W, t, rowBase, colBase, 0, K);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    if (kt + 32 < K){ loadkw(sA[buf^1], sWt[buf^1], A, W, t, rowBase, colBase, kt+32, K); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pWt = sWt[buf];\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int r0=(Mb+mi*16+gid)*32, r1=(Mb+mi*16+gid+8)*32;\n"
            "      ra[mi][0]=*(const int*)(pA+r0+c);   ra[mi][1]=*(const int*)(pA+r1+c);\n"
            "      ra[mi][2]=*(const int*)(pA+r0+c+16); ra[mi][3]=*(const int*)(pA+r1+c+16);\n"
            "    }\n"
            "    unsigned rb[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int nr=(Nb+ni*8+gid)*32;\n"                 // B col -> a contiguous sWt row
            "      rb[ni][0]=*(const int*)(pWt+nr+c); rb[ni][1]=*(const int*)(pWt+nr+c+16);\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=d[0]; C[(size_t)r0*N+oc+1]=d[1];\n"
            "      C[(size_t)r1*N+oc]=d[2]; C[(size_t)r1*N+oc+1]=d[3];\n"
            "    }\n"
            "}\n",
            "i8mmawt.cu", "i8mmawt", &gI8MmaWt) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dWt8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaWt, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_wp: _wt with the shared rows PADDED to 36-byte stride (32 data + 4 pad) so the
// per-row word offset (9 words) makes gid*9 mod 32 a near-permutation — dissolving _wt's residual
// 2-way A/B bank conflict (stride-32 aliased gid vs gid+4). Same [N][K] weight contract. M%64,N%64,K%32==0.
int cu_matmul_i8_mma_wp(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaWp && compile_kernel(
            "__device__ __forceinline__ void ld16p(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ void loadkp(signed char* sA, signed char* sWt, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K){\n"
            "  if (t < 128){ int r=t>>1, h=t&1; size_t g=(size_t)(rowBase+r)*K + kt + h*16; ld16p(sA + r*48 + h*16, A+g); }\n"
            "  else { int ci=t-128, r=ci>>1, h=ci&1; size_t g=(size_t)(colBase+r)*K + kt + h*16; ld16p(sWt + r*48 + h*16, W+g); }\n"
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmawp(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*48];\n"
            "  __shared__ __align__(16) signed char sWt[2][64*48];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  int acc[4][4]; for(int i=0;i<4;i++){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0;}\n"
            "  int buf = 0;\n"
            "  loadkp(sA[0], sWt[0], A, W, t, rowBase, colBase, 0, K);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    if (kt + 32 < K){ loadkp(sA[buf^1], sWt[buf^1], A, W, t, rowBase, colBase, kt+32, K); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pWt = sWt[buf];\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int r0=(Mb+mi*16+gid)*48, r1=(Mb+mi*16+gid+8)*48;\n"
            "      ra[mi][0]=*(const int*)(pA+r0+c);   ra[mi][1]=*(const int*)(pA+r1+c);\n"
            "      ra[mi][2]=*(const int*)(pA+r0+c+16); ra[mi][3]=*(const int*)(pA+r1+c+16);\n"
            "    }\n"
            "    unsigned rb[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int nr=(Nb+ni*8+gid)*48;\n"
            "      rb[ni][0]=*(const int*)(pWt+nr+c); rb[ni][1]=*(const int*)(pWt+nr+c+16);\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=d[0]; C[(size_t)r0*N+oc+1]=d[1];\n"
            "      C[(size_t)r1*N+oc]=d[2]; C[(size_t)r1*N+oc+1]=d[3];\n"
            "    }\n"
            "}\n",
            "i8mmawp.cu", "i8mmawp", &gI8MmaWp) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dWt8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaWp, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mma_lm: the _wp GEMM with LDMATRIX fragment loads. Same 64x64 / cp.async double-buffer
// / [N][K] / 48-pad scaffold, but the A fragment is loaded via ldmatrix.x4 (4 quadrants) and B via
// ldmatrix.x2 (2 K-halves) instead of the manual 4-int-load + byte-assembly — one hardware-swizzled
// load per fragment. Layouts proven by cu_ldmatrix_probe{,2}. M%64,N%64,K%32==0.
int cu_matmul_i8_mma_lm(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmaLm && compile_kernel(
            "__device__ __forceinline__ void ld16l(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ unsigned smem32(const void* p){ unsigned a; asm(\"{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }\" : \"=r\"(a) : \"l\"(p)); return a; }\n"
            "__device__ __forceinline__ void loadkl(signed char* sA, signed char* sWt, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K){\n"
            "  if (t < 128){ int r=t>>1, h=t&1; size_t g=(size_t)(rowBase+r)*K + kt + h*16; ld16l(sA + r*48 + h*16, A+g); }\n"
            "  else { int ci=t-128, r=ci>>1, h=ci&1; size_t g=(size_t)(colBase+r)*K + kt + h*16; ld16l(sWt + r*48 + h*16, W+g); }\n"
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmalm(const signed char* A, const signed char* W, int* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*48];\n"
            "  __shared__ __align__(16) signed char sWt[2][64*48];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3;\n"
            "  int acc[4][4]; for(int i=0;i<4;i++){acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0;}\n"
            "  int buf = 0;\n"
            "  loadkl(sA[0], sWt[0], A, W, t, rowBase, colBase, 0, K);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    if (kt + 32 < K){ loadkl(sA[buf^1], sWt[buf^1], A, W, t, rowBase, colBase, kt+32, K); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pWt = sWt[buf];\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int q=lane>>3, rit=(q&1)*8+(lane&7), ks=(q>>1)*16;\n"
            "      unsigned ad=smem32(pA + (size_t)(Mb+mi*16+rit)*48 + ks);\n"
            "      asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\" : \"=r\"(ra[mi][0]),\"=r\"(ra[mi][1]),\"=r\"(ra[mi][2]),\"=r\"(ra[mi][3]) : \"r\"(ad));\n"
            "    }\n"
            "    unsigned rb[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int q=(lane>>3)&1, ncol=Nb+ni*8+(lane&7), ks=q*16;\n"
            "      unsigned ad=smem32(pWt + (size_t)ncol*48 + ks);\n"
            "      asm volatile(\"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\" : \"=r\"(rb[ni][0]),\"=r\"(rb[ni][1]) : \"r\"(ad));\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int* d=acc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=d[0]; C[(size_t)r0*N+oc+1]=d[1];\n"
            "      C[(size_t)r1*N+oc]=d[2]; C[(size_t)r1*N+oc+1]=d[3];\n"
            "    }\n"
            "}\n",
            "i8mmalm.cu", "i8mmalm", &gI8MmaLm) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[6]; args[0] = &dA8; args[1] = &dWt8; args[2] = &dC32; args[3] = &M; args[4] = &K; args[5] = &N;
        rc = (cuLaunchKernel(gI8MmaLm, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mmq: TRUE per-block MMQ on the _wp GEMM core. Weights W8[N][K] int8 with per-32-K
// block scale wSc[N][K/32]; activations A8[M][K] int8 with per-block scale aSc[M][K/32]. BK=32 == one
// scale block, so each K-step does the int8 MMA into an int32 block-partial then accumulates
// d*aSc[blk]*wSc[blk] into an f32 accumulator. Output C[M][N] f32 = dequantized product.
// This is the GEMM that replaces dequant->f16->cuBLAS in prefill. M%64,N%64,K%32==0.
int cu_matmul_i8_mmq(const void* dA8, const void* dWt8, const void* daSc, const void* dwSc, void* dCf, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8Mmq && compile_kernel(
            "__device__ __forceinline__ void ld16m(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ void loadkm(signed char* sA, signed char* sWt, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K){\n"
            "  if (t < 128){ int r=t>>1, h=t&1; size_t g=(size_t)(rowBase+r)*K + kt + h*16; ld16m(sA + r*48 + h*16, A+g); }\n"
            "  else { int ci=t-128, r=ci>>1, h=ci&1; size_t g=(size_t)(colBase+r)*K + kt + h*16; ld16m(sWt + r*48 + h*16, W+g); }\n"
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmq(const signed char* A, const signed char* W, const float* aSc, const float* wSc, float* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*48];\n"
            "  __shared__ __align__(16) signed char sWt[2][64*48];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6, nb = K >> 5;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  float facc[4][4]; for(int i=0;i<4;i++){facc[i][0]=facc[i][1]=facc[i][2]=facc[i][3]=0.f;}\n"
            "  int buf = 0;\n"
            "  loadkm(sA[0], sWt[0], A, W, t, rowBase, colBase, 0, K);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    int bk = kt >> 5;\n"
            "    if (kt + 32 < K){ loadkm(sA[buf^1], sWt[buf^1], A, W, t, rowBase, colBase, kt+32, K); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pWt = sWt[buf];\n"
            "    unsigned ra[2][4]; float as[2][2];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int r0=(Mb+mi*16+gid)*48, r1=(Mb+mi*16+gid+8)*48;\n"
            "      ra[mi][0]=*(const int*)(pA+r0+c);   ra[mi][1]=*(const int*)(pA+r1+c);\n"
            "      ra[mi][2]=*(const int*)(pA+r0+c+16); ra[mi][3]=*(const int*)(pA+r1+c+16);\n"
            "      as[mi][0]=aSc[(size_t)(rowBase+Mb+mi*16+gid)*nb + bk]; as[mi][1]=aSc[(size_t)(rowBase+Mb+mi*16+gid+8)*nb + bk];\n"
            "    }\n"
            "    unsigned rb[2][2]; float ws[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int nr=(Nb+ni*8+gid)*48; int oc=colBase+Nb+ni*8+tid*2;\n"
            "      rb[ni][0]=*(const int*)(pWt+nr+c); rb[ni][1]=*(const int*)(pWt+nr+c+16);\n"
            "      ws[ni][0]=wSc[(size_t)oc*nb + bk]; ws[ni][1]=wSc[(size_t)(oc+1)*nb + bk];\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int d[4]; d[0]=d[1]=d[2]=d[3]=0;\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "        float* f=facc[mi*2+ni];\n"
            "        f[0]+=(float)d[0]*as[mi][0]*ws[ni][0]; f[1]+=(float)d[1]*as[mi][0]*ws[ni][1];\n"
            "        f[2]+=(float)d[2]*as[mi][1]*ws[ni][0]; f[3]+=(float)d[3]*as[mi][1]*ws[ni][1];\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++)\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ float* f=facc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=f[0]; C[(size_t)r0*N+oc+1]=f[1];\n"
            "      C[(size_t)r1*N+oc]=f[2]; C[(size_t)r1*N+oc+1]=f[3];\n"
            "    }\n"
            "}\n",
            "i8mmq.cu", "i8mmq", &gI8Mmq) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[8]; args[0] = &dA8; args[1] = &dWt8; args[2] = &daSc; args[3] = &dwSc; args[4] = &dCf; args[5] = &M; args[6] = &K; args[7] = &N;
        rc = (cuLaunchKernel(gI8Mmq, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_matmul_i8_mmq_r: MMQ with PER-ROW activation scale (dynamic per-token) — aSc[M], not [M][K/32].
// aScale[m] is constant over K so it hoists OUT of the K-loop: the inner accumulate is just
// facc += (float)d * wSc[block] (1 mul), and aScale[m] multiplies once in the epilogue. Halves the
// inner scale work + drops half the scale-loads vs _mmq. Weights stay per-32-block. M%64,N%64,K%32==0.
int cu_matmul_i8_mmq_r(const void* dA8, const void* dWt8, const void* daSc, const void* dwSc, void* dCf, int M, int K, int N) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gI8MmqR && compile_kernel(
            "__device__ __forceinline__ void ld16mr(signed char* sdst, const signed char* gsrc){\n"
            "  asm volatile(\"{ .reg .u64 s; cvta.to.shared.u64 s, %0; cp.async.cg.shared.global [s], [%1], 16; }\" :: \"l\"((void*)sdst), \"l\"((const void*)gsrc));\n"
            "}\n"
            "__device__ __forceinline__ unsigned smem32mr(const void* p){ unsigned a; asm(\"{ .reg .u64 s; cvta.to.shared.u64 s, %1; cvt.u32.u64 %0, s; }\" : \"=r\"(a) : \"l\"(p)); return a; }\n"
            "__device__ __forceinline__ void loadkmr(signed char* sA, signed char* sWt, const signed char* A, const signed char* W, int t, int rowBase, int colBase, int kt, int K){\n"
            "  if (t < 128){ int r=t>>1, h=t&1; size_t g=(size_t)(rowBase+r)*K + kt + h*16; ld16mr(sA + r*48 + h*16, A+g); }\n"
            "  else { int ci=t-128, r=ci>>1, h=ci&1; size_t g=(size_t)(colBase+r)*K + kt + h*16; ld16mr(sWt + r*48 + h*16, W+g); }\n"
            "  asm volatile(\"cp.async.commit_group;\");\n"
            "}\n"
            "extern \"C\" __global__ void i8mmqr(const signed char* A, const signed char* W, const float* aSc, const float* wSc, float* C, int M, int K, int N){\n"
            "  __shared__ __align__(16) signed char sA[2][64*48];\n"
            "  __shared__ __align__(16) signed char sWt[2][64*48];\n"
            "  int t = threadIdx.x, warp = t >> 5, lane = t & 31;\n"
            "  int nblk = N >> 6, nb = K >> 5;\n"
            "  int rowBase = (blockIdx.x / nblk)*64, colBase = (blockIdx.x % nblk)*64;\n"
            "  int wr = warp >> 2, wc = warp & 3;\n"
            "  int Mb = wr*32, Nb = wc*16;\n"
            "  int gid = lane >> 2, tid = lane & 3, c = tid*4;\n"
            "  float facc[4][4]; for(int i=0;i<4;i++){facc[i][0]=facc[i][1]=facc[i][2]=facc[i][3]=0.f;}\n"
            "  int buf = 0;\n"
            "  loadkmr(sA[0], sWt[0], A, W, t, rowBase, colBase, 0, K);\n"
            "  for (int kt = 0; kt < K; kt += 32){\n"
            "    int bk = kt >> 5;\n"
            "    if (kt + 32 < K){ loadkmr(sA[buf^1], sWt[buf^1], A, W, t, rowBase, colBase, kt+32, K); asm volatile(\"cp.async.wait_group 1;\"); }\n"
            "    else { asm volatile(\"cp.async.wait_group 0;\"); }\n"
            "    __syncthreads();\n"
            "    signed char* pA = sA[buf]; signed char* pWt = sWt[buf];\n"
            "    unsigned ra[2][4];\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++){ int q=lane>>3, rit=(q&1)*8+(lane&7), ks=(q>>1)*16;\n"
            "      unsigned ad=smem32mr(pA + (size_t)(Mb+mi*16+rit)*48 + ks);\n"
            "      asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\" : \"=r\"(ra[mi][0]),\"=r\"(ra[mi][1]),\"=r\"(ra[mi][2]),\"=r\"(ra[mi][3]) : \"r\"(ad));\n"
            "    }\n"
            "    unsigned rb[2][2]; float ws[2][2];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ int qn=(lane>>3)&1, ncol=Nb+ni*8+(lane&7), ksn=qn*16; int oc=colBase+Nb+ni*8+tid*2;\n"
            "      unsigned adn=smem32mr(pWt + (size_t)ncol*48 + ksn);\n"
            "      asm volatile(\"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\" : \"=r\"(rb[ni][0]),\"=r\"(rb[ni][1]) : \"r\"(adn));\n"
            "      ws[ni][0]=wSc[(size_t)oc*nb + bk]; ws[ni][1]=wSc[(size_t)(oc+1)*nb + bk];\n"
            "    }\n"
            "    #pragma unroll\n"
            "    for(int mi=0;mi<2;mi++)\n"
            "      #pragma unroll\n"
            "      for(int ni=0;ni<2;ni++){ int d[4]; d[0]=d[1]=d[2]=d[3]=0;\n"
            "        asm volatile(\"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\"\n"
            "          : \"+r\"(d[0]),\"+r\"(d[1]),\"+r\"(d[2]),\"+r\"(d[3])\n"
            "          : \"r\"(ra[mi][0]),\"r\"(ra[mi][1]),\"r\"(ra[mi][2]),\"r\"(ra[mi][3]),\"r\"(rb[ni][0]),\"r\"(rb[ni][1]));\n"
            "        float* f=facc[mi*2+ni];\n"
            "        f[0]+=(float)d[0]*ws[ni][0]; f[1]+=(float)d[1]*ws[ni][1];\n"
            "        f[2]+=(float)d[2]*ws[ni][0]; f[3]+=(float)d[3]*ws[ni][1];\n"
            "      }\n"
            "    __syncthreads();\n"
            "    buf ^= 1;\n"
            "  }\n"
            "  #pragma unroll\n"
            "  for(int mi=0;mi<2;mi++){\n"
            "    float a0=aSc[rowBase+Mb+mi*16+gid], a1=aSc[rowBase+Mb+mi*16+gid+8];\n"
            "    #pragma unroll\n"
            "    for(int ni=0;ni<2;ni++){ float* f=facc[mi*2+ni];\n"
            "      int oc=colBase+Nb+ni*8+tid*2; int r0=rowBase+Mb+mi*16+gid, r1=r0+8;\n"
            "      C[(size_t)r0*N+oc]=a0*f[0]; C[(size_t)r0*N+oc+1]=a0*f[1];\n"
            "      C[(size_t)r1*N+oc]=a1*f[2]; C[(size_t)r1*N+oc+1]=a1*f[3];\n"
            "    }\n"
            "  }\n"
            "}\n",
            "i8mmqr.cu", "i8mmqr", &gI8MmqR) != 0) { rc = -2; goto done; }
    {
        int nblk = N >> 6;
        long blocks = (long)(M/64) * nblk;
        void* args[8]; args[0] = &dA8; args[1] = &dWt8; args[2] = &daSc; args[3] = &dwSc; args[4] = &dCf; args[5] = &M; args[6] = &K; args[7] = &N;
        rc = (cuLaunchKernel(gI8MmqR, (unsigned)blocks, 1, 1, 256, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_quant_rows_i8: device-side dynamic per-ROW int8 quantization of f32 activations for MMQ. One
// warp per row: warp-reduce max|.| -> scale=amax/127 -> symmetric round-to-int8. Produces A8[M][K]
// int8 + aSc[M] f32, exactly the inputs cu_matmul_i8_mmq_r expects. This is how prefill activations
// (produced on-GPU by the prior layer) enter the int8 path without a host round-trip.
int cu_quant_rows_i8(const void* dAf, void* dA8, void* daSc, int M, int K) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gQrowsI8 && compile_kernel(
            "extern \"C\" __global__ void qrows_i8(const float* A, signed char* A8, float* aSc, int M, int K){\n"
            "  int gw = (blockIdx.x*blockDim.x + threadIdx.x) >> 5;\n"   // global warp = row
            "  int lane = threadIdx.x & 31;\n"
            "  if (gw >= M) return;\n"
            "  const float* row = A + (size_t)gw*K;\n"
            "  float amax = 0.f;\n"
            "  for (int j = lane; j < K; j += 32){ float v = fabsf(row[j]); amax = fmaxf(amax, v); }\n"
            "  #pragma unroll\n"
            "  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));\n"
            "  float scale = amax / 127.f;\n"
            "  float inv = scale > 0.f ? 1.f/scale : 0.f;\n"
            "  if (lane == 0) aSc[gw] = scale;\n"
            "  signed char* o8 = A8 + (size_t)gw*K;\n"
            "  for (int j = lane; j < K; j += 32){\n"
            "    int q = __float2int_rn(row[j]*inv);\n"
            "    q = q > 127 ? 127 : (q < -128 ? -128 : q);\n"
            "    o8[j] = (signed char)q;\n"
            "  }\n"
            "}\n",
            "qrows_i8.cu", "qrows_i8", &gQrowsI8) != 0) { rc = -2; goto done; }
    {
        int threads = 256;                          // 8 warps/block
        long blocks = ((long)M*32 + threads - 1) / threads;
        void* args[5]; args[0] = &dAf; args[1] = &dA8; args[2] = &daSc; args[3] = &M; args[4] = &K;
        rc = (cuLaunchKernel(gQrowsI8, (unsigned)blocks, 1, 1, (unsigned)threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_ldmatrix_probe: empirically map the ldmatrix.x4.b16 fragment layout (no codebase precedent).
// One warp loads four 8x8 b16 matrices (matrix m at sh[m*64], element (m,r,c) = (m<<6)|(r<<3)|c),
// lane t provides row (t%8) of matrix (t/8), then dumps its 4 result regs. Decoding out[t*4+k]'s two
// b16 halves tells us which (m,r,c) lands in (lane t, reg k, half) — the layout the int8 mma.s8 A/B
// fragment must match to load via ldmatrix. dOut = 128 u32.
int cu_ldmatrix_probe(void* dOut) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gLdmProbe && compile_kernel(
            "extern \"C\" __global__ void ldmprobe(unsigned* out){\n"
            "  __shared__ unsigned short sh[256];\n"
            "  int t = threadIdx.x;\n"
            "  #pragma unroll\n"
            "  for(int i=0;i<8;i++){ int idx=t*8+i; int m=idx>>6, rc=idx&63, r=rc>>3, c=rc&7; sh[idx]=(unsigned short)((m<<6)|(r<<3)|c); }\n"
            "  __syncthreads();\n"
            "  int matrix=t>>3, row=t&7;\n"
            "  unsigned short* rp = &sh[matrix*64 + row*8];\n"
            "  unsigned saddr;\n"
            "  asm volatile(\"{ .reg .u64 s64; cvta.to.shared.u64 s64, %1; cvt.u32.u64 %0, s64; }\" : \"=r\"(saddr) : \"l\"(rp));\n"
            "  unsigned r0,r1,r2,r3;\n"
            "  asm volatile(\"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\" : \"=r\"(r0),\"=r\"(r1),\"=r\"(r2),\"=r\"(r3) : \"r\"(saddr));\n"
            "  out[t*4+0]=r0; out[t*4+1]=r1; out[t*4+2]=r2; out[t*4+3]=r3;\n"
            "}\n",
            "ldmprobe.cu", "ldmprobe", &gLdmProbe) != 0) { rc = -2; goto done; }
    {
        void* args[1]; args[0] = &dOut;
        rc = (cuLaunchKernel(gLdmProbe, 1, 1, 1, 32, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
    pthread_mutex_unlock(&gLock);
    return rc;
}

// cu_ldmatrix_probe2: map ldmatrix.x2.b16 for the B fragment. Two 8x8 b16 matrices (matrix m at
// sh[m*64], element (m,r,c)=(m<<6)|(r<<3)|c); for x2 lanes 0-15 provide addresses (lane L → matrix
// L/8 row L%8). Dumps each lane's 2 regs. This confirms whether ldmatrix.x2 on the [N][K] weight
// gives the mma.s8 B fragment (rb0=N-row gid/K0-15, rb1=N-row gid/K16-31) with NO transpose. dOut=64 u32.
int cu_ldmatrix_probe2(void* dOut) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gLdmProbe2 && compile_kernel(
            "extern \"C\" __global__ void ldmprobe2(unsigned* out){\n"
            "  __shared__ unsigned short sh[128];\n"                  // 2 matrices x 64
            "  int t = threadIdx.x;\n"
            "  #pragma unroll\n"
            "  for(int i=0;i<4;i++){ int idx=t*4+i; int m=idx>>6, rc=idx&63, r=rc>>3, c=rc&7; sh[idx]=(unsigned short)((m<<6)|(r<<3)|c); }\n"
            "  __syncthreads();\n"
            "  int matrix=t>>3, row=t&7;\n"                            // lanes 0-15 address the 2 matrices' rows
            "  unsigned short* rp = &sh[(matrix&1)*64 + row*8];\n"
            "  unsigned saddr;\n"
            "  asm volatile(\"{ .reg .u64 s64; cvta.to.shared.u64 s64, %1; cvt.u32.u64 %0, s64; }\" : \"=r\"(saddr) : \"l\"(rp));\n"
            "  unsigned r0,r1;\n"
            "  asm volatile(\"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\" : \"=r\"(r0),\"=r\"(r1) : \"r\"(saddr));\n"
            "  out[t*2+0]=r0; out[t*2+1]=r1;\n"
            "}\n",
            "ldmprobe2.cu", "ldmprobe2", &gLdmProbe2) != 0) { rc = -2; goto done; }
    {
        void* args[1]; args[0] = &dOut;
        rc = (cuLaunchKernel(gLdmProbe2, 1, 1, 1, 32, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
    }
done:
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

// cu_rope_partial_dpos: device-position twin of cu_rope_partial — PARTIAL rotary (first
// rotaryDim channels of each head) reading posOffset from device int *dPos, so a captured
// decode graph rotates the partial-rotary architectures at the token's true position.
int cu_rope_partial_dpos(void* x, const void* inv, int seq, int heads, int hd, int rotaryDim, const void* dPos, double posDiv) {
    int rc = -1;
    pthread_mutex_lock(&gLock);
    if (ensure_init() != 0) { rc = -1; goto done; }
    if (cuCtxSetCurrent(gCtx) != CUDA_SUCCESS) { rc = -8; goto done; }
    if (!gRopePartialDpos && compile_kernel(
                          "extern \"C\" __global__ void rope_partial_dpos(float* x, const float* inv, int seq, int heads, int hd, int rotaryDim, const int* dPos, double posDiv){\n"
                          "  int half = rotaryDim/2;\n"
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
                          "rope_partial_dpos.cu", "rope_partial_dpos", &gRopePartialDpos) != 0) { rc = -2; goto done; }
    {
        long total = (long)seq * heads * (rotaryDim / 2);
        int threads = 256, blocks = (int)((total + threads - 1) / threads); if (blocks < 1) blocks = 1;
        void* args[8];
        args[0] = &x; args[1] = &inv; args[2] = &seq; args[3] = &heads; args[4] = &hd; args[5] = &rotaryDim; args[6] = &dPos; args[7] = &posDiv;
        rc = (cuLaunchKernel(gRopePartialDpos, blocks, 1, 1, threads, 1, 1, 0, (CUstream)gStream, args, NULL) == CUDA_SUCCESS) ? 0 : -3;
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
                              "  float* shk = shq + group * hd; // [16*hp] K tile (Tw84 half-tile → 2 blocks/SM)\n"
                              "  float* shv = shk + 16 * hp;    // [16*hp] V tile\n"
                              "  for (int i = t; i < group * hd; i += nt) shq[i] = q[(size_t)kvh * group * hd + i];\n"
                              "  float NEGINF = __int_as_float(0xff800000);\n"
                              "  float m = NEGINF, l = 0.f, a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;\n"
                              "  for (int base = begin; base < end; base += 16) {\n"
                              "    int nk = end - base; if (nk > 16) nk = 16;\n"
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
        size_t shmem = ((size_t)group * hd + 2u * 16u * (hd + 1)) * sizeof(float);
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
                                 "  float f;\n"
                                 "  asm(\"cvt.f32.f16 %0, %1;\" : \"=f\"(f) : \"h\"(h));\n"
                                 "  return f;\n"
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
                                 "  unsigned short* shk = (unsigned short*)(shq + group * hd);\n"
                                 "  unsigned short* shv = shk + 32 * hp;\n"
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
                                 "        const unsigned short* kr = shk + lane * hp;\n"
                                 "        const float* qh = shq + w * hd;\n"
                                 "        float dot = 0.f;\n"
                                 "        for (int d = 0; d < hd; d++) dot += qh[d] * h2f(kr[d]);\n"
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
                                 "        const unsigned short* vr = shv + j * hp;\n"
                                 "        if (lane < hd)      a0 += pj * h2f(vr[lane]);\n"
                                 "        if (lane + 32 < hd) a1 += pj * h2f(vr[lane + 32]);\n"
                                 "        if (lane + 64 < hd) a2 += pj * h2f(vr[lane + 64]);\n"
                                 "        if (lane + 96 < hd) a3 += pj * h2f(vr[lane + 96]);\n"
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
        size_t shmem = (size_t)group * hd * sizeof(float) + 2u * 32u * (hd + 1) * sizeof(unsigned short);
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
