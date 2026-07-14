//go:build darwin && cgo

// Objective-C side of the metal backend (§T20). ARC-compiled (see cgo CFLAGS).
// One shared device+queue; shared-storage buffers come from a process-wide
// size-classed pool (§T336, Apple Silicon UMA makes them host-visible;
// newBufferWithBytesNoCopy zero-copy is a later opt).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <pthread.h>

#include "metal_bridge.h"

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;

static int ensure_init(void) {
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (dev != nil && MPSSupportsMTLDevice(dev)) {
            gDevice = dev;
            gQueue = [dev newCommandQueue];
        }
    });
    return (gDevice != nil && gQueue != nil) ? 0 : -1;
}

int mtl_available(void) { return ensure_init() == 0 ? 1 : 0; }

// ---- process-wide buffer pool (§T336, twin of the Vulkan pool §T335) --------
// Every op previously created (and ARC-released) a fresh MTLBuffer per operand
// per call — a device allocation each time. Buffers are now pooled: capacities
// round up to a power of two and are reused best-fit across calls (training/
// inference loops repeat the same shapes, so the pool converges after the first
// step). Pooled reuse is safe with stale contents because no kernel relies on
// newBufferWithLength's zero-fill: atomic accumulators are either uploaded
// pre-zeroed via pool_in or explicitly memset before dispatch, and every other
// output element is written before the memcpy back (MPS matmul runs beta:0.0).
// gLock serializes ops (OP_BEGIN below); the __attribute__((cleanup)) guard
// releases the call's pooled buffers and unlocks on EVERY function exit path.
static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;

#define MTL_POOL_MAX 128
static id<MTLBuffer> gPool[MTL_POOL_MAX];
static int    gPoolUsed[MTL_POOL_MAX];
static int    gNPool = 0;
static size_t gPoolBytes = 0;
// Beyond this the pool stops growing and oversize requests fall back to a
// transient per-call buffer (old behavior) — bounds device memory held idle.
static const size_t kPoolByteCap = (size_t)512 << 20;

static size_t pool_round(size_t n) {
    size_t c = 256;
    while (c < n) c <<= 1;
    return c;
}

// pool_buf hands out an idle pooled buffer with length >= len (best fit),
// growing the pool on miss; beyond the table/byte cap it falls back to a
// transient buffer that ARC releases at the op's autoreleasepool exit.
static id<MTLBuffer> pool_buf(size_t len) {
    int best = -1;
    for (int i = 0; i < gNPool; ++i) {
        if (gPoolUsed[i] || gPool[i].length < len) continue;
        if (best < 0 || gPool[i].length < gPool[best].length) best = i;
    }
    if (best >= 0) {
        gPoolUsed[best] = 1;
        return gPool[best];
    }
    size_t cap = pool_round(len);
    if (gNPool < MTL_POOL_MAX && gPoolBytes + cap <= kPoolByteCap) {
        id<MTLBuffer> b = [gDevice newBufferWithLength:cap options:MTLResourceStorageModeShared];
        if (b == nil) return nil;
        gPool[gNPool] = b;
        gPoolUsed[gNPool] = 1;
        gNPool++;
        gPoolBytes += cap;
        return b;
    }
    return [gDevice newBufferWithLength:len options:MTLResourceStorageModeShared];
}

// pool_in is pool_buf plus the host→buffer upload copy (replaces newBufferWithBytes).
static id<MTLBuffer> pool_in(const void* src, size_t len) {
    id<MTLBuffer> b = pool_buf(len);
    if (b != nil) memcpy(b.contents, src, len);
    return b;
}

static void op_end(char* unused) {
    (void)unused;
    for (int i = 0; i < gNPool; ++i) gPoolUsed[i] = 0;
    pthread_mutex_unlock(&gLock);
}

// OP_BEGIN serializes the op and arms the scope guard: on ANY exit (early error
// return or success) the call's pooled buffers become reusable and gLock drops.
// The op is synchronous (waitUntilCompleted before returning), so nothing in
// flight references pool buffers once the guard runs.
#define OP_BEGIN \
    pthread_mutex_lock(&gLock); \
    __attribute__((cleanup(op_end))) char _opGuard = 0; \
    (void)_opGuard

// Q8_0 quantized matmul: one thread per output (mi,ni) streams the ni-th weight row's
// K/32 blocks, dequantizing each in-kernel (f16 scale · int8) — no materialized f32 weight.
// The f16 scale is reassembled from two little-endian bytes and bitcast to half, so the read
// is alignment-safe regardless of the 34-byte block stride.
// RMSNorm over the last axis (§R29): y = x/√(mean(x²)+eps)·γ, no mean-sub, no bias.
// COOPERATIVE (§T345): ONE threadgroup of 256 threads per row. The lanes stride the
// row with coalesced loads (lane t reads t, t+256, … — consecutive lanes hit
// consecutive addresses, unlike the old one-thread-per-row kernel whose neighbours
// were `dim` floats apart), accumulate a partial mean-of-squares, then a tree
// reduction in threadgroup memory combines them; all lanes re-read the total and
// write their strided output slice (coalesced). Reduction order differs from the
// f64 reference → §V11 dim-scaled f32 tolerance (unchanged). P = {rows, dim}, E = {eps}.
static NSString* const kRMSNormSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void rmsnorm(device const float* X [[buffer(0)]],\n"
     "                    device const float* G [[buffer(1)]],\n"
     "                    device float* O [[buffer(2)]],\n"
     "                    constant int* P [[buffer(3)]],\n"
     "                    constant float* E [[buffer(4)]],\n"
     "                    uint row [[threadgroup_position_in_grid]],\n"
     "                    uint lane [[thread_position_in_threadgroup]],\n"
     "                    uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], dim=P[1];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*dim;\n"
     "  threadgroup float red[256];\n"
     "  float part=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ float v=X[base+j]; part+=v*v; }\n"
     "  red[lane]=part;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float inv=rsqrt(red[0]/float(dim)+E[0]);\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ O[base+j]=X[base+j]*inv*G[j]; }\n"
     "}\n";

static id<MTLComputePipelineState> gRMSNorm = nil;

static int ensure_rmsnorm(void) {
    if (gRMSNorm != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRMSNormSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"rmsnorm"];
    if (fn == nil) return -6;
    gRMSNorm = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRMSNorm != nil ? 0 : -6;
}

int mtl_rmsnorm_f32(const float* X, const float* G, float* O, int rows, int dim, float eps) {
    if (ensure_init() != 0) return -1;
    if (ensure_rmsnorm() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)rows * dim * sizeof(float);
        size_t gLen = (size_t)dim * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> gb = pool_in(G, gLen);
        id<MTLBuffer> ob = pool_buf(xLen);
        int P[2] = {rows, dim};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float E[1] = {eps};
        id<MTLBuffer> eb = pool_in(E, sizeof(E));
        if (xb == nil || gb == nil || ob == nil || pb == nil || eb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRMSNorm];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        [enc setBuffer:eb offset:0 atIndex:4];
        // Cooperative (§T345): one 256-thread threadgroup per row (red[256] fixes the size).
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, xLen);
        return 0;
    }
}

// RoPE (§R28/§R92; split-half rotate_half) on Q[seq,width], width=heads·hd. One thread
// per rotated pair (seq·heads·(hd/2) threads): pair i of head h at row p rotates
// (Q[base+i], Q[base+i+half]) by pos·inv[i], pos=(posOffset+p)/posDiv. inv[] and posDiv
// are host-precomputed (PI/YaRN folded in). P={seq,width,heads,hd,half,posOffset}, F={posDiv}.
static NSString* const kRoPESource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void rope(device const float* Q [[buffer(0)]],\n"
     "                 device const float* INV [[buffer(1)]],\n"
     "                 device float* O [[buffer(2)]],\n"
     "                 constant int* P [[buffer(3)]],\n"
     "                 constant float* F [[buffer(4)]],\n"
     "                 uint gid [[thread_position_in_grid]]) {\n"
     "  int seq=P[0], width=P[1], heads=P[2], hd=P[3], halfd=P[4], posOffset=P[5];\n"
     "  int hh=heads*halfd; int total=seq*hh;\n"
     "  if ((int)gid >= total) return;\n"
     "  int p=(int)gid/hh; int rem=(int)gid%hh; int h=rem/halfd; int i=rem%halfd;\n"
     "  int base=p*width + h*hd + i;\n"
     "  float pos=float(p+posOffset)/F[0];\n"
     "  float theta=INV[i];\n"
     "  float c=cos(pos*theta), s=sin(pos*theta);\n"
     "  float qi=Q[base], qih=Q[base+halfd];\n"
     "  O[base]=qi*c-qih*s;\n"
     "  O[base+halfd]=qih*c+qi*s;\n"
     "}\n";

static id<MTLComputePipelineState> gRoPE = nil;

static int ensure_rope(void) {
    if (gRoPE != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRoPESource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"rope"];
    if (fn == nil) return -6;
    gRoPE = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRoPE != nil ? 0 : -6;
}

// rope2 (SPEC T613): ONE dispatch rotates BOTH bands of a fused QKV row in place — the q band
// (headsQ heads at element offset offQ) and the k band (headsK heads at offset offK), each row
// `stride` floats wide. Thread t covers pair i of virtual head h over seq rows: h < headsQ is a
// q-band head, the rest map to the k band. Replaces two back-to-back rope dispatches per layer.
// P={seq,stride,headsQ,offQ,headsK,offK,hd,half,posOffset}, F={posDiv}.
static NSString* const kRoPE2Source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void rope2(device float* Q [[buffer(0)]],\n"
     "                  device const float* INV [[buffer(1)]],\n"
     "                  constant int* P [[buffer(2)]],\n"
     "                  constant float* F [[buffer(3)]],\n"
     "                  uint gid [[thread_position_in_grid]]) {\n"
     "  int seq=P[0], stride=P[1], headsQ=P[2], offQ=P[3], headsK=P[4], offK=P[5];\n"
     "  int hd=P[6], halfd=P[7], posOffset=P[8];\n"
     "  int hh=(headsQ+headsK)*halfd; int total=seq*hh;\n"
     "  if ((int)gid >= total) return;\n"
     "  int p=(int)gid/hh; int rem=(int)gid%hh; int h=rem/halfd; int i=rem%halfd;\n"
     "  int base = (h < headsQ) ? (offQ + p*stride + h*hd + i)\n"
     "                          : (offK + p*stride + (h-headsQ)*hd + i);\n"
     "  float pos=float(p+posOffset)/F[0];\n"
     "  float theta=INV[i];\n"
     "  float c=cos(pos*theta), s=sin(pos*theta);\n"
     "  float qi=Q[base], qih=Q[base+halfd];\n"
     "  Q[base]=qi*c-qih*s;\n"
     "  Q[base+halfd]=qih*c+qi*s;\n"
     "}\n";

static id<MTLComputePipelineState> gRoPE2 = nil;

static int ensure_rope2(void) {
    if (gRoPE2 != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRoPE2Source options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"rope2"];
    if (fn == nil) return -6;
    gRoPE2 = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRoPE2 != nil ? 0 : -6;
}

// mtl_recorder_rope2 encodes the fused two-band in-place rotation (see kRoPE2Source) into the
// recorder's command buffer. No commit.
int mtl_recorder_rope2(void* rec, void* qh, void* invh,
                       int seq, int stride, int headsQ, int offQ, int headsK, int offK,
                       int hd, int half, int posOffset, float posDiv) {
    if (rec == NULL || qh == NULL || invh == NULL) return -2;
    if (ensure_rope2() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> qb = (__bridge id<MTLBuffer>)qh;
    id<MTLBuffer> ib = (__bridge id<MTLBuffer>)invh;
    int P[9] = {seq, stride, headsQ, offQ, headsK, offK, hd, half, posOffset};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    float F[1] = {posDiv};
    id<MTLBuffer> fb = [gDevice newBufferWithBytes:F length:sizeof(F) options:MTLResourceStorageModeShared];
    if (pb == nil || fb == nil) return -2;
    int total = seq * (headsQ + headsK) * half;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gRoPE2];
    [enc setBuffer:qb offset:0 atIndex:0];
    [enc setBuffer:ib offset:0 atIndex:1];
    [enc setBuffer:pb offset:0 atIndex:2];
    [enc setBuffer:fb offset:0 atIndex:3];
    NSUInteger tg = gRoPE2.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)total < tg) tg = (NSUInteger)total;
    [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

int mtl_rope_f32(const float* Q, const float* inv, float* O,
                 int seq, int width, int heads, int hd, int half, int posOffset, float posDiv) {
    if (ensure_init() != 0) return -1;
    if (ensure_rope() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t qLen = (size_t)seq * width * sizeof(float);
        size_t iLen = (size_t)half * sizeof(float);

        id<MTLBuffer> qb = pool_in(Q, qLen);
        id<MTLBuffer> ib = pool_in(inv, iLen);
        id<MTLBuffer> ob = pool_buf(qLen);
        int P[6] = {seq, width, heads, hd, half, posOffset};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float F[1] = {posDiv};
        id<MTLBuffer> fb = pool_in(F, sizeof(F));
        if (qb == nil || ib == nil || ob == nil || pb == nil || fb == nil) return -2;

        int total = seq * heads * half;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRoPE];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:ib offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        [enc setBuffer:fb offset:0 atIndex:4];
        NSUInteger tg = gRoPE.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, qLen);
        return 0;
    }
}

// Numerically-stable softmax over the last axis (§V12): y = exp(x−max)/Σexp(x−max).
// COOPERATIVE (§T345/§T346): ONE 256-thread threadgroup per row, coalesced strided
// access. Two tree reductions (row max, then Σexp) share the red[256] scratch with a
// barrier between phases. P = {rows, dim}.
static NSString* const kSoftmaxSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void softmax(device const float* X [[buffer(0)]],\n"
     "                    device float* O [[buffer(1)]],\n"
     "                    constant int* P [[buffer(2)]],\n"
     "                    uint row [[threadgroup_position_in_grid]],\n"
     "                    uint lane [[thread_position_in_threadgroup]],\n"
     "                    uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], dim=P[1];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*dim;\n"
     "  threadgroup float red[256];\n"
     "  float pm=-INFINITY;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ pm=max(pm,X[base+j]); }\n"
     "  red[lane]=pm;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) red[lane]=max(red[lane],red[lane+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float m=red[0];\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  float ps=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ ps+=exp(X[base+j]-m); }\n"
     "  red[lane]=ps;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float inv=1.0f/red[0];\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ O[base+j]=exp(X[base+j]-m)*inv; }\n"
     "}\n";

static id<MTLComputePipelineState> gSoftmax = nil;

static int ensure_softmax(void) {
    if (gSoftmax != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSoftmaxSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"softmax"];
    if (fn == nil) return -6;
    gSoftmax = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gSoftmax != nil ? 0 : -6;
}

int mtl_softmax_f32(const float* X, float* O, int rows, int dim) {
    if (ensure_init() != 0) return -1;
    if (ensure_softmax() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)rows * dim * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, len);
        id<MTLBuffer> ob = pool_buf(len);
        int P[2] = {rows, dim};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gSoftmax];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:ob offset:0 atIndex:1];
        [enc setBuffer:pb offset:0 atIndex:2];
        // Cooperative (§T346): one 256-thread threadgroup per row (red[256] fixes the size).
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, len);
        return 0;
    }
}

// crossentropy forward (§T401): per row b, lseᵢ = max + log Σ exp(z−max); Lᵢ = lseᵢ − z[targetᵢ].
// One 256-thread threadgroup per row (cooperative, §T346). §T401 measured the CPU-fallback forward
// at 20ms — ~2/3 of the GPT forward — a silent ref-fallback (metal had the backward but not this).
// Basic case only (label_smoothing=0, z_loss=0); the Go wrapper falls back to ref otherwise.
static NSString* const kCrossEntropySource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void crossentropy(device const float* Z [[buffer(0)]],\n"
     "                         device const float* T [[buffer(1)]],\n"
     "                         device float* L [[buffer(2)]],\n"
     "                         constant int* P [[buffer(3)]],\n"
     "                         uint row [[threadgroup_position_in_grid]],\n"
     "                         uint lane [[thread_position_in_threadgroup]],\n"
     "                         uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int b=P[0], c=P[1];\n"
     "  if ((int)row >= b) return;\n"
     "  int base=(int)row*c;\n"
     "  threadgroup float red[256];\n"
     "  float m=-INFINITY;\n"
     "  for (int j=(int)lane;j<c;j+=(int)tgsize) m=max(m,Z[base+j]);\n"
     "  red[lane]=m; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2;s>0;s>>=1){ if(lane<s) red[lane]=max(red[lane],red[lane+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  m=red[0];\n"
     "  float sum=0.0f;\n"
     "  for (int j=(int)lane;j<c;j+=(int)tgsize) sum+=exp(Z[base+j]-m);\n"
     "  red[lane]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2;s>0;s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  if (lane==0){ float lse=m+log(red[0]); int ti=(int)T[row]; L[row]=lse - Z[base+ti]; }\n"
     "}\n";

static id<MTLComputePipelineState> gCrossEntropy = nil;
static int ensure_crossentropy(void) {
    if (gCrossEntropy != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kCrossEntropySource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"crossentropy"];
    if (fn == nil) return -11;
    gCrossEntropy = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gCrossEntropy != nil ? 0 : -12;
}

// mtl_crossentropy_f32 writes the per-row loss L[b] = lse − z[target]; the caller means over b.
int mtl_crossentropy_f32(const float* Z, const float* T, float* L, int b, int c) {
    if (ensure_init() != 0) return -1;
    if (ensure_crossentropy() != 0) return -6;
    @autoreleasepool {
        id<MTLBuffer> zb = [gDevice newBufferWithBytes:Z length:(size_t)b*c*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> tb = [gDevice newBufferWithBytes:T length:(size_t)b*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> lb = [gDevice newBufferWithLength:(size_t)b*sizeof(float) options:MTLResourceStorageModeShared];
        int P[2] = {b, c};
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
        if (zb == nil || tb == nil || lb == nil || pb == nil) return -2;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gCrossEntropy];
        [enc setBuffer:zb offset:0 atIndex:0];
        [enc setBuffer:tb offset:0 atIndex:1];
        [enc setBuffer:lb offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(b,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
        memcpy(L, lb.contents, (size_t)b*sizeof(float));
        return 0;
    }
}

// embed forward (§T402): out[i,:] = table[idx[i],:] — a row gather. One thread per output element.
// The last silent CPU fallback in the metal GPT training step (§T401 audit); small (~1ms) but this
// makes the metal training fully fallback-free.
static NSString* const kEmbedSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void embed(device const float* T [[buffer(0)]],\n"
     "                  device const float* IDX [[buffer(1)]],\n"
     "                  device float* O [[buffer(2)]],\n"
     "                  constant int* P [[buffer(3)]],\n"
     "                  uint gid [[thread_position_in_grid]]) {\n"
     "  int m=P[0], d=P[1];\n"
     "  int total=m*d;\n"
     "  if ((int)gid >= total) return;\n"
     "  int i=(int)gid/d, j=(int)gid%d;\n"
     "  int row=(int)IDX[i];\n"
     "  O[gid]=T[row*d+j];\n"
     "}\n";

static id<MTLComputePipelineState> gEmbed = nil;
static int ensure_embed(void) {
    if (gEmbed != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kEmbedSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"embed"];
    if (fn == nil) return -11;
    gEmbed = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gEmbed != nil ? 0 : -12;
}

int mtl_embed_f32(const float* Table, const float* Idx, float* O, int n, int m, int d) {
    if (ensure_init() != 0) return -1;
    if (ensure_embed() != 0) return -6;
    @autoreleasepool {
        id<MTLBuffer> tb = [gDevice newBufferWithBytes:Table length:(size_t)n*d*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ib = [gDevice newBufferWithBytes:Idx length:(size_t)m*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [gDevice newBufferWithLength:(size_t)m*d*sizeof(float) options:MTLResourceStorageModeShared];
        int P[2] = {m, d};
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
        if (tb == nil || ib == nil || ob == nil || pb == nil) return -2;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gEmbed];
        [enc setBuffer:tb offset:0 atIndex:0];
        [enc setBuffer:ib offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = m*d;
        NSUInteger tg = gEmbed.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
        memcpy(O, ob.contents, (size_t)m*d*sizeof(float));
        return 0;
    }
}

// LayerNorm over the last axis (torch semantics): y = (x−mean)/√(var+eps)·γ + β.
// COOPERATIVE (§T345/§T346): ONE 256-thread threadgroup per row, coalesced strided
// access. Two tree reductions (row mean, then variance) share red[256] with a
// barrier between phases. P = {rows, dim}, E = {eps}.
static NSString* const kLayerNormSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void layernorm(device const float* X [[buffer(0)]],\n"
     "                      device const float* G [[buffer(1)]],\n"
     "                      device const float* B [[buffer(2)]],\n"
     "                      device float* O [[buffer(3)]],\n"
     "                      constant int* P [[buffer(4)]],\n"
     "                      constant float* E [[buffer(5)]],\n"
     "                      uint row [[threadgroup_position_in_grid]],\n"
     "                      uint lane [[thread_position_in_threadgroup]],\n"
     "                      uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], dim=P[1];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*dim;\n"
     "  threadgroup float red[256];\n"
     "  float psum=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ psum+=X[base+j]; }\n"
     "  red[lane]=psum;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float mean=red[0]/float(dim);\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  float pv=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ float dv=X[base+j]-mean; pv+=dv*dv; }\n"
     "  red[lane]=pv;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float inv=rsqrt(red[0]/float(dim)+E[0]);\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ O[base+j]=(X[base+j]-mean)*inv*G[j]+B[j]; }\n"
     "}\n";

static id<MTLComputePipelineState> gLayerNorm = nil;

static int ensure_layernorm(void) {
    if (gLayerNorm != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kLayerNormSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"layernorm"];
    if (fn == nil) return -6;
    gLayerNorm = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gLayerNorm != nil ? 0 : -6;
}

int mtl_layernorm_f32(const float* X, const float* G, const float* B, float* O,
                      int rows, int dim, float eps) {
    if (ensure_init() != 0) return -1;
    if (ensure_layernorm() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)rows * dim * sizeof(float);
        size_t gLen = (size_t)dim * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> gb = pool_in(G, gLen);
        id<MTLBuffer> bb = pool_in(B, gLen);
        id<MTLBuffer> ob = pool_buf(xLen);
        int P[2] = {rows, dim};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float E[1] = {eps};
        id<MTLBuffer> eb = pool_in(E, sizeof(E));
        if (xb == nil || gb == nil || bb == nil || ob == nil || pb == nil || eb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gLayerNorm];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gb offset:0 atIndex:1];
        [enc setBuffer:bb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        [enc setBuffer:eb offset:0 atIndex:5];
        // Cooperative (§T346): one 256-thread threadgroup per row (red[256] fixes the size).
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, xLen);
        return 0;
    }
}

// Generic unary elementwise op, one thread per element: O[i]=f(op,X[i]). Op codes match
// backend/metal/metal.go and the Vulkan kernel: 0 Neg,1 Exp,2 Log,3 Tanh,4 ReLU,5 Sigmoid,
// 6 SiLU,7 Sqrt,8 Abs,9 GELU. GELU = exact erf form 0.5·x·(1+erf(x/√2)) (§T352: was a
// ref/CPU fallback dominating the FFN — 11ms/op — now on-GPU; Metal's erf matches the
// reference's within §V11 f32 tolerance).
static NSString* const kUnarySource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     // erf via Abramowitz & Stegun 7.1.26 (max abs err ~1.5e-7 ≤ f32 tol); Metal lacks erf.
     "static inline float aserf(float x){ float ax=fabs(x);\n"
     "  float t=1.0f/(1.0f+0.3275911f*ax);\n"
     "  float y=1.0f-((((( 1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-ax*ax);\n"
     "  return x>=0.0f? y : -y; }\n"
     "kernel void unary(device const float* X [[buffer(0)]],\n"
     "                  device float* O [[buffer(1)]],\n"
     "                  constant int* P [[buffer(2)]],\n"
     "                  uint gid [[thread_position_in_grid]]) {\n"
     "  int n=P[0], op=P[1];\n"
     "  if ((int)gid >= n) return;\n"
     "  float v=X[gid], r;\n"
     "  switch (op) {\n"
     "    case 0: r=-v; break;\n"
     "    case 1: r=exp(v); break;\n"
     "    case 2: r=log(v); break;\n"
     "    case 3: r=tanh(v); break;\n"
     "    case 4: r=max(v,0.0f); break;\n"
     "    case 5: r=1.0f/(1.0f+exp(-v)); break;\n"
     "    case 6: r=v/(1.0f+exp(-v)); break;\n"
     "    case 7: r=sqrt(v); break;\n"
     "    case 8: r=fabs(v); break;\n"
     "    case 9: r=0.5f*v*(1.0f+aserf(v*0.70710678118654752f)); break;\n"
     "    default: r=v; break;\n"
     "  }\n"
     "  O[gid]=r;\n"
     "}\n";

static id<MTLComputePipelineState> gUnary = nil;

static int ensure_unary(void) {
    if (gUnary != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kUnarySource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"unary"];
    if (fn == nil) return -6;
    gUnary = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gUnary != nil ? 0 : -6;
}

// Bias add O[i,j] = X[i,j] + B[j] (bias broadcast over rows), one thread per element
// (§T352: was a ref/CPU fallback — ~4ms/op in the FFN — now on-GPU). P = {rows, n}.
static NSString* const kAddBiasSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void addbias(device const float* X [[buffer(0)]],\n"
     "                    device const float* B [[buffer(1)]],\n"
     "                    device float* O [[buffer(2)]],\n"
     "                    constant int* P [[buffer(3)]],\n"
     "                    uint gid [[thread_position_in_grid]]) {\n"
     "  int rows=P[0], n=P[1];\n"
     "  if ((int)gid >= rows*n) return;\n"
     "  O[gid]=X[gid]+B[(int)gid%n];\n"
     "}\n";

static id<MTLComputePipelineState> gAddBias = nil;

static int ensure_addbias(void) {
    if (gAddBias != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kAddBiasSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"addbias"];
    if (fn == nil) return -6;
    gAddBias = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gAddBias != nil ? 0 : -6;
}

int mtl_addbias_f32(const float* X, const float* B, float* O, int rows, int n) {
    if (ensure_init() != 0) return -1;
    if (ensure_addbias() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)rows * n * sizeof(float);
        size_t bLen = (size_t)n * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> bb = pool_in(B, bLen);
        id<MTLBuffer> ob = pool_buf(xLen);
        int P[2] = {rows, n};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || bb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gAddBias];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = rows * n;
        NSUInteger tg = gAddBias.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, xLen);
        return 0;
    }
}

// Bias-add backward db[j] = Σ_i G[rows,n][i,j] (column sum), one thread per column
// (§T354: the VJP's bias-grad was a CPU scalar row-sum). P = {rows, n}.
static NSString* const kAddBiasBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void addbias_bwd(device const float* G [[buffer(0)]],\n"
     "                        device float* DB [[buffer(1)]],\n"
     "                        constant int* P [[buffer(2)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int rows=P[0], n=P[1];\n"
     "  if ((int)gid >= n) return;\n"
     "  float s=0.0f; for (int i=0;i<rows;i++) s+=G[i*n+(int)gid]; DB[gid]=s;\n"
     "}\n";

static id<MTLComputePipelineState> gAddBiasBwd = nil;

static int ensure_addbias_bwd(void) {
    if (gAddBiasBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kAddBiasBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"addbias_bwd"];
    if (fn == nil) return -6;
    gAddBiasBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gAddBiasBwd != nil ? 0 : -6;
}

int mtl_addbias_backward_f32(const float* G, float* DB, int rows, int n) {
    if (ensure_init() != 0) return -1;
    if (ensure_addbias_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t gLen = (size_t)rows * n * sizeof(float);
        size_t bLen = (size_t)n * sizeof(float);
        id<MTLBuffer> gb = pool_in(G, gLen);
        id<MTLBuffer> db = pool_buf(bLen);
        int P[2] = {rows, n};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (gb == nil || db == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gAddBiasBwd];
        [enc setBuffer:gb offset:0 atIndex:0];
        [enc setBuffer:db offset:0 atIndex:1];
        [enc setBuffer:pb offset:0 atIndex:2];
        NSUInteger tg = gAddBiasBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DB, db.contents, bLen);
        return 0;
    }
}

// GELU backward dx = g·(Φ(x)+x·φ(x)), Φ=0.5(1+erf(x/√2)), φ=(1/√2π)exp(−x²/2) — one thread
// per element (§T353: the VJP was a ~30ms CPU scalar loop at the FFN's 256×2048). erf via the
// A&S 7.1.26 approx (Metal has none), matching the reference within §V11 f32 tol. P = {n}.
static NSString* const kGELUBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "static inline float aserf(float x){ float ax=fabs(x);\n"
     "  float t=1.0f/(1.0f+0.3275911f*ax);\n"
     "  float y=1.0f-((((( 1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-0.284496736f)*t+0.254829592f)*t*exp(-ax*ax);\n"
     "  return x>=0.0f? y : -y; }\n"
     "kernel void gelu_bwd(device const float* X [[buffer(0)]],\n"
     "                     device const float* G [[buffer(1)]],\n"
     "                     device float* DX [[buffer(2)]],\n"
     "                     constant int* P [[buffer(3)]],\n"
     "                     uint gid [[thread_position_in_grid]]) {\n"
     "  int n=P[0]; if ((int)gid >= n) return;\n"
     "  float x=X[gid];\n"
     "  float phi=0.5f*(1.0f+aserf(x*0.70710678118654752f));\n"
     "  float pdf=0.39894228040143270f*exp(-0.5f*x*x);\n"
     "  DX[gid]=G[gid]*(phi+x*pdf);\n"
     "}\n";

static id<MTLComputePipelineState> gGELUBwd = nil;

static int ensure_gelu_bwd(void) {
    if (gGELUBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kGELUBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"gelu_bwd"];
    if (fn == nil) return -6;
    gGELUBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gGELUBwd != nil ? 0 : -6;
}

// SiLU backward dx = g·σ(x)·(1+x·(1−σ(x))), σ=1/(1+e^−x) — one thread per element
// (§T362: SwiGLU's activation; the VJP was a ~29ms CPU scalar loop at the FFN's 256×2048).
static NSString* const kSiLUBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void silu_bwd(device const float* X [[buffer(0)]],\n"
     "                     device const float* G [[buffer(1)]],\n"
     "                     device float* DX [[buffer(2)]],\n"
     "                     constant int* P [[buffer(3)]],\n"
     "                     uint gid [[thread_position_in_grid]]) {\n"
     "  int n=P[0]; if ((int)gid >= n) return;\n"
     "  float x=X[gid]; float s=1.0f/(1.0f+exp(-x));\n"
     "  DX[gid]=G[gid]*s*(1.0f+x*(1.0f-s));\n"
     "}\n";

static id<MTLComputePipelineState> gSiLUBwd = nil;

static int ensure_silu_bwd(void) {
    if (gSiLUBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSiLUBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"silu_bwd"];
    if (fn == nil) return -6;
    gSiLUBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gSiLUBwd != nil ? 0 : -6;
}

int mtl_silu_backward_f32(const float* X, const float* G, float* DX, int n) {
    if (ensure_init() != 0) return -1;
    if (ensure_silu_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)n * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, len);
        id<MTLBuffer> gb = pool_in(G, len);
        id<MTLBuffer> ob = pool_buf(len);
        int P[1] = {n};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || gb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gSiLUBwd];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        NSUInteger tg = gSiLUBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DX, ob.contents, len);
        return 0;
    }
}

int mtl_gelu_backward_f32(const float* X, const float* G, float* DX, int n) {
    if (ensure_init() != 0) return -1;
    if (ensure_gelu_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)n * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, len);
        id<MTLBuffer> gb = pool_in(G, len);
        id<MTLBuffer> ob = pool_buf(len);
        int P[1] = {n};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || gb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gGELUBwd];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        NSUInteger tg = gGELUBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DX, ob.contents, len);
        return 0;
    }
}

int mtl_unary_f32(const float* X, float* O, int n, int op) {
    if (ensure_init() != 0) return -1;
    if (ensure_unary() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)n * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, len);
        id<MTLBuffer> ob = pool_buf(len);
        int P[2] = {n, op};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gUnary];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:ob offset:0 atIndex:1];
        [enc setBuffer:pb offset:0 atIndex:2];
        NSUInteger tg = gUnary.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, len);
        return 0;
    }
}

// Generic same-shape binary elementwise op, one thread per element: O[i]=f(op,A[i],B[i]).
// Op codes match backend/metal/metal.go and the Vulkan kernel: 0 Add,1 Sub,2 Mul,3 Div,
// 4 Maximum,5 Minimum. Broadcasting falls back to the reference (handled in Go).
static NSString* const kBinarySource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void binaryop(device const float* A [[buffer(0)]],\n"
     "                     device const float* B [[buffer(1)]],\n"
     "                     device float* O [[buffer(2)]],\n"
     "                     constant int* P [[buffer(3)]],\n"
     "                     uint gid [[thread_position_in_grid]]) {\n"
     "  int n=P[0], op=P[1];\n"
     "  if ((int)gid >= n) return;\n"
     "  float x=A[gid], y=B[gid], r;\n"
     "  switch (op) {\n"
     "    case 0: r=x+y; break;\n"
     "    case 1: r=x-y; break;\n"
     "    case 2: r=x*y; break;\n"
     "    case 3: r=x/y; break;\n"
     "    case 4: r=max(x,y); break;\n"
     "    case 5: r=min(x,y); break;\n"
     "    case 6: r=(x/(1.0f+exp(-x)))*y; break;\n" // fused SwiGLU: silu(gate)*up (SPEC T613)
     "    default: r=x; break;\n"
     "  }\n"
     "  O[gid]=r;\n"
     "}\n";

static id<MTLComputePipelineState> gBinary = nil;

static int ensure_binary(void) {
    if (gBinary != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kBinarySource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"binaryop"];
    if (fn == nil) return -6;
    gBinary = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gBinary != nil ? 0 : -6;
}

int mtl_binary_f32(const float* A, const float* B, float* O, int n, int op) {
    if (ensure_init() != 0) return -1;
    if (ensure_binary() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)n * sizeof(float);
        id<MTLBuffer> ab = pool_in(A, len);
        id<MTLBuffer> bb = pool_in(B, len);
        id<MTLBuffer> ob = pool_buf(len);
        int P[2] = {n, op};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (ab == nil || bb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gBinary];
        [enc setBuffer:ab offset:0 atIndex:0];
        [enc setBuffer:bb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        NSUInteger tg = gBinary.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)n < tg) tg = (NSUInteger)n;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, len);
        return 0;
    }
}

// RoPE BACKWARD (inverse rotation, angle→−angle) on the upstream gradient G[seq,width].
// Mirrors kRoPESource with the transposed rotation: dq[i]=g[i]c+g[i+half]s,
// dq[i+half]=g[i+half]c−g[i]s. Same host-precomputed inv/posDiv.
static NSString* const kRoPEBackwardSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void rope_bwd(device const float* G [[buffer(0)]],\n"
     "                     device const float* INV [[buffer(1)]],\n"
     "                     device float* O [[buffer(2)]],\n"
     "                     constant int* P [[buffer(3)]],\n"
     "                     constant float* F [[buffer(4)]],\n"
     "                     uint gid [[thread_position_in_grid]]) {\n"
     "  int seq=P[0], width=P[1], heads=P[2], hd=P[3], halfd=P[4], posOffset=P[5];\n"
     "  int hh=heads*halfd; int total=seq*hh;\n"
     "  if ((int)gid >= total) return;\n"
     "  int p=(int)gid/hh; int rem=(int)gid%hh; int h=rem/halfd; int i=rem%halfd;\n"
     "  int base=p*width + h*hd + i;\n"
     "  float pos=float(p+posOffset)/F[0];\n"
     "  float theta=INV[i];\n"
     "  float c=cos(pos*theta), s=sin(pos*theta);\n"
     "  float gi=G[base], gih=G[base+halfd];\n"
     "  O[base]=gi*c+gih*s;\n"
     "  O[base+halfd]=gih*c-gi*s;\n"
     "}\n";

static id<MTLComputePipelineState> gRoPEBwd = nil;

static int ensure_rope_bwd(void) {
    if (gRoPEBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRoPEBackwardSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"rope_bwd"];
    if (fn == nil) return -6;
    gRoPEBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRoPEBwd != nil ? 0 : -6;
}

int mtl_rope_backward_f32(const float* G, const float* inv, float* O,
                          int seq, int width, int heads, int hd, int half, int posOffset, float posDiv) {
    if (ensure_init() != 0) return -1;
    if (ensure_rope_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t gLen = (size_t)seq * width * sizeof(float);
        size_t iLen = (size_t)half * sizeof(float);

        id<MTLBuffer> gb = pool_in(G, gLen);
        id<MTLBuffer> ib = pool_in(inv, iLen);
        id<MTLBuffer> ob = pool_buf(gLen);
        int P[6] = {seq, width, heads, hd, half, posOffset};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float F[1] = {posDiv};
        id<MTLBuffer> fb = pool_in(F, sizeof(F));
        if (gb == nil || ib == nil || ob == nil || pb == nil || fb == nil) return -2;

        int total = seq * heads * half;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRoPEBwd];
        [enc setBuffer:gb offset:0 atIndex:0];
        [enc setBuffer:ib offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        [enc setBuffer:fb offset:0 atIndex:4];
        NSUInteger tg = gRoPEBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, gLen);
        return 0;
    }
}

// RMSNorm backward (§R29/§R35). COOPERATIVE (§T347): ONE 256-thread threadgroup per
// row, coalesced strided access. A single strided pass accumulates both per-row sums
// (ms=Σx², s=Σg·γ·x) into two threadgroup scratch arrays, one tree reduction combines
// them, then each lane writes its strided dx_j=r·a_j−x_j·r³·s/d and atomically adds its
// dγ_j=g_j·x_j·r into DGamma (cross-row accumulator, uploaded zero). P={rows,dim}, E={eps}.
static NSString* const kRMSNormBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void rmsnorm_bwd(device const float* X [[buffer(0)]],\n"
     "                        device const float* Gam [[buffer(1)]],\n"
     "                        device const float* G [[buffer(2)]],\n"
     "                        device float* DX [[buffer(3)]],\n"
     "                        device atomic_float* DGamma [[buffer(4)]],\n"
     "                        constant int* P [[buffer(5)]],\n"
     "                        constant float* E [[buffer(6)]],\n"
     "                        uint row [[threadgroup_position_in_grid]],\n"
     "                        uint lane [[thread_position_in_threadgroup]],\n"
     "                        uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], dim=P[1];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*dim;\n"
     "  threadgroup float redA[256]; threadgroup float redB[256];\n"
     "  float pms=0.0f, ps=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ float v=X[base+j]; pms+=v*v; ps+=G[base+j]*Gam[j]*v; }\n"
     "  redA[lane]=pms; redB[lane]=ps;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s){ redA[lane]+=redA[lane+s]; redB[lane]+=redB[lane+s]; } threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float r=rsqrt(redA[0]/float(dim)+E[0]);\n"
     "  float coef=r*r*r*redB[0]/float(dim);\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){\n"
     "    float aj=G[base+j]*Gam[j];\n"
     "    DX[base+j]=r*aj - X[base+j]*coef;\n"
     "    atomic_fetch_add_explicit(&DGamma[j], G[base+j]*X[base+j]*r, memory_order_relaxed);\n"
     "  }\n"
     "}\n";

static id<MTLComputePipelineState> gRMSNormBwd = nil;

static int ensure_rmsnorm_bwd(void) {
    if (gRMSNormBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRMSNormBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"rmsnorm_bwd"];
    if (fn == nil) return -6;
    gRMSNormBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRMSNormBwd != nil ? 0 : -6;
}

int mtl_rmsnorm_backward_f32(const float* X, const float* Gamma, const float* G,
                             float* DX, float* DGamma, int rows, int dim, float eps) {
    if (ensure_init() != 0) return -1;
    if (ensure_rmsnorm_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)rows * dim * sizeof(float);
        size_t gLen = (size_t)dim * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> gamb = pool_in(Gamma, gLen);
        id<MTLBuffer> gb = pool_in(G, xLen);
        id<MTLBuffer> dxb = pool_buf(xLen);
        id<MTLBuffer> dgb = pool_in(DGamma, gLen); // zero-init from host
        int P[2] = {rows, dim};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float E[1] = {eps};
        id<MTLBuffer> eb = pool_in(E, sizeof(E));
        if (xb == nil || gamb == nil || gb == nil || dxb == nil || dgb == nil || pb == nil || eb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRMSNormBwd];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gamb offset:0 atIndex:1];
        [enc setBuffer:gb offset:0 atIndex:2];
        [enc setBuffer:dxb offset:0 atIndex:3];
        [enc setBuffer:dgb offset:0 atIndex:4];
        [enc setBuffer:pb offset:0 atIndex:5];
        [enc setBuffer:eb offset:0 atIndex:6];
        // Cooperative (§T347): one 256-thread threadgroup per row (redA/redB[256] fix the size).
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DX, dxb.contents, xLen);
        memcpy(DGamma, dgb.contents, gLen);
        return 0;
    }
}

// LayerNorm backward (§R35). COOPERATIVE (§T347): ONE 256-thread threadgroup per row,
// coalesced strided access. Reductions in threadgroup scratch: (1) row mean, (2) row
// variance, (3) meanA=Σa_j & meanAX=Σa_j·x̂_j (combined pass that also atomically adds
// dγ_j=g_j·x̂_j and dβ_j=g_j into the cross-row accumulators). Then each lane writes its
// strided dx_j=inv·(a_j−meanA−x̂_j·meanAX). P={rows,dim}, E={eps}.
static NSString* const kLayerNormBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void layernorm_bwd(device const float* X [[buffer(0)]],\n"
     "                          device const float* Gam [[buffer(1)]],\n"
     "                          device const float* G [[buffer(2)]],\n"
     "                          device float* DX [[buffer(3)]],\n"
     "                          device atomic_float* DGamma [[buffer(4)]],\n"
     "                          device atomic_float* DBeta [[buffer(5)]],\n"
     "                          constant int* P [[buffer(6)]],\n"
     "                          constant float* E [[buffer(7)]],\n"
     "                          uint row [[threadgroup_position_in_grid]],\n"
     "                          uint lane [[thread_position_in_threadgroup]],\n"
     "                          uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], dim=P[1];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*dim; float fd=float(dim);\n"
     "  threadgroup float redA[256]; threadgroup float redB[256];\n"
     "  float psum=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ psum+=X[base+j]; }\n"
     "  redA[lane]=psum;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) redA[lane]+=redA[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float mean=redA[0]/fd;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  float pv=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){ float dv=X[base+j]-mean; pv+=dv*dv; }\n"
     "  redA[lane]=pv;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s) redA[lane]+=redA[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float inv=rsqrt(redA[0]/fd+E[0]);\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  float pa=0.0f, pax=0.0f;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){\n"
     "    float xhat=(X[base+j]-mean)*inv; float gj=G[base+j]; float aj=gj*Gam[j];\n"
     "    pa+=aj; pax+=aj*xhat;\n"
     "    atomic_fetch_add_explicit(&DGamma[j], gj*xhat, memory_order_relaxed);\n"
     "    atomic_fetch_add_explicit(&DBeta[j], gj, memory_order_relaxed);\n"
     "  }\n"
     "  redA[lane]=pa; redB[lane]=pax;\n"
     "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2; s>0; s>>=1){ if(lane<s){ redA[lane]+=redA[lane+s]; redB[lane]+=redB[lane+s]; } threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float meanA=redA[0]/fd, meanAX=redB[0]/fd;\n"
     "  for (int j=(int)lane;j<dim;j+=(int)tgsize){\n"
     "    float xhat=(X[base+j]-mean)*inv; float aj=G[base+j]*Gam[j];\n"
     "    DX[base+j]=inv*(aj-meanA-xhat*meanAX);\n"
     "  }\n"
     "}\n";

static id<MTLComputePipelineState> gLayerNormBwd = nil;

static int ensure_layernorm_bwd(void) {
    if (gLayerNormBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kLayerNormBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"layernorm_bwd"];
    if (fn == nil) return -6;
    gLayerNormBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gLayerNormBwd != nil ? 0 : -6;
}

int mtl_layernorm_backward_f32(const float* X, const float* Gamma, const float* G,
                               float* DX, float* DGamma, float* DBeta, int rows, int dim, float eps) {
    if (ensure_init() != 0) return -1;
    if (ensure_layernorm_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)rows * dim * sizeof(float);
        size_t gLen = (size_t)dim * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> gamb = pool_in(Gamma, gLen);
        id<MTLBuffer> gb = pool_in(G, xLen);
        id<MTLBuffer> dxb = pool_buf(xLen);
        id<MTLBuffer> dgb = pool_in(DGamma, gLen); // zero-init
        id<MTLBuffer> dbb = pool_in(DBeta, gLen);  // zero-init
        int P[2] = {rows, dim};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float E[1] = {eps};
        id<MTLBuffer> eb = pool_in(E, sizeof(E));
        if (xb == nil || gamb == nil || gb == nil || dxb == nil || dgb == nil || dbb == nil || pb == nil || eb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gLayerNormBwd];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:gamb offset:0 atIndex:1];
        [enc setBuffer:gb offset:0 atIndex:2];
        [enc setBuffer:dxb offset:0 atIndex:3];
        [enc setBuffer:dgb offset:0 atIndex:4];
        [enc setBuffer:dbb offset:0 atIndex:5];
        [enc setBuffer:pb offset:0 atIndex:6];
        [enc setBuffer:eb offset:0 atIndex:7];
        // Cooperative (§T347): one 256-thread threadgroup per row (redA/redB[256] fix the size).
        [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DX, dxb.contents, xLen);
        memcpy(DGamma, dgb.contents, gLen);
        memcpy(DBeta, dbb.contents, gLen);
        return 0;
    }
}

// Cross-entropy backward (ADR-0007, §R52/§R113), one thread per row of Z[b,c]:
// dz[i,k]=gv·(p_k − q'_k + 2·zl·lse·p_k)/b, stable softmax p, q'(k)=(1−ε)δ(k,tᵢ)+ε/c.
// P={b,c}, F={gv,eps,zl}; targets T as floats.
static NSString* const kCEBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void ce_bwd(device const float* Z [[buffer(0)]],\n"
     "                   device const float* T [[buffer(1)]],\n"
     "                   device float* DZ [[buffer(2)]],\n"
     "                   constant int* P [[buffer(3)]],\n"
     "                   constant float* F [[buffer(4)]],\n"
     "                   uint gid [[thread_position_in_grid]]) {\n"
     "  int b=P[0], c=P[1];\n"
     "  if ((int)gid >= b) return;\n"
     "  float gv=F[0], eps=F[1], zl=F[2];\n"
     "  int base=(int)gid*c;\n"
     "  float m=Z[base];\n"
     "  for (int j=1;j<c;j++){ m=max(m,Z[base+j]); }\n"
     "  float sum=0.0f;\n"
     "  for (int j=0;j<c;j++){ sum+=exp(Z[base+j]-m); }\n"
     "  int ti=(int)T[gid];\n"
     "  float lseTerm = (zl!=0.0f) ? 2.0f*zl*(m+log(sum)) : 0.0f;\n"
     "  float invb=1.0f/float(b);\n"
     "  for (int j=0;j<c;j++){\n"
     "    float p=exp(Z[base+j]-m)/sum;\n"
     "    float q=eps/float(c); if (j==ti) q+=1.0f-eps;\n"
     "    DZ[base+j]=gv*(p-q+lseTerm*p)*invb;\n"
     "  }\n"
     "}\n";

static id<MTLComputePipelineState> gCEBwd = nil;

static int ensure_ce_bwd(void) {
    if (gCEBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kCEBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"ce_bwd"];
    if (fn == nil) return -6;
    gCEBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gCEBwd != nil ? 0 : -6;
}

int mtl_crossentropy_backward_f32(const float* Z, const float* T, float* DZ,
                                  int b, int c, float gv, float eps, float zl) {
    if (ensure_init() != 0) return -1;
    if (ensure_ce_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t zLen = (size_t)b * c * sizeof(float);
        size_t tLen = (size_t)b * sizeof(float);

        id<MTLBuffer> zb = pool_in(Z, zLen);
        id<MTLBuffer> tb = pool_in(T, tLen);
        id<MTLBuffer> db = pool_buf(zLen);
        int P[2] = {b, c};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        float F[3] = {gv, eps, zl};
        id<MTLBuffer> fb = pool_in(F, sizeof(F));
        if (zb == nil || tb == nil || db == nil || pb == nil || fb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gCEBwd];
        [enc setBuffer:zb offset:0 atIndex:0];
        [enc setBuffer:tb offset:0 atIndex:1];
        [enc setBuffer:db offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        [enc setBuffer:fb offset:0 atIndex:4];
        NSUInteger tg = 64; // row-parallel kernel: 64 rows/threadgroup like the vulkan twin — maxTotal (~1024) yields too few threadgroups to occupy all GPU cores (§T339)
        if ((NSUInteger)b < tg) tg = (NSUInteger)b;
        [enc dispatchThreads:MTLSizeMake(b, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DZ, db.contents, zLen);
        return 0;
    }
}

// Embedding backward (§T34), one thread per (token i, dim j): scatter-add G[i,j] into
// DTable[Idx[i],j] with a float atomic (tokens sharing an index collide). DTable uploaded
// zero. P={n,d,m}; Idx as floats.
static NSString* const kEmbedBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void embed_bwd(device const float* Idx [[buffer(0)]],\n"
     "                      device const float* G [[buffer(1)]],\n"
     "                      device atomic_float* DTable [[buffer(2)]],\n"
     "                      constant int* P [[buffer(3)]],\n"
     "                      uint gid [[thread_position_in_grid]]) {\n"
     "  int n=P[0], d=P[1], m=P[2];\n"
     "  if ((int)gid >= m*d) return;\n"
     "  int i=(int)gid/d, j=(int)gid%d;\n"
     "  int t=(int)Idx[i];\n"
     "  if (t<0 || t>=n) return;\n"
     "  atomic_fetch_add_explicit(&DTable[t*d+j], G[gid], memory_order_relaxed);\n"
     "}\n";

static id<MTLComputePipelineState> gEmbedBwd = nil;

static int ensure_embed_bwd(void) {
    if (gEmbedBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kEmbedBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"embed_bwd"];
    if (fn == nil) return -6;
    gEmbedBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gEmbedBwd != nil ? 0 : -6;
}

int mtl_embed_backward_f32(const float* Idx, const float* G, float* DTable, int n, int d, int m) {
    if (ensure_init() != 0) return -1;
    if (ensure_embed_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t iLen = (size_t)m * sizeof(float);
        size_t gLen = (size_t)m * d * sizeof(float);
        size_t tLen = (size_t)n * d * sizeof(float);

        id<MTLBuffer> ib = pool_in(Idx, iLen);
        id<MTLBuffer> gb = pool_in(G, gLen);
        id<MTLBuffer> tb = pool_in(DTable, tLen); // zero-init
        int P[3] = {n, d, m};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (ib == nil || gb == nil || tb == nil || pb == nil) return -2;

        int total = m * d;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gEmbedBwd];
        [enc setBuffer:ib offset:0 atIndex:0];
        [enc setBuffer:gb offset:0 atIndex:1];
        [enc setBuffer:tb offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        NSUInteger tgs = gEmbedBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tgs) tgs = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(DTable, tb.contents, tLen);
        return 0;
    }
}

static NSString* const kQMatMulQ8Source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q8_0(device const float* X [[buffer(0)]],\n"
     "                         device const uchar* W [[buffer(1)]],\n"
     "                         device float* O [[buffer(2)]],\n"
     "                         constant int* P [[buffer(3)]],\n"
     "                         uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nb=K/32; int rowBytes=nb*34; int woffRow=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int b=0;b<nb;b++){\n"
     "    int woff=woffRow+b*34;\n"
     "    ushort raw=(ushort)W[woff] | ((ushort)W[woff+1]<<8);\n"
     "    float scale=float(as_type<half>(raw));\n"
     "    int xoff=mi*K+b*32;\n"
     "    for (int i=0;i<32;i++){ char q=(char)W[woff+2+i]; acc += X[xoff+i]*scale*float(q); }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ8 = nil;

static int ensure_qmatmul_q8(void) {
    if (gQMatMulQ8 != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ8Source options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q8_0"];
    if (fn == nil) return -6;
    gQMatMulQ8 = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ8 != nil ? 0 : -6;
}

int mtl_qmatmul_q8_0(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q8() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nb = K / 32;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nb * 34;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ8];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ8.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Q4_0 quantized matmul (§T166): mirrors qmatmul_q8_0 with 18-byte blocks (f16 scale + 16 bytes of
// 32 packed 4-bit nibbles). Per block: low nibble of byte i → element i, high nibble → element i+16,
// value = d·(nibble−8) (exactly matching gguf dequantQ4_0). One thread per output (mi,ni).
static NSString* const kQMatMulQ4_0Source =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q4_0(device const float* X [[buffer(0)]],\n"
     "                         device const uchar* W [[buffer(1)]],\n"
     "                         device float* O [[buffer(2)]],\n"
     "                         constant int* P [[buffer(3)]],\n"
     "                         uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nb=K/32; int rowBytes=nb*18; int woffRow=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int b=0;b<nb;b++){\n"
     "    int woff=woffRow+b*18;\n"
     "    ushort raw=(ushort)W[woff] | ((ushort)W[woff+1]<<8);\n"
     "    float d=float(as_type<half>(raw));\n"
     "    int xoff=mi*K+b*32;\n"
     "    for (int i=0;i<16;i++){ uchar q=W[woff+2+i];\n"
     "      float lo=float((int)(q & 0x0F) - 8); float hi=float((int)(q >> 4) - 8);\n"
     "      acc += X[xoff+i]*d*lo + X[xoff+i+16]*d*hi; }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ4_0 = nil;

static int ensure_qmatmul_q4_0(void) {
    if (gQMatMulQ4_0 != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ4_0Source options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q4_0"];
    if (fn == nil) return -6;
    gQMatMulQ4_0 = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ4_0 != nil ? 0 : -6;
}

int mtl_qmatmul_q4_0(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q4_0() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nb = K / 32;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nb * 18;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ4_0];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ4_0.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Forward declarations so mtl_qmatmul_resident (below) can select any k-quant pipeline, whose
// ensure_*/g* definitions appear later in this file (§T155). Tentative static definitions.
static int ensure_qmatmul_q2k(void);
static int ensure_qmatmul_q3k(void);
static int ensure_qmatmul_q4k(void);
static int ensure_qmatmul_q5k(void);
static int ensure_qmatmul_q6k(void);
static id<MTLComputePipelineState> gQMatMulQ2K, gQMatMulQ3K, gQMatMulQ4K, gQMatMulQ5K, gQMatMulQ6K;

// Device-resident quantized weights: upload a weight blob into a retained MTLBuffer once and
// reuse it across many matmuls (§T153). ARC bridging keeps the buffer alive past the call.
void* mtl_qweight_upload(const unsigned char* W, int len) {
    if (ensure_init() != 0) return NULL;
    // Deliberately NOT pooled: the weight buffer outlives this call (resident until
    // mtl_qweight_free), while pool buffers become reusable after each op returns.
    id<MTLBuffer> buf = [gDevice newBufferWithBytes:W length:(size_t)len options:MTLResourceStorageModeShared];
    if (buf == nil) return NULL;
    return (__bridge_retained void*)buf; // +1 retain transferred to the handle
}

void mtl_qweight_free(void* handle) {
    if (handle == NULL) return;
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle; // ARC releases at scope end
    (void)buf;
}

// ---- batch-dispatch validation bench (§T368, validates ADR-0019's premise) --
// Dispatches the (cached) unary kernel `iters` times over a tiny buffer, either as
// `iters` separate command buffers each committed+waited (oneBuffer=0, the current
// per-op model) or ONE command buffer with all dispatches recorded + a single
// commit+wait (oneBuffer=1, the batched model). Measures whether batching recovers
// the per-op round-trip overhead — the whole point of ADR-0019 Phase 2. Returns 0.
int mtl_batch_dispatch_bench(int iters, int oneBuffer) {
    if (ensure_init() != 0) return -1;
    if (ensure_unary() != 0) return -6;
    @autoreleasepool {
        const int n = 256;
        id<MTLBuffer> xb = [gDevice newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [gDevice newBufferWithLength:n*sizeof(float) options:MTLResourceStorageModeShared];
        int P[2] = {n, 4}; // op 4 = ReLU (trivial)
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
        if (xb == nil || ob == nil || pb == nil) return -2;
        if (oneBuffer) {
            id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
            for (int i = 0; i < iters; ++i) {
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:gUnary];
                [enc setBuffer:xb offset:0 atIndex:0];
                [enc setBuffer:ob offset:0 atIndex:1];
                [enc setBuffer:pb offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];
                [enc endEncoding];
            }
            [cmd commit];
            [cmd waitUntilCompleted];
        } else {
            for (int i = 0; i < iters; ++i) {
                id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
                [enc setComputePipelineState:gUnary];
                [enc setBuffer:xb offset:0 atIndex:0];
                [enc setBuffer:ob offset:0 atIndex:1];
                [enc setBuffer:pb offset:0 atIndex:2];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(n,1,1)];
                [enc endEncoding];
                [cmd commit];
                [cmd waitUntilCompleted];
            }
        }
        return 0;
    }
}

// ---- batch recorder (§T370, ADR-0019 Phase 2) ------------------------------
// A recorder holds ONE open command buffer; record-mode ops encode into it over
// device-resident buffers (§T367) without per-op commit/wait, and finish submits it
// once. Intermediates stay on the GPU between ops (Metal's default hazard tracking
// orders dependent buffer accesses, §T369). This is the reusable primitive the batched
// decode path (Phase 2) is built from; this fire wires the elementwise ops, later fires
// add matmul/norm/attention record-mode variants.
void* mtl_recorder_begin(void) {
    if (ensure_init() != 0) return NULL;
    id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
    if (cmd == nil) return NULL;
    return (__bridge_retained void*)cmd; // retained until finish/free
}

// mtl_recorder_unary encodes O = f(op, X) over n f32 elements of the device buffers
// xh, oh (may alias) into the recorder's command buffer. No commit.
int mtl_recorder_unary(void* rec, void* xh, void* oh, int n, int op) {
    if (rec == NULL || xh == NULL || oh == NULL) return -2;
    if (ensure_unary() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)xh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[2] = {n, op};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    if (pb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gUnary];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:ob offset:0 atIndex:1];
    [enc setBuffer:pb offset:0 atIndex:2];
    NSUInteger tg = gUnary.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)n < tg) tg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_blit encodes a buffer→buffer copy of nbytes from src[srcOff] to dst[dstOff]
// into the recorder's command buffer (a blit-encoder). This is the KV-cache APPEND primitive:
// copy the freshly-computed k/v row [1,dkv] into cache[cacheLen] each decode step. Metal's
// default hazard tracking orders a preceding compute write before this copy. No commit.
int mtl_recorder_blit(void* rec, void* srcH, int srcOff, void* dstH, int dstOff, int nbytes) {
    if (rec == NULL || srcH == NULL || dstH == NULL || nbytes <= 0) return -2;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> sb = (__bridge id<MTLBuffer>)srcH;
    id<MTLBuffer> db = (__bridge id<MTLBuffer>)dstH;
    if ((NSUInteger)(srcOff + nbytes) > sb.length || (NSUInteger)(dstOff + nbytes) > db.length) return -2;
    id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
    [enc copyFromBuffer:sb sourceOffset:(NSUInteger)srcOff toBuffer:db destinationOffset:(NSUInteger)dstOff size:(NSUInteger)nbytes];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_copy2d encodes a strided rows×rowFloats sub-matrix copy src→dst into the
// recorder's command buffer: row r copies rowFloats f32 from src[srcOff + r·srcStride] to
// dst[dstOff + r·dstStride] (offsets/strides in ELEMENTS). One blit encoder, `rows` copy
// commands. This extracts/deposits the q/k/v sub-columns of a fused QKV buffer at prefill
// (SPEC T613); at rows=1 it degenerates to mtl_recorder_blit.
int mtl_recorder_copy2d(void* rec, void* srcH, int srcOff, int srcStride,
                        void* dstH, int dstOff, int dstStride, int rows, int rowFloats) {
    if (rec == NULL || srcH == NULL || dstH == NULL || rows < 1 || rowFloats < 1 ||
        srcOff < 0 || dstOff < 0 || srcStride < rowFloats || dstStride < rowFloats) return -2;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> sb = (__bridge id<MTLBuffer>)srcH;
    id<MTLBuffer> db = (__bridge id<MTLBuffer>)dstH;
    NSUInteger sEnd = ((NSUInteger)srcOff + (NSUInteger)(rows-1)*srcStride + rowFloats) * 4;
    NSUInteger dEnd = ((NSUInteger)dstOff + (NSUInteger)(rows-1)*dstStride + rowFloats) * 4;
    if (sEnd > sb.length || dEnd > db.length) return -2;
    id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
    for (int r = 0; r < rows; r++) {
        [enc copyFromBuffer:sb sourceOffset:((NSUInteger)srcOff + (NSUInteger)r*srcStride)*4
                   toBuffer:db destinationOffset:((NSUInteger)dstOff + (NSUInteger)r*dstStride)*4
                       size:(NSUInteger)rowFloats*4];
    }
    [enc endEncoding];
    return 0;
}

// mtl_recorder_binary encodes O = A (op) B over n f32 elements of device buffers into the
// recorder's command buffer (op 0=add,1=sub,2=mul,3=div,4=max,5=min). No commit.
int mtl_recorder_binary(void* rec, void* ah, void* bh, void* oh, int n, int op) {
    if (rec == NULL || ah == NULL || bh == NULL || oh == NULL) return -2;
    if (ensure_binary() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> ab = (__bridge id<MTLBuffer>)ah;
    id<MTLBuffer> bb = (__bridge id<MTLBuffer>)bh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[2] = {n, op};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    if (pb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gBinary];
    [enc setBuffer:ab offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:ob offset:0 atIndex:2];
    [enc setBuffer:pb offset:0 atIndex:3];
    NSUInteger tg = gBinary.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)n < tg) tg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_matmul encodes C = A·B (MPS, f32) over the device buffers ah, bh, ch into the
// recorder's command buffer. C must be a distinct device buffer (MPS forbids aliasing result).
// M×K · K×N → M×N, all row-major. No commit — intermediates stay device-resident (§T369).
int mtl_recorder_matmul(void* rec, void* ah, void* bh, void* ch, int M, int K, int N,
                        int accumulate) {
    if (rec == NULL || ah == NULL || bh == NULL || ch == NULL) return -2;
    if (ensure_init() != 0) return -1;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)ah;
    id<MTLBuffer> bBuf = (__bridge id<MTLBuffer>)bh;
    id<MTLBuffer> cBuf = (__bridge id<MTLBuffer>)ch;
    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(size_t)K*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:(size_t)N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(size_t)N*sizeof(float) dataType:MPSDataTypeFloat32];
    MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
    MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
    MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];
    // accumulate!=0: beta=1 turns the multiply into C += A·B — the residual-add
    // epilogue (SPEC T613), one dispatch instead of matmul-then-elementwise-add.
    MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
        initWithDevice:gDevice transposeLeft:NO transposeRight:NO
        resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:(accumulate != 0 ? 1.0 : 0.0)];
    [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
    return 0;
}

// mtl_recorder_rmsnorm encodes O = rmsnorm(X)·gamma over device buffers xh (rows×dim),
// gh (dim gamma weights), oh (rows×dim) into the recorder's command buffer. Cooperative
// kernel: one 256-thread threadgroup per row (§T345). No commit.
int mtl_recorder_rmsnorm(void* rec, void* xh, void* gh, void* oh, int rows, int dim, float eps) {
    if (rec == NULL || xh == NULL || gh == NULL || oh == NULL) return -2;
    if (ensure_rmsnorm() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)xh;
    id<MTLBuffer> gb = (__bridge id<MTLBuffer>)gh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[2] = {rows, dim};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    float E[1] = {eps};
    id<MTLBuffer> eb = [gDevice newBufferWithBytes:E length:sizeof(E) options:MTLResourceStorageModeShared];
    if (pb == nil || eb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gRMSNorm];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:gb offset:0 atIndex:1];
    [enc setBuffer:ob offset:0 atIndex:2];
    [enc setBuffer:pb offset:0 atIndex:3];
    [enc setBuffer:eb offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_layernorm encodes O = layernorm(X)·gamma + beta over device buffers xh (rows×dim),
// gh/bh (dim gamma/beta), oh (rows×dim) into the recorder's command buffer. Cooperative kernel:
// one 256-thread threadgroup per row (§T346). GPT-2-style norm (RMSNorm is the Llama variant). No commit.
int mtl_recorder_layernorm(void* rec, void* xh, void* gh, void* bh, void* oh, int rows, int dim, float eps) {
    if (rec == NULL || xh == NULL || gh == NULL || bh == NULL || oh == NULL) return -2;
    if (ensure_layernorm() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)xh;
    id<MTLBuffer> gb = (__bridge id<MTLBuffer>)gh;
    id<MTLBuffer> bb = (__bridge id<MTLBuffer>)bh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[2] = {rows, dim};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    float E[1] = {eps};
    id<MTLBuffer> eb = [gDevice newBufferWithBytes:E length:sizeof(E) options:MTLResourceStorageModeShared];
    if (pb == nil || eb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gLayerNorm];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:gb offset:0 atIndex:1];
    [enc setBuffer:bb offset:0 atIndex:2];
    [enc setBuffer:ob offset:0 atIndex:3];
    [enc setBuffer:pb offset:0 atIndex:4];
    [enc setBuffer:eb offset:0 atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_addbias encodes O[i,j] = X[i,j] + B[j] over device buffers (X,O [rows,n]; B [n])
// into the recorder's command buffer — the broadcast bias-add GPT's multi-token StepN needs
// (§T423; at rows=1 Binary add sufficed). No commit.
int mtl_recorder_addbias(void* rec, void* xh, void* bh, void* oh, int rows, int n) {
    if (rec == NULL || xh == NULL || bh == NULL || oh == NULL) return -2;
    if (ensure_addbias() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)xh;
    id<MTLBuffer> bb = (__bridge id<MTLBuffer>)bh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[2] = {rows, n};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    if (pb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gAddBias];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:bb offset:0 atIndex:1];
    [enc setBuffer:ob offset:0 atIndex:2];
    [enc setBuffer:pb offset:0 atIndex:3];
    int total = rows * n;
    NSUInteger tg = gAddBias.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)total < tg) tg = (NSUInteger)total;
    [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

// mtl_recorder_rope encodes O = RoPE(Q) over device buffers into the recorder's command
// buffer. Q,O are [seq,width]; invh holds `half` inverse frequencies. posOffset is the
// query's absolute start position (= KV-cache length at a decode step). No commit.
int mtl_recorder_rope(void* rec, void* qh, void* invh, void* oh,
                      int seq, int width, int heads, int hd, int half, int posOffset, float posDiv,
                      int elemOff) {
    if (rec == NULL || qh == NULL || invh == NULL || oh == NULL || elemOff < 0) return -2;
    if (ensure_rope() != 0) return -6;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> qb = (__bridge id<MTLBuffer>)qh;
    id<MTLBuffer> ib = (__bridge id<MTLBuffer>)invh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[6] = {seq, width, heads, hd, half, posOffset};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    float F[1] = {posDiv};
    id<MTLBuffer> fb = [gDevice newBufferWithBytes:F length:sizeof(F) options:MTLResourceStorageModeShared];
    if (pb == nil || fb == nil) return -2;
    int total = seq * heads * half;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gRoPE];
    // elemOff is a float-element offset into Q AND O (the fused-QKV sub-row view, §T613);
    // Metal buffer-bind offsets are bytes with 4-byte alignment for float — no shader change.
    [enc setBuffer:qb offset:(NSUInteger)elemOff*4 atIndex:0];
    [enc setBuffer:ib offset:0 atIndex:1];
    [enc setBuffer:ob offset:(NSUInteger)elemOff*4 atIndex:2];
    [enc setBuffer:pb offset:0 atIndex:3];
    [enc setBuffer:fb offset:0 atIndex:4];
    NSUInteger tg = gRoPE.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)total < tg) tg = (NSUInteger)total;
    [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

int mtl_recorder_finish(void* rec) {
    if (rec == NULL) return -2;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    [cmd commit];
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -4;
}

// mtl_recorder_commit submits without waiting — the SPEC T614 encode-overlap split: the host
// encodes the NEXT step's command buffer while this one executes. Pair with mtl_recorder_wait.
int mtl_recorder_commit(void* rec) {
    if (rec == NULL) return -2;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    [cmd commit];
    return 0;
}

// mtl_recorder_wait blocks until a mtl_recorder_commit'ed buffer completes.
int mtl_recorder_wait(void* rec) {
    if (rec == NULL) return -2;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    [cmd waitUntilCompleted];
    return cmd.status == MTLCommandBufferStatusCompleted ? 0 : -4;
}

void mtl_recorder_free(void* rec) {
    if (rec == NULL) return;
    id<MTLCommandBuffer> cmd = (__bridge_transfer id<MTLCommandBuffer>)rec; // ARC releases
    (void)cmd;
}

// ---- chained-op feasibility proof (§T369, ADR-0019 Phase 2) -----------------
// Records an MPS matmul C=A·B THEN a ReLU on C into ONE command buffer, with C a
// DEVICE-RESIDENT intermediate that never round-trips to the host between the two ops
// (only Out downloads at the end). Proves the record-mode chaining Phase 2 needs: MPS +
// a custom kernel in one command buffer over a device buffer, relying on Metal's default
// hazard tracking to order the MPS write before the ReLU read. Returns 0 on success.
int mtl_chain_matmul_relu(const float* A, const float* B, float* Out, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_unary() != 0) return -6;
    @autoreleasepool {
        id<MTLBuffer> aBuf = [gDevice newBufferWithBytes:A length:(size_t)M*K*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [gDevice newBufferWithBytes:B length:(size_t)K*N*sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [gDevice newBufferWithLength:(size_t)M*N*sizeof(float) options:MTLResourceStorageModeShared]; // device-resident intermediate
        if (aBuf == nil || bBuf == nil || cBuf == nil) return -2;

        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(size_t)K*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:(size_t)N*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(size_t)N*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:gDevice transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

        int total = M*N;
        int P[2] = {total, 4}; // op 4 = ReLU
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        // op 1: MPS matmul → cBuf (device-resident)
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        // op 2: ReLU on cBuf in place (reads cBuf, writes cBuf) — same command buffer.
        // Metal's default hazard tracking orders the MPS write before this read.
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gUnary];
        [enc setBuffer:cBuf offset:0 atIndex:0];
        [enc setBuffer:cBuf offset:0 atIndex:1];
        [enc setBuffer:pb offset:0 atIndex:2];
        NSUInteger tg = gUnary.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(Out, cBuf.contents, (size_t)M*N*sizeof(float));
        return 0;
    }
}

// ---- device-resident tensor primitive (§T367, ADR-0019 Phase 1) -------------
// A general device-resident buffer: upload nbytes into a retained Shared MTLBuffer
// (host-visible on UMA), read it back with download, free with devbuf_free. This is
// the storage primitive that the batched-decode path (Phase 2) will chain ops over
// without per-op host round-trips. Nothing uses it yet — Phase 1 is the primitive +
// its round-trip test only.
void* mtl_devbuf_upload(const void* data, int nbytes) {
    if (ensure_init() != 0) return NULL;
    id<MTLBuffer> buf = [gDevice newBufferWithBytes:data length:(size_t)nbytes options:MTLResourceStorageModeShared];
    if (buf == nil) return NULL;
    return (__bridge_retained void*)buf; // +1 retain transferred to the handle
}

int mtl_devbuf_download(void* handle, void* dst, int nbytes) {
    if (handle == NULL) return -2;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle; // not owned here
    if ((int)buf.length < nbytes) return -3;
    memcpy(dst, buf.contents, (size_t)nbytes);
    return 0;
}

// mtl_devbuf_upload_into overwrites an existing device buffer's contents (host→buffer memcpy).
// Reuses a buffer across decode steps (e.g. the new token's hidden) without realloc. Shared
// storage on UMA → the write is visible to the next command buffer.
int mtl_devbuf_upload_into(void* handle, const void* src, int nbytes) {
    if (handle == NULL) return -2;
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)handle; // not owned here
    if ((int)buf.length < nbytes) return -3;
    memcpy(buf.contents, src, (size_t)nbytes);
    return 0;
}

void mtl_devbuf_free(void* handle) {
    if (handle == NULL) return;
    id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)handle; // ARC releases
    (void)buf;
}

int mtl_qmatmul_resident(const float* X, void* wbuf, float* O, int M, int K, int N, int qtype) {
    if (ensure_init() != 0) return -1;
    if (wbuf == NULL) return -2;
    id<MTLComputePipelineState> pipe = nil;
    switch (qtype) {
        case 2:  if (ensure_qmatmul_q4_0() != 0) return -6; pipe = gQMatMulQ4_0; break;
        case 8:  if (ensure_qmatmul_q8()  != 0) return -6; pipe = gQMatMulQ8;  break;
        case 10: if (ensure_qmatmul_q2k() != 0) return -6; pipe = gQMatMulQ2K; break;
        case 11: if (ensure_qmatmul_q3k() != 0) return -6; pipe = gQMatMulQ3K; break;
        case 12: if (ensure_qmatmul_q4k() != 0) return -6; pipe = gQMatMulQ4K; break;
        case 13: if (ensure_qmatmul_q5k() != 0) return -6; pipe = gQMatMulQ5K; break;
        case 14: if (ensure_qmatmul_q6k() != 0) return -6; pipe = gQMatMulQ6K; break;
        default: return -7;
    }
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t oLen = (size_t)M * N * sizeof(float);
        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        id<MTLBuffer> wb = (__bridge id<MTLBuffer>)wbuf; // resident, not owned here
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:pipe];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// mtl_recorder_qmatmul (§T413): the record-mode QUANTIZED matmul — O = X·dequant(W)ᵀ encoded into
// the recorder's open command buffer over a DeviceBuffer X/O and a RESIDENT quantized weight
// (mtl_qweight_upload). Marries the §T156 resident qweights with the ADR-0019 batched decode so a
// quantized model's decode step can run as ONE command buffer. Same qtype codes as
// mtl_qmatmul_resident. No commit.
int mtl_recorder_qmatmul(void* rec, void* xh, void* wbuf, void* oh, int M, int K, int N, int qtype) {
    if (rec == NULL || xh == NULL || wbuf == NULL || oh == NULL) return -2;
    id<MTLComputePipelineState> pipe = nil;
    switch (qtype) {
        case 2:  if (ensure_qmatmul_q4_0() != 0) return -6; pipe = gQMatMulQ4_0; break;
        case 8:  if (ensure_qmatmul_q8()  != 0) return -6; pipe = gQMatMulQ8;  break;
        case 10: if (ensure_qmatmul_q2k() != 0) return -6; pipe = gQMatMulQ2K; break;
        case 11: if (ensure_qmatmul_q3k() != 0) return -6; pipe = gQMatMulQ3K; break;
        case 12: if (ensure_qmatmul_q4k() != 0) return -6; pipe = gQMatMulQ4K; break;
        case 13: if (ensure_qmatmul_q5k() != 0) return -6; pipe = gQMatMulQ5K; break;
        case 14: if (ensure_qmatmul_q6k() != 0) return -6; pipe = gQMatMulQ6K; break;
        default: return -7;
    }
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> xb = (__bridge id<MTLBuffer>)xh;
    id<MTLBuffer> wb = (__bridge id<MTLBuffer>)wbuf; // resident, not owned here
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[3] = {M, K, N};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    if (pb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:xb offset:0 atIndex:0];
    [enc setBuffer:wb offset:0 atIndex:1];
    [enc setBuffer:ob offset:0 atIndex:2];
    [enc setBuffer:pb offset:0 atIndex:3];
    int total = M * N;
    NSUInteger tg = pipe.maxTotalThreadsPerThreadgroup;
    if ((NSUInteger)total < tg) tg = (NSUInteger)total;
    [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    return 0;
}

// Q4_K quantized matmul: one thread per output (mi,ni) streams the ni-th weight row's K/256
// Q4_K super-blocks (144 B: f16 d, f16 dmin, scales[12], qs[128]), dequantizing in-kernel.
// Asymmetric affine y = d·sc6·nibble − dmin·min6; the 6-bit scale/min per 32-element sub-block
// come from the get_scale_min_k4 bit splice. Traversal mirrors dequantize_row_q4_K exactly:
// for each of 4 pairs, the low nibbles of 32 qs bytes give sub-block is+0, the high nibbles is+1.
static NSString* const kQMatMulQ4KSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q4k(device const float* X [[buffer(0)]],\n"
     "                        device const uchar* W [[buffer(1)]],\n"
     "                        device float* O [[buffer(2)]],\n"
     "                        constant int* P [[buffer(3)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nsb=K/256; int rowBytes=nsb*144; int rowOff=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int sb=0; sb<nsb; sb++){\n"
     "    int base=rowOff+sb*144;\n"
     "    ushort dr=(ushort)W[base] | ((ushort)W[base+1]<<8);\n"
     "    ushort dmr=(ushort)W[base+2] | ((ushort)W[base+3]<<8);\n"
     "    float d=float(as_type<half>(dr)); float dmin=float(as_type<half>(dmr));\n"
     "    int sbase=base+4; int qbase=base+16;\n"
     "    for (int pr=0; pr<4; pr++){\n"
     "      for (int hf=0; hf<2; hf++){\n"
     "        int is=pr*2+hf; int sc, mn;\n"
     "        if (is<4){ sc=W[sbase+is]&63; mn=W[sbase+is+4]&63; }\n"
     "        else { sc=(W[sbase+is+4]&0xF)|((W[sbase+is-4]>>6)<<4); mn=(W[sbase+is+4]>>4)|((W[sbase+is]>>6)<<4); }\n"
     "        float dl=d*(float)sc, ml=dmin*(float)mn;\n"
     "        int qb=qbase+pr*32; int xk=mi*K + sb*256 + pr*64 + hf*32;\n"
     "        for (int l=0; l<32; l++){ uchar qbyte=W[qb+l]; int nib=(hf==0)?(qbyte&0xF):(qbyte>>4); acc += X[xk+l]*(dl*(float)nib - ml); }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ4K = nil;

static int ensure_qmatmul_q4k(void) {
    if (gQMatMulQ4K != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ4KSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q4k"];
    if (fn == nil) return -6;
    gQMatMulQ4K = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ4K != nil ? 0 : -6;
}

int mtl_qmatmul_q4k(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q4k() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nsb = K / 256;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nsb * 144;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ4K];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ4K.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Q6_K quantized matmul: one thread per output (mi,ni) streams the ni-th weight row's K/256
// Q6_K super-blocks (210 B: ql[128], qh[64], scales[16] int8, f16 d), dequantizing in-kernel.
// SYMMETRIC y = d·scale·(q6−32); the 6-bit quant is assembled from a ql nibble + 2 qh high
// bits, the scale is a signed int8 per 16-element sub-block. Traversal mirrors
// dequantize_row_q6_K exactly (two 128-groups, four quants per l with scale indices +0/2/4/6).
static NSString* const kQMatMulQ6KSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q6k(device const float* X [[buffer(0)]],\n"
     "                        device const uchar* W [[buffer(1)]],\n"
     "                        device float* O [[buffer(2)]],\n"
     "                        constant int* P [[buffer(3)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nsb=K/256; int rowBytes=nsb*210; int rowOff=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int sb=0; sb<nsb; sb++){\n"
     "    int base=rowOff+sb*210;\n"
     "    int qlB=base; int qhB=base+128; int scB=base+192;\n"
     "    ushort dr=(ushort)W[base+208] | ((ushort)W[base+209]<<8);\n"
     "    float d=float(as_type<half>(dr));\n"
     "    for (int grp=0; grp<2; grp++){\n"
     "      int qlo=grp*64, qho=grp*32, sco=grp*8; int xk=mi*K + sb*256 + grp*128;\n"
     "      for (int l=0; l<32; l++){\n"
     "        int is=l/16;\n"
     "        int q1=(W[qlB+qlo+l]&0xF) | ((W[qhB+qho+l]&3)<<4);\n"
     "        int q2=(W[qlB+qlo+l+32]&0xF) | (((W[qhB+qho+l]>>2)&3)<<4);\n"
     "        int q3=(W[qlB+qlo+l]>>4) | (((W[qhB+qho+l]>>4)&3)<<4);\n"
     "        int q4=(W[qlB+qlo+l+32]>>4) | (((W[qhB+qho+l]>>6)&3)<<4);\n"
     "        float s0=(float)((char)W[scB+sco+is+0]), s2=(float)((char)W[scB+sco+is+2]);\n"
     "        float s4=(float)((char)W[scB+sco+is+4]), s6=(float)((char)W[scB+sco+is+6]);\n"
     "        acc += X[xk+l]    * d*s0*(float)(q1-32);\n"
     "        acc += X[xk+l+32] * d*s2*(float)(q2-32);\n"
     "        acc += X[xk+l+64] * d*s4*(float)(q3-32);\n"
     "        acc += X[xk+l+96] * d*s6*(float)(q4-32);\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ6K = nil;

static int ensure_qmatmul_q6k(void) {
    if (gQMatMulQ6K != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ6KSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q6k"];
    if (fn == nil) return -6;
    gQMatMulQ6K = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ6K != nil ? 0 : -6;
}

int mtl_qmatmul_q6k(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q6k() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nsb = K / 256;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nsb * 210;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ6K];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ6K.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Q5_K quantized matmul: Q4_K's asymmetric-affine kernel plus the qh high-bit plane. Block is
// 176 B (f16 d, f16 dmin, scales[12], qh[32], qs[128]); q5 = nibble | (qh bit)<<4 ∈ [0,31], and
// per pair pr the low-nibble sub-block uses qh bit u1=1<<2pr, the high-nibble uses u2=2<<2pr.
static NSString* const kQMatMulQ5KSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q5k(device const float* X [[buffer(0)]],\n"
     "                        device const uchar* W [[buffer(1)]],\n"
     "                        device float* O [[buffer(2)]],\n"
     "                        constant int* P [[buffer(3)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nsb=K/256; int rowBytes=nsb*176; int rowOff=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int sb=0; sb<nsb; sb++){\n"
     "    int base=rowOff+sb*176;\n"
     "    ushort dr=(ushort)W[base] | ((ushort)W[base+1]<<8);\n"
     "    ushort dmr=(ushort)W[base+2] | ((ushort)W[base+3]<<8);\n"
     "    float d=float(as_type<half>(dr)); float dmin=float(as_type<half>(dmr));\n"
     "    int sbase=base+4; int qhB=base+16; int qsB=base+48;\n"
     "    for (int pr=0; pr<4; pr++){\n"
     "      int u1=1<<(2*pr), u2=2<<(2*pr);\n"
     "      for (int hf=0; hf<2; hf++){\n"
     "        int is=pr*2+hf; int sc, mn;\n"
     "        if (is<4){ sc=W[sbase+is]&63; mn=W[sbase+is+4]&63; }\n"
     "        else { sc=(W[sbase+is+4]&0xF)|((W[sbase+is-4]>>6)<<4); mn=(W[sbase+is+4]>>4)|((W[sbase+is]>>6)<<4); }\n"
     "        float dl=d*(float)sc, ml=dmin*(float)mn; int um=(hf==0)?u1:u2;\n"
     "        int qb=qsB+pr*32; int xk=mi*K + sb*256 + pr*64 + hf*32;\n"
     "        for (int l=0; l<32; l++){ uchar qbyte=W[qb+l]; int nib=(hf==0)?(qbyte&0xF):(qbyte>>4); int hb=(W[qhB+l]&um)?16:0; acc += X[xk+l]*(dl*(float)(nib+hb) - ml); }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ5K = nil;

static int ensure_qmatmul_q5k(void) {
    if (gQMatMulQ5K != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ5KSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q5k"];
    if (fn == nil) return -6;
    gQMatMulQ5K = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ5K != nil ? 0 : -6;
}

int mtl_qmatmul_q5k(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q5k() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nsb = K / 256;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nsb * 176;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ5K];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ5K.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Q2_K quantized matmul: the simplest k-quant matmul. Block is 84 B (scales[16], qs[64],
// f16 d, f16 dmin); each sub-block's scale/min are the low/high nibbles of scales[is],
// y = d·(sc&0xF)·q2 − dmin·(sc>>4), q2 = (qs[l]>>shift)&3 ∈ [0,3]. Traversal mirrors
// dequantize_row_q2_K: two 128-groups, shift=2j (j=0..3), two 16-groups per j.
static NSString* const kQMatMulQ2KSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q2k(device const float* X [[buffer(0)]],\n"
     "                        device const uchar* W [[buffer(1)]],\n"
     "                        device float* O [[buffer(2)]],\n"
     "                        constant int* P [[buffer(3)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nsb=K/256; int rowBytes=nsb*84; int rowOff=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int sb=0; sb<nsb; sb++){\n"
     "    int base=rowOff+sb*84; int scB=base; int qsB=base+16;\n"
     "    ushort dr=(ushort)W[base+80] | ((ushort)W[base+81]<<8);\n"
     "    ushort dmr=(ushort)W[base+82] | ((ushort)W[base+83]<<8);\n"
     "    float d=float(as_type<half>(dr)); float dmin=float(as_type<half>(dmr));\n"
     "    for (int nb=0; nb<2; nb++){\n"
     "      int q0=qsB+nb*32;\n"
     "      for (int j=0; j<4; j++){\n"
     "        int shift=2*j;\n"
     "        for (int grp=0; grp<2; grp++){\n"
     "          int is=nb*8+j*2+grp; uchar sc=W[scB+is];\n"
     "          float dl=d*(float)(sc&0xF), ml=dmin*(float)(sc>>4);\n"
     "          int gOff=grp*16; int xk=mi*K + sb*256 + nb*128 + j*32 + grp*16;\n"
     "          for (int l=0; l<16; l++){ int q2=(W[q0+gOff+l]>>shift)&3; acc += X[xk+l]*(dl*(float)q2 - ml); }\n"
     "        }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ2K = nil;

static int ensure_qmatmul_q2k(void) {
    if (gQMatMulQ2K != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ2KSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q2k"];
    if (fn == nil) return -6;
    gQMatMulQ2K = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ2K != nil ? 0 : -6;
}

int mtl_qmatmul_q2k(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q2k() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nsb = K / 256;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nsb * 84;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ2K];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ2K.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Q3_K quantized matmul — the most intricate k-quant matmul. Block is 110 B (hmask[32],
// qs[64], scales[12], f16 d LAST). The 16 signed 6-bit sub-block scales are unpacked from the
// 12 packed bytes with the ggml aux/kmask splice (once per super-block); then y =
// d·(sc6−32)·(q3−4), q3 = 2 low bits (qs) | 1 high bit (hmask)<<2, the hmask arithmetic
// INVERTED (bit clear → subtract 4). m doubles per inner j (hmask bits 0..7 over the block).
static NSString* const kQMatMulQ3KSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void qmatmul_q3k(device const float* X [[buffer(0)]],\n"
     "                        device const uchar* W [[buffer(1)]],\n"
     "                        device float* O [[buffer(2)]],\n"
     "                        constant int* P [[buffer(3)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int M=P[0], K=P[1], N=P[2];\n"
     "  if ((int)gid >= M*N) return;\n"
     "  int mi=gid/N, ni=gid%N;\n"
     "  int nsb=K/256; int rowBytes=nsb*110; int rowOff=ni*rowBytes;\n"
     "  float acc=0.0f;\n"
     "  for (int sb=0; sb<nsb; sb++){\n"
     "    int base=rowOff+sb*110; int hmB=base; int qsB=base+32; int scB=base+96;\n"
     "    ushort dr=(ushort)W[base+108] | ((ushort)W[base+109]<<8);\n"
     "    float dAll=float(as_type<half>(dr));\n"
     "    uint aux0=(uint)W[scB]|((uint)W[scB+1]<<8)|((uint)W[scB+2]<<16)|((uint)W[scB+3]<<24);\n"
     "    uint aux1=(uint)W[scB+4]|((uint)W[scB+5]<<8)|((uint)W[scB+6]<<16)|((uint)W[scB+7]<<24);\n"
     "    uint aux2=(uint)W[scB+8]|((uint)W[scB+9]<<8)|((uint)W[scB+10]<<16)|((uint)W[scB+11]<<24);\n"
     "    uint tmp=aux2;\n"
     "    uint a2=((aux0>>4)&0x0f0f0f0fu)|(((tmp>>4)&0x03030303u)<<4);\n"
     "    uint a3=((aux1>>4)&0x0f0f0f0fu)|(((tmp>>6)&0x03030303u)<<4);\n"
     "    uint a0=(aux0&0x0f0f0f0fu)|(((tmp>>0)&0x03030303u)<<4);\n"
     "    uint a1=(aux1&0x0f0f0f0fu)|(((tmp>>2)&0x03030303u)<<4);\n"
     "    int sc[16];\n"
     "    sc[0]=(int)(a0&0xFFu); sc[1]=(int)((a0>>8)&0xFFu); sc[2]=(int)((a0>>16)&0xFFu); sc[3]=(int)((a0>>24)&0xFFu);\n"
     "    sc[4]=(int)(a1&0xFFu); sc[5]=(int)((a1>>8)&0xFFu); sc[6]=(int)((a1>>16)&0xFFu); sc[7]=(int)((a1>>24)&0xFFu);\n"
     "    sc[8]=(int)(a2&0xFFu); sc[9]=(int)((a2>>8)&0xFFu); sc[10]=(int)((a2>>16)&0xFFu); sc[11]=(int)((a2>>24)&0xFFu);\n"
     "    sc[12]=(int)(a3&0xFFu); sc[13]=(int)((a3>>8)&0xFFu); sc[14]=(int)((a3>>16)&0xFFu); sc[15]=(int)((a3>>24)&0xFFu);\n"
     "    int is=0;\n"
     "    for (int nb=0; nb<2; nb++){\n"
     "      int qb=qsB+nb*32;\n"
     "      for (int j=0; j<4; j++){\n"
     "        int shift=2*j; int mbit=1<<(nb*4+j);\n"
     "        for (int grp=0; grp<2; grp++){\n"
     "          float dl=dAll*(float)(sc[is]-32); is++;\n"
     "          int gOff=grp*16; int xk=mi*K + sb*256 + nb*128 + j*32 + grp*16;\n"
     "          for (int l=0; l<16; l++){ int q3=(W[qb+gOff+l]>>shift)&3; int h=(W[hmB+gOff+l]&mbit)?0:4; acc += X[xk+l]*dl*(float)(q3-h); }\n"
     "        }\n"
     "      }\n"
     "    }\n"
     "  }\n"
     "  O[mi*N+ni]=acc;\n"
     "}\n";

static id<MTLComputePipelineState> gQMatMulQ3K = nil;

static int ensure_qmatmul_q3k(void) {
    if (gQMatMulQ3K != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kQMatMulQ3KSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"qmatmul_q3k"];
    if (fn == nil) return -6;
    gQMatMulQ3K = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gQMatMulQ3K != nil ? 0 : -6;
}

int mtl_qmatmul_q3k(const float* X, const unsigned char* W, float* O, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    if (ensure_qmatmul_q3k() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int nsb = K / 256;
        size_t xLen = (size_t)M * K * sizeof(float);
        size_t wLen = (size_t)N * nsb * 110;
        size_t oLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[3] = {M, K, N};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || pb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gQMatMulQ3K];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:pb offset:0 atIndex:3];
        int total = M * N;
        NSUInteger tg = gQMatMulQ3K.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, oLen);
        return 0;
    }
}

// Fused SDPA compute kernel, compiled once. One thread per (head, query row);
// two passes (max, then sum+weighted-accumulate) for a numerically stable softmax
// that mirrors the reference kernel. dk is bounded at 128 so the per-thread output
// accumulator is a fixed-size register array.
static NSString* const kMHASource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void mha_f32(device const float* Q [[buffer(0)]],\n"
     "                    device const float* K [[buffer(1)]],\n"
     "                    device const float* V [[buffer(2)]],\n"
     "                    device float* O [[buffer(3)]],\n"
     "                    constant int* P [[buffer(4)]],\n"
     "                    constant float* FP [[buffer(5)]],\n"
     "                    uint gid [[thread_position_in_grid]]) {\n"
     "  int sq=P[0], sk=P[1], dm=P[2], heads=P[3], kvHeads=P[4], dk=P[5], causal=P[6], window=P[7];\n"
     "  float scale=FP[0];\n"
     "  int total=heads*sq;\n"
     "  if ((int)gid>=total) return;\n"
     "  int h=gid/sq, i=gid%sq;\n"
     "  int rep=heads/kvHeads, dkv=kvHeads*dk;\n"
     "  int qOff=h*dk, kvOff=(h/rep)*dk, off=sk-sq;\n"
     "  int jmax = causal ? (off+i+1) : sk;\n"
     "  int jmin = (window>0 && off+i-window+1>0) ? (off+i-window+1) : 0;\n"
     "  float m=-INFINITY;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0.0f; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; s*=scale; if(s>m) m=s; }\n"
     "  float sum=0.0f; float acc[128]; for(int d=0;d<dk;d++) acc[d]=0.0f;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0.0f; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; s*=scale; float e=exp(s-m); sum+=e; for(int d=0;d<dk;d++) acc[d]+=e*V[j*dkv+kvOff+d]; }\n"
     "  for(int d=0;d<dk;d++) O[i*dm+qOff+d]=acc[d]/sum;\n"
     "}\n";

static id<MTLComputePipelineState> gMHA = nil;

static int ensure_mha(void) {
    if (gMHA != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kMHASource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"mha_f32"];
    if (fn == nil) return -11;
    gMHA = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gMHA != nil ? 0 : -12;
}

int mtl_mha_f32(const float* Q, const float* K, const float* V, float* O,
                int sq, int sk, int dm, int heads, int kvHeads, int dk,
                int causal, int window, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_mha() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        int dkv = kvHeads * dk;
        size_t qLen = (size_t)sq * dm * sizeof(float);
        size_t kvLen = (size_t)sk * dkv * sizeof(float);

        id<MTLBuffer> qb = pool_in(Q, qLen);
        id<MTLBuffer> kb = pool_in(K, kvLen);
        id<MTLBuffer> vb = pool_in(V, kvLen);
        id<MTLBuffer> ob = pool_buf(qLen);
        int P[8] = {sq, sk, dm, heads, kvHeads, dk, causal, window};
        float FP[1] = {scale};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (qb == nil || kb == nil || vb == nil || ob == nil || pb == nil || fpb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gMHA];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        [enc setBuffer:fpb offset:0 atIndex:5];
        int total = heads * sq;
        // 64 threads/threadgroup, NOT maxTotalThreadsPerThreadgroup: this kernel keeps a
        // float acc[128] register array per thread, and large threadgroups collapse
        // occupancy (measured 2.2× slower than the identical 64-wide Vulkan kernel). §T337
        NSUInteger tg = 64;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, qLen);
        return 0;
    }
}

// mtl_recorder_mha encodes O = attention(Q,K,V) over device buffers into the recorder's
// command buffer, with SEPARATE query/key lengths (sq queries against sk cached keys) — the
// decode-against-KV-cache shape (sq=1, sk=cacheLen). Q,O are [sq,dm]; K,V are [sk,kvHeads*dk].
// One thread per (head, query-row); causal aligns via off=sk-sq. No commit.
// mha_decode_f32 (§T428): COOPERATIVE single-query attention for the batched decoder. The two-pass
// mha_f32 kernel runs ONE thread per (head, query row) — at decode (sq=1) that is `heads` threads on
// the whole GPU, each streaming all sk cached keys serially, so step time exploded linearly with
// context (242ms @L=1920 vs a <1ms floor). Here ONE SIMDGROUP (32 lanes) works per head: lanes
// partition the keys, each keeps an online-softmax partial (m, l, acc[dk]) in registers, and the
// combine uses simd_max/simd_shuffle_down (the standard (m,l,acc) merge) — no threadgroup memory,
// no atomics. dk ≤ 128.
static NSString* const kMHADecodeSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void mha_decode_f32(device const float* Q [[buffer(0)]],\n"
     "                           device const float* K [[buffer(1)]],\n"
     "                           device const float* V [[buffer(2)]],\n"
     "                           device float* O [[buffer(3)]],\n"
     "                           constant int* P [[buffer(4)]],\n"
     "                           constant float* FP [[buffer(5)]],\n"
     "                           uint2 tg [[threadgroup_position_in_grid]],\n"
     "                           uint lane [[thread_index_in_simdgroup]]) {\n"
     "  int sq=P[0], sk=P[1], dm=P[2], heads=P[3], kvHeads=P[4], dk=P[5], causal=P[6];\n"
     "  float scale=FP[0];\n"
     "  int h=(int)tg.x, i=(int)tg.y;\n"
     "  if (h >= heads || i >= sq) return;\n"
     "  int rep=heads/kvHeads, dkv=kvHeads*dk;\n"
     "  int qOff=h*dk, kvOff=(h/rep)*dk;\n"
     "  int jmax = causal ? (sk - sq + i + 1) : sk;\n"
     "  float q[128]; for (int d=0; d<dk; d++) q[d]=Q[i*dm+qOff+d];\n"
     "  float m=-INFINITY, l=0.0f; float acc[128]; for (int d=0; d<dk; d++) acc[d]=0.0f;\n"
     "  for (int j=(int)lane; j<jmax; j+=32) {\n"
     "    float s=0.0f; for (int d=0; d<dk; d++) s+=q[d]*K[j*dkv+kvOff+d]; s*=scale;\n"
     "    float mNew=max(m,s); float corr=exp(m-mNew); float p=exp(s-mNew);\n"
     "    l=corr*l+p; for (int d=0; d<dk; d++) acc[d]=corr*acc[d]+p*V[j*dkv+kvOff+d]; m=mNew;\n"
     "  }\n"
     "  // simdgroup merge of the (m, l, acc) online-softmax partials (tree over 32 lanes).\n"
     "  for (uint off=16; off>0; off>>=1) {\n"
     "    float mo=simd_shuffle_down(m,off); float lo=simd_shuffle_down(l,off);\n"
     "    float M=max(m,mo);\n"
     "    // guard the both-empty case: (-inf)-(-inf) = NaN would poison the merge (sk<32 lanes).\n"
     "    float c1=(M==-INFINITY)?0.0f:exp(m-M), c2=(M==-INFINITY)?0.0f:exp(mo-M);\n"
     "    for (int d=0; d<dk; d++) { float ao=simd_shuffle_down(acc[d],off); acc[d]=c1*acc[d]+c2*ao; }\n"
     "    l=c1*l+c2*lo; m=M;\n"
     "  }\n"
     "  if (lane==0) { for (int d=0; d<dk; d++) O[i*dm+qOff+d]=acc[d]/l; }\n"
     "}\n";

static id<MTLComputePipelineState> gMHADecode = nil;
static int ensure_mha_decode(void) {
    if (gMHADecode != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kMHADecodeSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"mha_decode_f32"];
    if (fn == nil) return -11;
    gMHADecode = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gMHADecode != nil ? 0 : -12;
}

// mtl_mha_decode_host (§T432): the cooperative decode/prefill attention (§T428/431 kernel) over
// HOST slices — upload, one dispatch, download — so the PER-OP backend path (nlp DecodeStep etc.)
// gets the same fix as the recorder: the two-pass kernel is ~serial in the context length at small
// sq. Q,O are [sq,dm]; K,V are [sk,kvHeads·dk]; causal row bound jmax=sk-sq+i+1.
int mtl_mha_decode_host(const float* Q, const float* K, const float* V, float* O,
                        int sq, int sk, int dm, int heads, int kvHeads, int dk,
                        int causal, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_mha_decode() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        int dkv = kvHeads * dk;
        size_t qLen = (size_t)sq * dm * sizeof(float);
        size_t kvLen = (size_t)sk * dkv * sizeof(float);
        id<MTLBuffer> qb = pool_in(Q, qLen);
        id<MTLBuffer> kb = pool_in(K, kvLen);
        id<MTLBuffer> vb = pool_in(V, kvLen);
        id<MTLBuffer> ob = pool_buf(qLen);
        int P[7] = {sq, sk, dm, heads, kvHeads, dk, causal};
        float FP[1] = {scale};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (qb == nil || kb == nil || vb == nil || ob == nil || pb == nil || fpb == nil) return -2;
        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gMHADecode];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        [enc setBuffer:fpb offset:0 atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(heads,sq,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
        memcpy(O, ob.contents, qLen);
        return 0;
    }
}

int mtl_recorder_mha(void* rec, void* qh, void* kh, void* vh, void* oh,
                     int sq, int sk, int dm, int heads, int kvHeads, int dk,
                     int causal, int window, float scale, int qElemOff) {
    if (rec == NULL || qh == NULL || kh == NULL || vh == NULL || oh == NULL || qElemOff < 0) return -2;
    // Cooperative fast path (§T428 decode, §T431 generalized to sq>1 for prefill windows): one
    // 32-lane simdgroup per (query row, head) — sq·heads simdgroups vs the two-pass kernel's
    // sq·heads single THREADS, whose serial key streaming made long-context prefill windows
    // ~12× slower (§T431: 291ms @pos1792). causal row bound jmax = sk-sq+i+1; window stays on
    // the two-pass kernel. At sq==1 causal is irrelevant, so non-causal decode also routes here.
    if (window == 0 && dk <= 128 && (causal != 0 || sq == 1)) {
        if (ensure_mha_decode() != 0) return -5;
        id<MTLCommandBuffer> dcmd = (__bridge id<MTLCommandBuffer>)rec;
        id<MTLBuffer> dqb = (__bridge id<MTLBuffer>)qh;
        id<MTLBuffer> dkb = (__bridge id<MTLBuffer>)kh;
        id<MTLBuffer> dvb = (__bridge id<MTLBuffer>)vh;
        id<MTLBuffer> dob = (__bridge id<MTLBuffer>)oh;
        int DP[7] = {sq, sk, dm, heads, kvHeads, dk, causal};
        float DFP[1] = {scale};
        id<MTLBuffer> dpb = [gDevice newBufferWithBytes:DP length:sizeof(DP) options:MTLResourceStorageModeShared];
        id<MTLBuffer> dfpb = [gDevice newBufferWithBytes:DFP length:sizeof(DFP) options:MTLResourceStorageModeShared];
        if (dpb == nil || dfpb == nil) return -2;
        id<MTLComputeCommandEncoder> denc = [dcmd computeCommandEncoder];
        [denc setComputePipelineState:gMHADecode];
        // qElemOff: float-element offset into Q (the fused-QKV sub-row view, §T613).
        [denc setBuffer:dqb offset:(NSUInteger)qElemOff*4 atIndex:0];
        [denc setBuffer:dkb offset:0 atIndex:1];
        [denc setBuffer:dvb offset:0 atIndex:2];
        [denc setBuffer:dob offset:0 atIndex:3];
        [denc setBuffer:dpb offset:0 atIndex:4];
        [denc setBuffer:dfpb offset:0 atIndex:5];
        [denc dispatchThreadgroups:MTLSizeMake(heads,sq,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
        [denc endEncoding];
        return 0;
    }
    if (ensure_mha() != 0) return -5;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> qb = (__bridge id<MTLBuffer>)qh;
    id<MTLBuffer> kb = (__bridge id<MTLBuffer>)kh;
    id<MTLBuffer> vb = (__bridge id<MTLBuffer>)vh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[8] = {sq, sk, dm, heads, kvHeads, dk, causal, window};
    float FP[1] = {scale};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fpb = [gDevice newBufferWithBytes:FP length:sizeof(FP) options:MTLResourceStorageModeShared];
    if (pb == nil || fpb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gMHA];
    [enc setBuffer:qb offset:(NSUInteger)qElemOff*4 atIndex:0];
    [enc setBuffer:kb offset:0 atIndex:1];
    [enc setBuffer:vb offset:0 atIndex:2];
    [enc setBuffer:ob offset:0 atIndex:3];
    [enc setBuffer:pb offset:0 atIndex:4];
    [enc setBuffer:fpb offset:0 atIndex:5];
    int total = heads * sq;
    NSUInteger tg = 64; // §T337: acc[128] register array — large threadgroups collapse occupancy
    if ((NSUInteger)total < tg) tg = (NSUInteger)total;
    [enc dispatchThreads:MTLSizeMake(total,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
    [enc endEncoding];
    return 0;
}

// FlashAttention-2 forward kernel (Dao 2023, §T109). TILED (§T349): a threadgroup of
// FTG query rows of ONE head cooperatively streams K/V in tiles of FTILE keys through
// threadgroup memory (sK/sV), so each K/V element is read from GLOBAL memory once per
// threadgroup instead of once per query row — the flash kernel ran at ~5% of peak,
// bandwidth-bound on those re-reads (§V22-measured). The PER-QUERY online-softmax
// recurrence (running max m, normalizer l, register acc[dk]) is UNCHANGED from the
// one-thread-per-row kernel, so the result is identical up to FP reassociation
// (V-CROSS §V3/§V11). Grid: threadgroups (ceil(seq/FTG), heads), FTG threads each.
// Threads whose query row ≥ seq stay active for the cooperative loads/barriers (uniform
// control flow) but write no output. dk bounded at 128; threadgroup tile is 16 KiB.
// corr=exp(m_old−m_new) rescales acc when the max grows; first key m=−INF → corr=0.
static NSString* const kFlashAttnSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "#define FTILE 16\n"
     "kernel void flashattn_f32(device const float* Q [[buffer(0)]],\n"
     "                          device const float* K [[buffer(1)]],\n"
     "                          device const float* V [[buffer(2)]],\n"
     "                          device float* O [[buffer(3)]],\n"
     "                          constant int* P [[buffer(4)]],\n"
     "                          constant float* FP [[buffer(5)]],\n"
     "                          uint3 tgpos [[threadgroup_position_in_grid]],\n"
     "                          uint3 tid3 [[thread_position_in_threadgroup]],\n"
     "                          uint3 ntg3 [[threads_per_threadgroup]]) {\n"
     "  int seq=P[0], dm=P[1], heads=P[2], dk=P[3], causal=P[4], kvHeads=P[5];\n"
     "  float scale=FP[0];\n"
     "  uint tid=tid3.x, ntg=ntg3.x;\n"
     "  int h=(int)tgpos.y; int i=(int)(tgpos.x*ntg + tid);\n"
     "  int rep=heads/kvHeads, dkv=kvHeads*dk, qOff=h*dk, kvOff=(h/rep)*dk;\n"
     "  bool active = i < seq;\n"
     "  int jmax = (active ? (causal ? (i+1) : seq) : 0);\n"
     "  threadgroup float sK[FTILE*128]; threadgroup float sV[FTILE*128];\n"
     "  float q[128]; if (active) for(int d=0;d<dk;d++) q[d]=Q[i*dm+qOff+d];\n"
     "  float m=-INFINITY, l=0.0f; float acc[128]; for(int d=0;d<dk;d++) acc[d]=0.0f;\n"
     "  int blockMaxJ = causal ? min(seq, (int)(tgpos.x*ntg + ntg)) : seq;\n"
     "  for (int t0=0; t0<blockMaxJ; t0+=FTILE){\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    for (int e=(int)tid; e<FTILE*dk; e+=(int)ntg){ int t=e/dk, d=e%dk; int j=t0+t;\n"
     "      float kv0=0.0f, vv0=0.0f; if (j<seq){ kv0=K[j*dkv+kvOff+d]; vv0=V[j*dkv+kvOff+d]; }\n"
     "      sK[e]=kv0; sV[e]=vv0; }\n"
     "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "    int thi = min(FTILE, jmax-t0);\n"
     "    for (int t=0;t<thi;t++){ int j=t0+t; if(j>=jmax) break;\n"
     "      float s=0.0f; for(int d=0;d<dk;d++) s+=q[d]*sK[t*dk+d]; s*=scale;\n"
     "      float mNew=max(m,s); float corr=exp(m-mNew); float p=exp(s-mNew);\n"
     "      l=corr*l+p; for(int d=0;d<dk;d++) acc[d]=corr*acc[d]+p*sV[t*dk+d]; m=mNew;\n"
     "    }\n"
     "  }\n"
     "  if (active) for(int d=0;d<dk;d++) O[i*dm+qOff+d]=acc[d]/l;\n"
     "}\n";

static id<MTLComputePipelineState> gFlashAttn = nil;

static int ensure_flashattn(void) {
    if (gFlashAttn != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kFlashAttnSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"flashattn_f32"];
    if (fn == nil) return -11;
    gFlashAttn = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gFlashAttn != nil ? 0 : -12;
}

int mtl_flashattn_f32(const float* Q, const float* K, const float* V, float* O,
                      int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_flashattn() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        size_t len = (size_t)seq * dm * sizeof(float);
        size_t kvLen = (size_t)seq * kvHeads * dk * sizeof(float); // K,V are [seq,kvHeads*dk]
        id<MTLBuffer> qb = pool_in(Q, len);
        id<MTLBuffer> kb = pool_in(K, kvLen);
        id<MTLBuffer> vb = pool_in(V, kvLen);
        id<MTLBuffer> ob = pool_buf(len);
        int P[6] = {seq, dm, heads, dk, causal, kvHeads};
        float FP[1] = {scale};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (qb == nil || kb == nil || vb == nil || ob == nil || pb == nil || fpb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gFlashAttn];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        [enc setBuffer:fpb offset:0 atIndex:5];
        // Tiled (§T349): threadgroup = FTG query rows of one head; grid (ceil(seq/FTG), heads).
        // FTG=64 matches the register-heavy §T337 guidance (acc[dk] + q[dk] per thread).
        NSUInteger ftg = 64;
        NSUInteger gx = ((NSUInteger)seq + ftg - 1) / ftg;
        [enc dispatchThreadgroups:MTLSizeMake(gx, (NSUInteger)heads, 1) threadsPerThreadgroup:MTLSizeMake(ftg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, len);
        return 0;
    }
}

// mtl_recorder_flashattn encodes O = flash-attention(Q,K,V) over device buffers into the
// recorder's command buffer. Q,O are [seq,dm]; K,V are [seq,kvHeads*dk]. Tiled kernel (§T349):
// threadgroups (ceil(seq/64), heads), 64 threads. No commit.
int mtl_recorder_flashattn(void* rec, void* qh, void* kh, void* vh, void* oh,
                           int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale) {
    if (rec == NULL || qh == NULL || kh == NULL || vh == NULL || oh == NULL) return -2;
    if (ensure_flashattn() != 0) return -5;
    id<MTLCommandBuffer> cmd = (__bridge id<MTLCommandBuffer>)rec;
    id<MTLBuffer> qb = (__bridge id<MTLBuffer>)qh;
    id<MTLBuffer> kb = (__bridge id<MTLBuffer>)kh;
    id<MTLBuffer> vb = (__bridge id<MTLBuffer>)vh;
    id<MTLBuffer> ob = (__bridge id<MTLBuffer>)oh;
    int P[6] = {seq, dm, heads, dk, causal, kvHeads};
    float FP[1] = {scale};
    id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];
    id<MTLBuffer> fpb = [gDevice newBufferWithBytes:FP length:sizeof(FP) options:MTLResourceStorageModeShared];
    if (pb == nil || fpb == nil) return -2;
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:gFlashAttn];
    [enc setBuffer:qb offset:0 atIndex:0];
    [enc setBuffer:kb offset:0 atIndex:1];
    [enc setBuffer:vb offset:0 atIndex:2];
    [enc setBuffer:ob offset:0 atIndex:3];
    [enc setBuffer:pb offset:0 atIndex:4];
    [enc setBuffer:fpb offset:0 atIndex:5];
    NSUInteger ftg = 64;
    NSUInteger gx = ((NSUInteger)seq + ftg - 1) / ftg;
    [enc dispatchThreadgroups:MTLSizeMake(gx, (NSUInteger)heads, 1) threadsPerThreadgroup:MTLSizeMake(ftg, 1, 1)];
    [enc endEncoding];
    return 0;
}

// softmax_causal (§T394): row softmax of the [rows,cols] score matrix S IN PLACE, with an optional
// causal mask (col j > row i → excluded). One 256-thread threadgroup per row (cooperative, §T346).
// Masked-out entries are zeroed so the subsequent S·V ignores them. Scale is pre-folded into S.
static NSString* const kSoftmaxCausalSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void softmax_causal(device float* S [[buffer(0)]],\n"
     "                           constant int* P [[buffer(1)]],\n"
     "                           uint row [[threadgroup_position_in_grid]],\n"
     "                           uint lane [[thread_position_in_threadgroup]],\n"
     "                           uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=P[0], cols=P[1], causal=P[2];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*cols;\n"
     "  int jmax = causal ? ((int)row + 1) : cols;\n"
     "  threadgroup float red[256];\n"
     "  float m=-INFINITY;\n"
     "  for (int j=(int)lane;j<jmax;j+=(int)tgsize) m=max(m,S[base+j]);\n"
     "  red[lane]=m; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2;s>0;s>>=1){ if(lane<s) red[lane]=max(red[lane],red[lane+s]); threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  m=red[0];\n"
     "  float sum=0.0f;\n"
     "  for (int j=(int)lane;j<jmax;j+=(int)tgsize){ float e=exp(S[base+j]-m); S[base+j]=e; sum+=e; }\n"
     "  red[lane]=sum; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2;s>0;s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float inv=1.0f/red[0];\n"
     "  for (int j=(int)lane;j<cols;j+=(int)tgsize){ S[base+j] = (j<jmax) ? S[base+j]*inv : 0.0f; }\n"
     "}\n";

static id<MTLComputePipelineState> gSoftmaxCausal = nil;
static int ensure_softmax_causal(void) {
    if (gSoftmaxCausal != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSoftmaxCausalSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"softmax_causal"];
    if (fn == nil) return -11;
    gSoftmaxCausal = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gSoftmaxCausal != nil ? 0 : -12;
}

// mtl_mha_mps (§T394): attention forward as MPS(Q·Kᵀ)→softmax(causal)→MPS(P·V), the reformulation
// §T393 showed is ~15× faster than the hand-written flash kernel (the two matmuls alone are 0.71ms
// vs flash's 11ms). Per head h: S = (Q_h·K_hᵀ)·scale via strided MPSMatrix views (column offset
// h·dk, rowBytes = full row) → causal softmax in place → O_h = S·V_h. One [seq,seq] scratch S is
// reused across heads; Metal hazard tracking orders each head's P·V read before the next Q·Kᵀ write.
// Q,O are [seq,dm]; K,V are [seq,kvHeads·dk]. GQA: kv head = h/(heads/kvHeads).
int mtl_mha_mps(const float* Q, const float* K, const float* V, float* O,
                int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_softmax_causal() != 0) return -5;
    @autoreleasepool {
        int kvDim = kvHeads * dk;
        int rep = heads / kvHeads;
        size_t qLen = (size_t)seq * dm * sizeof(float);
        size_t kvLen = (size_t)seq * kvDim * sizeof(float);
        size_t sLen = (size_t)seq * seq * sizeof(float);
        id<MTLBuffer> qb = [gDevice newBufferWithBytes:Q length:qLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> kb = [gDevice newBufferWithBytes:K length:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb = [gDevice newBufferWithBytes:V length:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> ob = [gDevice newBufferWithLength:qLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb = [gDevice newBufferWithLength:sLen options:MTLResourceStorageModeShared];
        if (qb == nil || kb == nil || vb == nil || ob == nil || sb == nil) return -2;

        int P[3] = {seq, seq, causal};
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P length:sizeof(P) options:MTLResourceStorageModeShared];

        // strided descriptors: a head is a dk-column window of the full [seq,·] row.
        MPSMatrixDescriptor* qDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:dk rowBytes:(size_t)dm*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* kvDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:dk rowBytes:(size_t)kvDim*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* sDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:seq rowBytes:(size_t)seq*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixMultiplication* mmQK = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
            transposeLeft:NO transposeRight:YES resultRows:seq resultColumns:seq interiorColumns:dk alpha:scale beta:0.0];
        MPSMatrixMultiplication* mmPV = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice
            transposeLeft:NO transposeRight:NO resultRows:seq resultColumns:dk interiorColumns:seq alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        for (int h = 0; h < heads; h++) {
            int kvh = h / rep;
            MPSMatrix* mQ = [[MPSMatrix alloc] initWithBuffer:qb offset:(size_t)h*dk*sizeof(float) descriptor:qDesc];
            MPSMatrix* mK = [[MPSMatrix alloc] initWithBuffer:kb offset:(size_t)kvh*dk*sizeof(float) descriptor:kvDesc];
            MPSMatrix* mV = [[MPSMatrix alloc] initWithBuffer:vb offset:(size_t)kvh*dk*sizeof(float) descriptor:kvDesc];
            MPSMatrix* mO = [[MPSMatrix alloc] initWithBuffer:ob offset:(size_t)h*dk*sizeof(float) descriptor:qDesc];
            MPSMatrix* mS = [[MPSMatrix alloc] initWithBuffer:sb offset:0 descriptor:sDesc];
            // S = scale · Q_h · K_hᵀ
            [mmQK encodeToCommandBuffer:cmd leftMatrix:mQ rightMatrix:mK resultMatrix:mS];
            // softmax(S) row-wise with causal mask, in place
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:gSoftmaxCausal];
            [enc setBuffer:sb offset:0 atIndex:0];
            [enc setBuffer:pb offset:0 atIndex:1];
            [enc dispatchThreadgroups:MTLSizeMake(seq, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc endEncoding];
            // O_h = S · V_h
            [mmPV encodeToCommandBuffer:cmd leftMatrix:mS rightMatrix:mV resultMatrix:mO];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
        memcpy(O, ob.contents, qLen);
        return 0;
    }
}

// softmax_jacobian (§T399): given P=softmax(S) and dP (both [rows,cols]), compute the softmax VJP
// dS[i,j] = P[i,j]·(dP[i,j] − Σ_k P[i,k]·dP[i,k]) IN PLACE on dP (dS overwrites dP). Causal: only
// j≤i valid (P is already 0 for j>i, so masked terms drop out; the tail is set to 0). One
// 256-thread threadgroup per row (cooperative Σ). §T399 attention-backward via MPS matmuls.
static NSString* const kSoftmaxJacobianSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void softmax_jacobian(device const float* P [[buffer(0)]],\n"
     "                             device float* dP [[buffer(1)]],\n"
     "                             constant int* PP [[buffer(2)]],\n"
     "                             uint row [[threadgroup_position_in_grid]],\n"
     "                             uint lane [[thread_position_in_threadgroup]],\n"
     "                             uint tgsize [[threads_per_threadgroup]]) {\n"
     "  int rows=PP[0], cols=PP[1], causal=PP[2];\n"
     "  if ((int)row >= rows) return;\n"
     "  int base=(int)row*cols;\n"
     "  int jmax = causal ? ((int)row + 1) : cols;\n"
     "  threadgroup float red[256];\n"
     "  float part=0.0f;\n"
     "  for (int j=(int)lane;j<jmax;j+=(int)tgsize) part += P[base+j]*dP[base+j];\n"
     "  red[lane]=part; threadgroup_barrier(mem_flags::mem_threadgroup);\n"
     "  for (uint s=tgsize/2;s>0;s>>=1){ if(lane<s) red[lane]+=red[lane+s]; threadgroup_barrier(mem_flags::mem_threadgroup); }\n"
     "  float delta=red[0];\n"
     "  for (int j=(int)lane;j<cols;j+=(int)tgsize){ dP[base+j] = (j<jmax) ? P[base+j]*(dP[base+j]-delta) : 0.0f; }\n"
     "}\n";

static id<MTLComputePipelineState> gSoftmaxJacobian = nil;
static int ensure_softmax_jacobian(void) {
    if (gSoftmaxJacobian != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kSoftmaxJacobianSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"softmax_jacobian"];
    if (fn == nil) return -11;
    gSoftmaxJacobian = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gSoftmaxJacobian != nil ? 0 : -12;
}

// mtl_mha_backward_mps (§T399): the attention backward as MPS matmuls — the backward analog of
// §T394's forward. §T399 measured the hand flash-backward at 70ms (54× the 1.3ms forward), the
// training-step bottleneck. Per head: recompute S=scale·Q·Kᵀ → P=softmax(S) → dV=Pᵀ·dO ,
// dP=dO·Vᵀ → dS=softmax_jacobian(P,dP) → dQ=scale·dS·K , dK=scale·dSᵀ·Q — all MPS over strided
// head views + one softmax + one jacobian, in one command buffer (S,dP scratch reused, hazard
// tracking orders). MHA only (kvHeads==heads, window==0); GQA/window fall back to the flash-backward.
int mtl_mha_backward_mps(const float* Q, const float* K, const float* V, const float* dO,
                         float* dQ, float* dK, float* dV,
                         int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_softmax_causal() != 0) return -5;
    if (ensure_softmax_jacobian() != 0) return -6;
    @autoreleasepool {
        int kvDim = kvHeads * dk;
        int rep = heads / kvHeads; // query heads per kv head (GQA); 1 for standard MHA
        size_t qLen = (size_t)seq * dm * sizeof(float);
        size_t kvLen = (size_t)seq * kvDim * sizeof(float);
        size_t sLen = (size_t)seq * seq * sizeof(float);
        id<MTLBuffer> qb  = [gDevice newBufferWithBytes:Q  length:qLen  options:MTLResourceStorageModeShared];
        id<MTLBuffer> kb  = [gDevice newBufferWithBytes:K  length:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> vb  = [gDevice newBufferWithBytes:V  length:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> dob = [gDevice newBufferWithBytes:dO length:qLen  options:MTLResourceStorageModeShared];
        id<MTLBuffer> dqb = [gDevice newBufferWithLength:qLen  options:MTLResourceStorageModeShared];
        id<MTLBuffer> dkb = [gDevice newBufferWithLength:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> dvb = [gDevice newBufferWithLength:kvLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> sb  = [gDevice newBufferWithLength:sLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> dpb = [gDevice newBufferWithLength:sLen options:MTLResourceStorageModeShared];
        if (!qb||!kb||!vb||!dob||!dqb||!dkb||!dvb||!sb||!dpb) return -2;

        int P3[3] = {seq, seq, causal};
        id<MTLBuffer> pb = [gDevice newBufferWithBytes:P3 length:sizeof(P3) options:MTLResourceStorageModeShared];

        MPSMatrixDescriptor* hDesc  = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:dk rowBytes:(size_t)dm*sizeof(float) dataType:MPSDataTypeFloat32];    // [seq,dk] window of [seq,dm]
        MPSMatrixDescriptor* kvDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:dk rowBytes:(size_t)kvDim*sizeof(float) dataType:MPSDataTypeFloat32]; // [seq,dk] window of [seq,kvDim]
        MPSMatrixDescriptor* sDesc  = [MPSMatrixDescriptor matrixDescriptorWithRows:seq columns:seq rowBytes:(size_t)seq*sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixMultiplication* mmQK  = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:NO  transposeRight:YES resultRows:seq resultColumns:seq interiorColumns:dk  alpha:scale beta:0.0]; // S=scale·Q·Kᵀ
        MPSMatrixMultiplication* mmdP  = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:NO  transposeRight:YES resultRows:seq resultColumns:seq interiorColumns:dk  alpha:1.0   beta:0.0]; // dP=dO·Vᵀ
        MPSMatrixMultiplication* mmdQ  = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:NO  transposeRight:NO  resultRows:seq resultColumns:dk  interiorColumns:seq alpha:scale beta:0.0]; // dQ=scale·dS·K
        // dV/dK accumulate across the rep query heads sharing a kv head (GQA): beta=0 for the first
        // query head of each group (h%rep==0), beta=1 for the rest. For MHA rep=1 → always beta=0.
        MPSMatrixMultiplication* mmdV0 = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:YES transposeRight:NO  resultRows:seq resultColumns:dk  interiorColumns:seq alpha:1.0   beta:0.0]; // dV=Pᵀ·dO
        MPSMatrixMultiplication* mmdV1 = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:YES transposeRight:NO  resultRows:seq resultColumns:dk  interiorColumns:seq alpha:1.0   beta:1.0];
        MPSMatrixMultiplication* mmdK0 = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:YES transposeRight:NO  resultRows:seq resultColumns:dk  interiorColumns:seq alpha:scale beta:0.0]; // dK=scale·dSᵀ·Q
        MPSMatrixMultiplication* mmdK1 = [[MPSMatrixMultiplication alloc] initWithDevice:gDevice transposeLeft:YES transposeRight:NO  resultRows:seq resultColumns:dk  interiorColumns:seq alpha:scale beta:1.0];

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        for (int h = 0; h < heads; h++) {
            int kvh = h / rep;
            size_t hoff  = (size_t)h*dk*sizeof(float);   // query-head window stride dm
            size_t kvoff = (size_t)kvh*dk*sizeof(float);  // kv-head window stride kvDim
            MPSMatrix* mQ  = [[MPSMatrix alloc] initWithBuffer:qb  offset:hoff  descriptor:hDesc];
            MPSMatrix* mK  = [[MPSMatrix alloc] initWithBuffer:kb  offset:kvoff descriptor:kvDesc];
            MPSMatrix* mV  = [[MPSMatrix alloc] initWithBuffer:vb  offset:kvoff descriptor:kvDesc];
            MPSMatrix* mdO = [[MPSMatrix alloc] initWithBuffer:dob offset:hoff  descriptor:hDesc];
            MPSMatrix* mdQ = [[MPSMatrix alloc] initWithBuffer:dqb offset:hoff  descriptor:hDesc];
            MPSMatrix* mdK = [[MPSMatrix alloc] initWithBuffer:dkb offset:kvoff descriptor:kvDesc];
            MPSMatrix* mdV = [[MPSMatrix alloc] initWithBuffer:dvb offset:kvoff descriptor:kvDesc];
            MPSMatrix* mS  = [[MPSMatrix alloc] initWithBuffer:sb  offset:0 descriptor:sDesc];
            MPSMatrix* mdP = [[MPSMatrix alloc] initWithBuffer:dpb offset:0 descriptor:sDesc];
            int firstInGroup = (h % rep) == 0;
            // S = scale·Q·Kᵀ ; P = softmax(S)
            [mmQK encodeToCommandBuffer:cmd leftMatrix:mQ rightMatrix:mK resultMatrix:mS];
            id<MTLComputeCommandEncoder> e1 = [cmd computeCommandEncoder];
            [e1 setComputePipelineState:gSoftmaxCausal];
            [e1 setBuffer:sb offset:0 atIndex:0]; [e1 setBuffer:pb offset:0 atIndex:1];
            [e1 dispatchThreadgroups:MTLSizeMake(seq,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e1 endEncoding];
            // dV += Pᵀ·dO (accumulate per kv head) ; dP = dO·Vᵀ
            [(firstInGroup ? mmdV0 : mmdV1) encodeToCommandBuffer:cmd leftMatrix:mS rightMatrix:mdO resultMatrix:mdV];
            [mmdP encodeToCommandBuffer:cmd leftMatrix:mdO rightMatrix:mV resultMatrix:mdP];
            // dS = P ⊙ (dP − rowsum(P⊙dP))  (in place on dpb)
            id<MTLComputeCommandEncoder> e2 = [cmd computeCommandEncoder];
            [e2 setComputePipelineState:gSoftmaxJacobian];
            [e2 setBuffer:sb offset:0 atIndex:0]; [e2 setBuffer:dpb offset:0 atIndex:1]; [e2 setBuffer:pb offset:0 atIndex:2];
            [e2 dispatchThreadgroups:MTLSizeMake(seq,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [e2 endEncoding];
            // dQ = scale·dS·K (per query head) ; dK += scale·dSᵀ·Q (accumulate per kv head)
            [mmdQ encodeToCommandBuffer:cmd leftMatrix:mdP rightMatrix:mK resultMatrix:mdQ];
            [(firstInGroup ? mmdK0 : mmdK1) encodeToCommandBuffer:cmd leftMatrix:mdP rightMatrix:mQ resultMatrix:mdK];
        }
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;
        memcpy(dQ, dqb.contents, qLen);
        memcpy(dK, dkb.contents, kvLen);
        memcpy(dV, dvb.contents, kvLen);
        return 0;
    }
}

// RetNet retention forward (§T172). One thread per query-row n streams keys m=n..0 with an
// incremental γ-decay (decay=1 at m=n, ×γ each step down) → acc[j] += (Σ_i Q[n,i]K[m,i])·decay·V[m,j],
// a register acc[d] (d≤128) and NO [L,L] score matrix — the parallel form O=(QKᵀ⊙D)V, D_nm=γ^(n−m).
static NSString* const kRetentionSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void retention_f32(device const float* Q [[buffer(0)]],\n"
     "                          device const float* K [[buffer(1)]],\n"
     "                          device const float* V [[buffer(2)]],\n"
     "                          device float* O [[buffer(3)]],\n"
     "                          constant int* P [[buffer(4)]],\n"
     "                          constant float* FP [[buffer(5)]],\n"
     "                          uint gid [[thread_position_in_grid]]) {\n"
     "  int L=P[0], dk=P[1], dv=P[2]; float gamma=FP[0];\n"
     "  if ((int)gid >= L) return;\n"
     "  int n=gid; float acc[128]; for(int j=0;j<dv;j++) acc[j]=0.0f;\n"
     "  float decay=1.0f;\n"
     "  for (int m=n;m>=0;m--){\n"
     "    float dot=0.0f; for(int i=0;i<dk;i++) dot+=Q[n*dk+i]*K[m*dk+i];\n"
     "    float pm=dot*decay;\n"
     "    for(int j=0;j<dv;j++) acc[j]+=pm*V[m*dv+j];\n"
     "    decay*=gamma;\n"
     "  }\n"
     "  for(int j=0;j<dv;j++) O[n*dv+j]=acc[j];\n"
     "}\n";

static id<MTLComputePipelineState> gRetention = nil;

static int ensure_retention(void) {
    if (gRetention != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRetentionSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"retention_f32"];
    if (fn == nil) return -11;
    gRetention = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRetention != nil ? 0 : -12;
}

int mtl_retention_f32(const float* Q, const float* K, const float* V, float* O,
                      int L, int dk, int dv, float gamma) {
    if (ensure_init() != 0) return -1;
    if (ensure_retention() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        size_t lenqk = (size_t)L * dk * sizeof(float); // Q,K are [L,dk]
        size_t lenvo = (size_t)L * dv * sizeof(float);  // V,O are [L,dv]
        id<MTLBuffer> qb = pool_in(Q, lenqk);
        id<MTLBuffer> kb = pool_in(K, lenqk);
        id<MTLBuffer> vb = pool_in(V, lenvo);
        id<MTLBuffer> ob = pool_buf(lenvo);
        int P[3] = {L, dk, dv};
        float FP[1] = {gamma};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (qb == nil || kb == nil || vb == nil || ob == nil || pb == nil || fpb == nil) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRetention];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:ob offset:0 atIndex:3];
        [enc setBuffer:pb offset:0 atIndex:4];
        [enc setBuffer:fpb offset:0 atIndex:5];
        NSUInteger tg = 64; // register-heavy kernel (acc[128]); see §T337 note at mha_f32
        if ((NSUInteger)L < tg) tg = (NSUInteger)L;
        [enc dispatchThreads:MTLSizeMake(L, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(O, ob.contents, lenvo);
        return 0;
    }
}

// RetNet retention backward (§T174). One thread per query-row n: writes its own dQ row (register)
// and ATOMICALLY accumulates into the shared dK/dV (row m gets contributions from every n≥m), exactly
// like mha_bwd. dK/dV buffers start pre-zeroed. Forward O=(A⊙D)V, P=A⊙D; dQ_n=Σ_{m≤n}dA_nm·K_m,
// dK_m+=dA_nm·Q_n, dV_m+=P_nm·dO_n, dA_nm=(Σ_j dO_nj V_mj)·γ^(n−m). d≤128.
static NSString* const kRetentionBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void retention_bwd_f32(device const float* Q [[buffer(0)]],\n"
     "                              device const float* K [[buffer(1)]],\n"
     "                              device const float* V [[buffer(2)]],\n"
     "                              device const float* dO [[buffer(3)]],\n"
     "                              device float* dQ [[buffer(4)]],\n"
     "                              device atomic_float* dK [[buffer(5)]],\n"
     "                              device atomic_float* dV [[buffer(6)]],\n"
     "                              constant int* P [[buffer(7)]],\n"
     "                              constant float* FP [[buffer(8)]],\n"
     "                              uint gid [[thread_position_in_grid]]) {\n"
     "  int L=P[0], dk=P[1], dv=P[2]; float gamma=FP[0];\n"
     "  if ((int)gid >= L) return;\n"
     "  int n=gid; float dqloc[128]; for(int i=0;i<dk;i++) dqloc[i]=0.0f;\n"
     "  float decay=1.0f;\n"
     "  for (int m=n;m>=0;m--){\n"
     "    float a=0.0f; for(int i=0;i<dk;i++) a+=Q[n*dk+i]*K[m*dk+i];\n"
     "    float pnm=a*decay;\n"
     "    float dp=0.0f;\n"
     "    for(int j=0;j<dv;j++){ float gnj=dO[n*dv+j]; dp+=gnj*V[m*dv+j];\n"
     "      atomic_fetch_add_explicit(&dV[m*dv+j], pnm*gnj, memory_order_relaxed); }\n"
     "    float dA=dp*decay;\n"
     "    for(int i=0;i<dk;i++){ dqloc[i]+=dA*K[m*dk+i];\n"
     "      atomic_fetch_add_explicit(&dK[m*dk+i], dA*Q[n*dk+i], memory_order_relaxed); }\n"
     "    decay*=gamma;\n"
     "  }\n"
     "  for(int i=0;i<dk;i++) dQ[n*dk+i]=dqloc[i];\n"
     "}\n";

static id<MTLComputePipelineState> gRetentionBwd = nil;

static int ensure_retention_bwd(void) {
    if (gRetentionBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kRetentionBwdSource options:nil error:&err];
    if (lib == nil) return -10;
    id<MTLFunction> fn = [lib newFunctionWithName:@"retention_bwd_f32"];
    if (fn == nil) return -11;
    gRetentionBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gRetentionBwd != nil ? 0 : -12;
}

int mtl_retention_backward_f32(const float* Q, const float* K, const float* V, const float* dO,
                               float* dQ, float* dK, float* dV, int L, int dk, int dv, float gamma) {
    if (ensure_init() != 0) return -1;
    if (ensure_retention_bwd() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        size_t lenqk = (size_t)L * dk * sizeof(float); // Q,K,dQ,dK are [L,dk]
        size_t lenvo = (size_t)L * dv * sizeof(float);  // V,dO,dV are [L,dv]
        id<MTLBuffer> qb = pool_in(Q, lenqk);
        id<MTLBuffer> kb = pool_in(K, lenqk);
        id<MTLBuffer> vb = pool_in(V, lenvo);
        id<MTLBuffer> dob = pool_in(dO, lenvo);
        id<MTLBuffer> dqb = pool_buf(lenqk);
        id<MTLBuffer> dkb = pool_buf(lenqk);
        id<MTLBuffer> dvb = pool_buf(lenvo);
        if (!qb || !kb || !vb || !dob || !dqb || !dkb || !dvb) return -2;
        memset(dkb.contents, 0, lenqk); // atomic accumulators must start at zero
        memset(dvb.contents, 0, lenvo);

        int P[3] = {L, dk, dv};
        float FP[1] = {gamma};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (!pb || !fpb) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gRetentionBwd];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:dob offset:0 atIndex:3];
        [enc setBuffer:dqb offset:0 atIndex:4];
        [enc setBuffer:dkb offset:0 atIndex:5];
        [enc setBuffer:dvb offset:0 atIndex:6];
        [enc setBuffer:pb offset:0 atIndex:7];
        [enc setBuffer:fpb offset:0 atIndex:8];
        NSUInteger tg = 64; // register-heavy kernel (dqloc[128]); see §T337 note at mha_f32
        if ((NSUInteger)L < tg) tg = (NSUInteger)L;
        [enc dispatchThreads:MTLSizeMake(L, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(dQ, dqb.contents, lenqk);
        memcpy(dK, dkb.contents, lenqk);
        memcpy(dV, dvb.contents, lenvo);
        return 0;
    }
}

// SDPA backward kernel. One thread per (head, query row) recomputes the softmax
// (max/sum/dot passes mirroring the reference), writes its own dQ row directly,
// and atomically accumulates into the shared dK/dV. dQloc is a fixed-size register
// array so dk is bounded at 128.
static NSString* const kMHABwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void mha_bwd_f32(device const float* Q [[buffer(0)]],\n"
     "                        device const float* K [[buffer(1)]],\n"
     "                        device const float* V [[buffer(2)]],\n"
     "                        device const float* dO [[buffer(3)]],\n"
     "                        device float* dQ [[buffer(4)]],\n"
     "                        device atomic_float* dK [[buffer(5)]],\n"
     "                        device atomic_float* dV [[buffer(6)]],\n"
     "                        constant int* P [[buffer(7)]],\n"
     "                        constant float* FP [[buffer(8)]],\n"
     "                        uint gid [[thread_position_in_grid]]) {\n"
     "  int sq=P[0], sk=P[1], dm=P[2], heads=P[3], kvHeads=P[4], dk=P[5], causal=P[6], window=P[7];\n"
     "  float scale=FP[0];\n"
     "  if ((int)gid >= heads*sq) return;\n"
     "  int h=gid/sq, i=gid%sq;\n"
     "  int rep=heads/kvHeads, dkv=kvHeads*dk;\n"
     "  int qOff=h*dk, kvOff=(h/rep)*dk, off=sk-sq;\n"
     "  int jmax = causal ? (off+i+1) : sk;\n"
     "  int jmin = (window>0 && off+i-window+1>0) ? (off+i-window+1) : 0;\n"
     "  float m=-INFINITY;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; s*=scale; if(s>m) m=s; }\n"
     "  float sum=0;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; sum+=exp(s*scale-m); }\n"
     "  float dot=0;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; float a=exp(s*scale-m)/sum; float da=0; for(int d=0;d<dk;d++) da+=dO[i*dm+qOff+d]*V[j*dkv+kvOff+d]; dot+=a*da; }\n"
     "  float dQloc[128]; for(int d=0;d<dk;d++) dQloc[d]=0;\n"
     "  for (int j=jmin;j<jmax;j++){ float s=0; for(int d=0;d<dk;d++) s+=Q[i*dm+qOff+d]*K[j*dkv+kvOff+d]; float a=exp(s*scale-m)/sum; float da=0; for(int d=0;d<dk;d++) da+=dO[i*dm+qOff+d]*V[j*dkv+kvOff+d]; float dS=scale*a*(da-dot);\n"
     "    for(int d=0;d<dk;d++){ atomic_fetch_add_explicit(&dV[j*dkv+kvOff+d], a*dO[i*dm+qOff+d], memory_order_relaxed); atomic_fetch_add_explicit(&dK[j*dkv+kvOff+d], dS*Q[i*dm+qOff+d], memory_order_relaxed); dQloc[d]+=dS*K[j*dkv+kvOff+d]; } }\n"
     "  for(int d=0;d<dk;d++) dQ[i*dm+qOff+d]=dQloc[d];\n"
     "}\n";

static id<MTLComputePipelineState> gMHABwd = nil;

static int ensure_mha_bwd(void) {
    if (gMHABwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kMHABwdSource options:nil error:&err];
    if (lib == nil) return -5;
    id<MTLFunction> fn = [lib newFunctionWithName:@"mha_bwd_f32"];
    if (fn == nil) return -5;
    gMHABwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gMHABwd != nil ? 0 : -5;
}

int mtl_mha_backward_f32(const float* Q, const float* K, const float* V, const float* dO,
                         float* dQ, float* dK, float* dV,
                         int sq, int sk, int dm, int heads, int kvHeads, int dk,
                         int causal, int window, float scale) {
    if (ensure_init() != 0) return -1;
    if (ensure_mha_bwd() != 0) return -5;
    OP_BEGIN;
    @autoreleasepool {
        int dkv = kvHeads * dk;
        size_t qLen = (size_t)sq * dm * sizeof(float);
        size_t kvLen = (size_t)sk * dkv * sizeof(float);

        id<MTLBuffer> qb = pool_in(Q, qLen);
        id<MTLBuffer> kb = pool_in(K, kvLen);
        id<MTLBuffer> vb = pool_in(V, kvLen);
        id<MTLBuffer> dob = pool_in(dO, qLen);
        id<MTLBuffer> dqb = pool_buf(qLen);
        id<MTLBuffer> dkb = pool_buf(kvLen);
        id<MTLBuffer> dvb = pool_buf(kvLen);
        if (!qb || !kb || !vb || !dob || !dqb || !dkb || !dvb) return -2;
        memset(dkb.contents, 0, kvLen); // atomic accumulators must start at zero
        memset(dvb.contents, 0, kvLen);

        int P[8] = {sq, sk, dm, heads, kvHeads, dk, causal, window};
        float FP[1] = {scale};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        id<MTLBuffer> fpb = pool_in(FP, sizeof(FP));
        if (!pb || !fpb) return -2;

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gMHABwd];
        [enc setBuffer:qb offset:0 atIndex:0];
        [enc setBuffer:kb offset:0 atIndex:1];
        [enc setBuffer:vb offset:0 atIndex:2];
        [enc setBuffer:dob offset:0 atIndex:3];
        [enc setBuffer:dqb offset:0 atIndex:4];
        [enc setBuffer:dkb offset:0 atIndex:5];
        [enc setBuffer:dvb offset:0 atIndex:6];
        [enc setBuffer:pb offset:0 atIndex:7];
        [enc setBuffer:fpb offset:0 atIndex:8];
        int total = heads * sq;
        NSUInteger tg = 64; // register-heavy kernel (dQloc[128]); see §T337 note at mha_f32
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(dQ, dqb.contents, qLen);
        memcpy(dK, dkb.contents, kvLen);
        memcpy(dV, dvb.contents, kvLen);
        return 0;
    }
}

// 2-D convolution (cross-correlation) forward as im2col + MPS GEMM (§T341). The
// naive one-thread-per-output kernel peaked at ~105 GFLOP/s while the MPS matmul
// sustains ~1200 — so the convolution is lowered to a matrix multiply:
//   col[c·KH·KW + ky·KW + kx, (n·ho+oy)·wo + ox] = X[n,c,oy·stride+ky−pad,ox·stride+kx−pad] (0 when padded)
//   gemm = W[F, C·KH·KW] · col                     (weights are already row-major [F, C·KH·KW])
//   Out[n,f,oy,ox] = gemm[f, (n·ho+oy)·wo + ox] + B[f]
// im2col writes every col element (padding writes 0), MPS runs beta:0.0, and the
// scatter kernel writes every output element — pooled buffers stay stale-safe.
static NSString* const kIm2ColSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void im2col_f32(device const float* X [[buffer(0)]],\n"
     "                       device float* COL [[buffer(1)]],\n"
     "                       constant int* P [[buffer(2)]],\n"
     "                       uint gid [[thread_position_in_grid]]) {\n"
     "  int N=P[0],C=P[1],H=P[2],Wd=P[3],KH=P[5],KW=P[6],stride=P[7],pad=P[8],ho=P[9],wo=P[10];\n"
     "  int cols=N*ho*wo; int rows=C*KH*KW;\n"
     "  if ((int)gid >= rows*cols) return;\n"
     "  int row=(int)gid/cols, col=(int)gid%cols;\n"
     "  int kx=row%KW; int ky=(row/KW)%KH; int c=row/(KW*KH);\n"
     "  int ox=col%wo; int oy=(col/wo)%ho; int n=col/(wo*ho);\n"
     "  int iy=oy*stride+ky-pad, ix=ox*stride+kx-pad;\n"
     "  float v = (iy<0||iy>=H||ix<0||ix>=Wd) ? 0.0f : X[((n*C+c)*H+iy)*Wd+ix];\n"
     "  COL[gid]=v;\n"
     "}\n";

// colout_f32 scatters the GEMM result [F, N·ho·wo] into the NCHW output layout and
// adds the bias. One thread per output element.
static NSString* const kColOutSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void colout_f32(device const float* G [[buffer(0)]],\n"
     "                       device const float* B [[buffer(1)]],\n"
     "                       device float* O [[buffer(2)]],\n"
     "                       constant int* P [[buffer(3)]],\n"
     "                       uint gid [[thread_position_in_grid]]) {\n"
     "  int N=P[0],F=P[4],ho=P[9],wo=P[10];\n"
     "  if ((int)gid >= N*F*ho*wo) return;\n"
     "  int idx=gid; int ox=idx%wo; idx/=wo; int oy=idx%ho; idx/=ho; int f=idx%F; int n=idx/F;\n"
     "  int cols=N*ho*wo;\n"
     "  O[gid] = G[f*cols + (n*ho+oy)*wo + ox] + B[f];\n"
     "}\n";

static id<MTLComputePipelineState> gIm2Col = nil;
static id<MTLComputePipelineState> gColOut = nil;

static int ensure_conv2d(void) {
    if (gIm2Col != nil && gColOut != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib1 = [gDevice newLibraryWithSource:kIm2ColSource options:nil error:&err];
    if (lib1 == nil) return -6;
    id<MTLFunction> fn1 = [lib1 newFunctionWithName:@"im2col_f32"];
    if (fn1 == nil) return -6;
    gIm2Col = [gDevice newComputePipelineStateWithFunction:fn1 error:&err];
    if (gIm2Col == nil) return -6;
    id<MTLLibrary> lib2 = [gDevice newLibraryWithSource:kColOutSource options:nil error:&err];
    if (lib2 == nil) return -6;
    id<MTLFunction> fn2 = [lib2 newFunctionWithName:@"colout_f32"];
    if (fn2 == nil) return -6;
    gColOut = [gDevice newComputePipelineStateWithFunction:fn2 error:&err];
    return gColOut != nil ? 0 : -6;
}

int mtl_conv2d_f32(const float* X, const float* W, const float* B, float* Out,
                   int N, int C, int H, int Wd, int F, int KH, int KW,
                   int stride, int pad, int ho, int wo) {
    if (ensure_init() != 0) return -1;
    if (ensure_conv2d() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        int rows = C * KH * KW;   // GEMM interior dim
        int cols = N * ho * wo;   // GEMM result columns
        size_t xLen = (size_t)N * C * H * Wd * sizeof(float);
        size_t wLen = (size_t)F * rows * sizeof(float);
        size_t bLen = (size_t)F * sizeof(float);
        size_t colLen = (size_t)rows * cols * sizeof(float);
        size_t gemmLen = (size_t)F * cols * sizeof(float);
        size_t oLen = (size_t)N * F * ho * wo * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> bb = pool_in(B, bLen);
        id<MTLBuffer> colb = pool_buf(colLen);
        id<MTLBuffer> gb = pool_buf(gemmLen);
        id<MTLBuffer> ob = pool_buf(oLen);
        int P[11] = {N, C, H, Wd, F, KH, KW, stride, pad, ho, wo};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || bb == nil || colb == nil || gb == nil || ob == nil || pb == nil) return -2;

        MPSMatrixDescriptor* wDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:F columns:rows rowBytes:(size_t)rows * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* colDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:cols rowBytes:(size_t)cols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* gDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:F columns:cols rowBytes:(size_t)cols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix* mW = [[MPSMatrix alloc] initWithBuffer:wb descriptor:wDesc];
        MPSMatrix* mCol = [[MPSMatrix alloc] initWithBuffer:colb descriptor:colDesc];
        MPSMatrix* mG = [[MPSMatrix alloc] initWithBuffer:gb descriptor:gDesc];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:gDevice transposeLeft:NO transposeRight:NO
            resultRows:F resultColumns:cols interiorColumns:rows alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;

        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gIm2Col];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:colb offset:0 atIndex:1];
        [enc setBuffer:pb offset:0 atIndex:2];
        int colTotal = rows * cols;
        NSUInteger tg1 = 256; // elementwise, huge thread count (§T339 rule)
        if ((NSUInteger)colTotal < tg1) tg1 = (NSUInteger)colTotal;
        [enc dispatchThreads:MTLSizeMake(colTotal, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg1, 1, 1)];
        [enc endEncoding];

        [mm encodeToCommandBuffer:cmd leftMatrix:mW rightMatrix:mCol resultMatrix:mG];

        id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
        [enc2 setComputePipelineState:gColOut];
        [enc2 setBuffer:gb offset:0 atIndex:0];
        [enc2 setBuffer:bb offset:0 atIndex:1];
        [enc2 setBuffer:ob offset:0 atIndex:2];
        [enc2 setBuffer:pb offset:0 atIndex:3];
        int outTotal = N * F * ho * wo;
        NSUInteger tg2 = 256;
        if ((NSUInteger)outTotal < tg2) tg2 = (NSUInteger)outTotal;
        [enc2 dispatchThreads:MTLSizeMake(outTotal, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg2, 1, 1)];
        [enc2 endEncoding];

        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(Out, ob.contents, oLen);
        return 0;
    }
}

// Conv2d backward kernel. One thread per output-gradient element (n,f,oy,ox)
// scatters into the shared dX/dW/dBias with atomic float adds (dX/dW/dBias passed
// in pre-zeroed).
static NSString* const kConv2DBwdSource =
    @"#include <metal_stdlib>\n"
     "using namespace metal;\n"
     "kernel void conv2d_bwd_f32(device const float* X [[buffer(0)]],\n"
     "                           device const float* W [[buffer(1)]],\n"
     "                           device const float* dO [[buffer(2)]],\n"
     "                           device atomic_float* dX [[buffer(3)]],\n"
     "                           device atomic_float* dW [[buffer(4)]],\n"
     "                           device atomic_float* dB [[buffer(5)]],\n"
     "                           constant int* P [[buffer(6)]],\n"
     "                           uint gid [[thread_position_in_grid]]) {\n"
     "  int N=P[0],C=P[1],H=P[2],Wd=P[3],F=P[4],KH=P[5],KW=P[6],stride=P[7],pad=P[8],ho=P[9],wo=P[10];\n"
     "  if ((int)gid >= N*F*ho*wo) return;\n"
     "  int idx=gid; int ox=idx%wo; idx/=wo; int oy=idx%ho; idx/=ho; int f=idx%F; int n=idx/F;\n"
     "  float g = dO[((n*F+f)*ho+oy)*wo+ox];\n"
     "  atomic_fetch_add_explicit(&dB[f], g, memory_order_relaxed);\n"
     "  for (int c=0;c<C;c++){\n"
     "    for (int ky=0;ky<KH;ky++){ int iy=oy*stride+ky-pad; if(iy<0||iy>=H) continue;\n"
     "      for (int kx=0;kx<KW;kx++){ int ix=ox*stride+kx-pad; if(ix<0||ix>=Wd) continue;\n"
     "        int xi=((n*C+c)*H+iy)*Wd+ix; int wi=((f*C+c)*KH+ky)*KW+kx;\n"
     "        atomic_fetch_add_explicit(&dX[xi], g*W[wi], memory_order_relaxed);\n"
     "        atomic_fetch_add_explicit(&dW[wi], g*X[xi], memory_order_relaxed); } } }\n"
     "}\n";

static id<MTLComputePipelineState> gConv2DBwd = nil;

static int ensure_conv2d_bwd(void) {
    if (gConv2DBwd != nil) return 0;
    NSError* err = nil;
    id<MTLLibrary> lib = [gDevice newLibraryWithSource:kConv2DBwdSource options:nil error:&err];
    if (lib == nil) return -6;
    id<MTLFunction> fn = [lib newFunctionWithName:@"conv2d_bwd_f32"];
    if (fn == nil) return -6;
    gConv2DBwd = [gDevice newComputePipelineStateWithFunction:fn error:&err];
    return gConv2DBwd != nil ? 0 : -6;
}

int mtl_conv2d_backward_f32(const float* X, const float* W, const float* dO,
                            float* dX, float* dW, float* dBias,
                            int N, int C, int H, int Wd, int F, int KH, int KW,
                            int stride, int pad, int ho, int wo) {
    if (ensure_init() != 0) return -1;
    if (ensure_conv2d_bwd() != 0) return -6;
    OP_BEGIN;
    @autoreleasepool {
        size_t xLen = (size_t)N * C * H * Wd * sizeof(float);
        size_t wLen = (size_t)F * C * KH * KW * sizeof(float);
        size_t oLen = (size_t)N * F * ho * wo * sizeof(float);
        size_t bLen = (size_t)F * sizeof(float);

        id<MTLBuffer> xb = pool_in(X, xLen);
        id<MTLBuffer> wb = pool_in(W, wLen);
        id<MTLBuffer> ob = pool_in(dO, oLen);
        id<MTLBuffer> dxb = pool_buf(xLen);
        id<MTLBuffer> dwb = pool_buf(wLen);
        id<MTLBuffer> dbb = pool_buf(bLen);
        int P[11] = {N, C, H, Wd, F, KH, KW, stride, pad, ho, wo};
        id<MTLBuffer> pb = pool_in(P, sizeof(P));
        if (xb == nil || wb == nil || ob == nil || dxb == nil || dwb == nil || dbb == nil || pb == nil) return -2;
        memset(dxb.contents, 0, xLen); // atomic accumulators start at zero
        memset(dwb.contents, 0, wLen);
        memset(dbb.contents, 0, bLen);

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:gConv2DBwd];
        [enc setBuffer:xb offset:0 atIndex:0];
        [enc setBuffer:wb offset:0 atIndex:1];
        [enc setBuffer:ob offset:0 atIndex:2];
        [enc setBuffer:dxb offset:0 atIndex:3];
        [enc setBuffer:dwb offset:0 atIndex:4];
        [enc setBuffer:dbb offset:0 atIndex:5];
        [enc setBuffer:pb offset:0 atIndex:6];
        int total = N * F * ho * wo;
        NSUInteger tg = gConv2DBwd.maxTotalThreadsPerThreadgroup;
        if ((NSUInteger)total < tg) tg = (NSUInteger)total;
        [enc dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(dX, dxb.contents, xLen);
        memcpy(dW, dwb.contents, wLen);
        memcpy(dBias, dbb.contents, bLen);
        return 0;
    }
}

// mtl_matmul_f32 computes C[M,N] = opA(A)·opB(B). When transA, A is STORED as [K,M] and
// MPS transposes it (opA = Aᵀ); else A is [M,K]. Likewise transB: B stored [N,K] → opB=Bᵀ,
// else [K,N]. This lets the matmul backward pass a weight/input transpose as an MPS flag
// instead of materializing it with a CPU strided-gather copy (§T356, was ~8ms/operand).
int mtl_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N,
                   int transA, int transB) {
    if (ensure_init() != 0) return -1;
    OP_BEGIN;
    @autoreleasepool {
        size_t aLen = (size_t)M * K * sizeof(float); // A holds M·K elements in either layout
        size_t bLen = (size_t)K * N * sizeof(float);
        size_t cLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> aBuf = pool_in(A, aLen);
        id<MTLBuffer> bBuf = pool_in(B, bLen);
        id<MTLBuffer> cBuf = pool_buf(cLen);
        if (aBuf == nil || bBuf == nil || cBuf == nil) return -2;

        // Descriptor dims describe the STORED matrix; the transpose flag reinterprets it.
        int aRows = transA ? K : M, aCols = transA ? M : K;
        int bRows = transB ? N : K, bCols = transB ? K : N;
        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:aRows columns:aCols rowBytes:(size_t)aCols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:bRows columns:bCols rowBytes:(size_t)bCols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(size_t)N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:gDevice transposeLeft:(transA?YES:NO) transposeRight:(transB?YES:NO)
            resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [gQueue commandBuffer];
        if (cmd == nil) return -3;
        [mm encodeToCommandBuffer:cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cmd commit];
        [cmd waitUntilCompleted];
        if (cmd.status != MTLCommandBufferStatusCompleted) return -4;

        memcpy(C, cBuf.contents, cLen);
        return 0;
    }
}
