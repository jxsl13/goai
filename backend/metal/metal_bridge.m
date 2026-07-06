//go:build metal && darwin && cgo

// Objective-C side of the metal backend (§T20). ARC-compiled (see cgo CFLAGS).
// One shared device+queue; per-call shared-storage buffers (Apple Silicon UMA
// makes these host-visible; newBufferWithBytesNoCopy zero-copy is a later opt).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

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

int mtl_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N) {
    if (ensure_init() != 0) return -1;
    @autoreleasepool {
        size_t aLen = (size_t)M * K * sizeof(float);
        size_t bLen = (size_t)K * N * sizeof(float);
        size_t cLen = (size_t)M * N * sizeof(float);

        id<MTLBuffer> aBuf = [gDevice newBufferWithBytes:A length:aLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> bBuf = [gDevice newBufferWithBytes:B length:bLen options:MTLResourceStorageModeShared];
        id<MTLBuffer> cBuf = [gDevice newBufferWithLength:cLen options:MTLResourceStorageModeShared];
        if (aBuf == nil || bBuf == nil || cBuf == nil) return -2;

        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:(size_t)K * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:(size_t)N * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:(size_t)N * sizeof(float) dataType:MPSDataTypeFloat32];

        MPSMatrix* mA = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* mB = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        MPSMatrix* mC = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];

        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:gDevice transposeLeft:NO transposeRight:NO
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
