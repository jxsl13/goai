// wmma_gemm.cu — first nvcc-compiled tensor-core kernel in the CUDA backend (Tw-FLASHATTN
// slice 1: prove the nvcc -> cubin -> cuModuleLoadDataEx pipeline). C[M,N] f32 = A[M,K] f16 ·
// B[K,N] f16, f32 accumulate. One warp per 16x16 output tile; the K loop streams 16x16
// fragments. mma.h is exactly what nvrtc could not compile — this cubin is built by nvcc
// (scripts/cuda-nvcc-env.sh) and committed like the vulkan .spv artifacts. Correctness gate
// only (cuBLAS is faster for plain GEMM); the real payoff is the fused-attention kernel this
// pipeline unblocks.
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

extern "C" __global__ void wmma_gemm(const half* A, const half* B, float* C, int M, int N, int K) {
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);   // 16-row tile index
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // 16-col tile index
    int row = warpM * 16, col = warpN * 16;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(fa, A + row * K + k, K);
        wmma::load_matrix_sync(fb, B + k * N + col, N);
        wmma::mma_sync(acc, fa, fb, acc);
    }
    wmma::store_matrix_sync(C + row * N + col, acc, N, wmma::mem_row_major);
}
