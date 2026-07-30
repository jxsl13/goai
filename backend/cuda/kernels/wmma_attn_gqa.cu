// wmma_attn_gqa.cu — fused prefill attention in the SHIPPED prefill layout [seq, heads·hd] with
// GQA, so it drops into the resident prefill forward replacing the cuBLAS-batched
// GroupedQueryAttention (which materializes global scores). Same math as wmma_attn.cu but reads
// the interleaved layout via a row STRIDE (qWidth / kvWidth) instead of a per-head contiguous
// base, and maps query head h -> kv head h/(qHeads/kvHeads). One warp per (q-head, 16-query
// tile). hd==64, seq%16==0. Q:[seq,qHeads·hd] K,V:[seq,kvHeads·hd] O:[seq,qHeads·hd] f16 in / f32 out.
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

extern "C" __global__ void wmma_attn_gqa(const half* Q, const half* K, const half* V, float* O,
                                         int seq, int qHeads, int kvHeads, int hd, float scale) {
    int qt = blockIdx.x;            // query tile (16 queries)
    int h  = blockIdx.y;           // query head
    int kvh = h / (qHeads / kvHeads);
    int q0 = qt * 16;
    int lane = threadIdx.x & 31;
    int qW = qHeads * hd, kvW = kvHeads * hd;
    const half* Qh = Q + (size_t)h  * hd;   // column offset of this q-head; row stride = qW
    const half* Kh = K + (size_t)kvh * hd;  // row stride = kvW
    const half* Vh = V + (size_t)kvh * hd;
    float* Oh = O + (size_t)h * hd;

    extern __shared__ float sscore[]; // [16 * seq]

    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> qf[4];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++)
        wmma::load_matrix_sync(qf[kk], Qh + (size_t)q0 * qW + kk*16, qW);

    for (int jt = 0; jt < seq; jt += 16) {
        wmma::fragment<wmma::accumulator, 16,16,16, float> sfrag;
        wmma::fill_fragment(sfrag, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> kf;
            wmma::load_matrix_sync(kf, Kh + (size_t)jt * kvW + kk*16, kvW);
            wmma::mma_sync(sfrag, qf[kk], kf, sfrag);
        }
        wmma::store_matrix_sync(sscore + jt, sfrag, seq, wmma::mem_row_major);
    }
    __syncwarp();

    // Warp-parallel stable softmax: all 32 lanes cooperate on each of the 16 query rows in turn,
    // each lane reducing a strided (lane, lane+32, …) slice then combining via __shfl_xor
    // butterflies — replacing the old 16-lanes-each-serial-over-seq tail that dominated (and
    // regressed) at large seq. Serial length per row drops from seq to seq/32. Max is
    // order-independent (bit-identical to the sequential max); the sum reassociates over the 32
    // strided partials (tol-gated — matching the f16 accumulation this kernel already rides).
    for (int r = 0; r < 16; r++) {
        int qi = q0 + r;
        float* row = sscore + (size_t)r * seq;
        float m = -1e30f;
        for (int j = lane; j < seq; j += 32) {
            float v = row[j] * scale;
            if (j > qi) v = -1e30f;
            row[j] = v;
            if (v > m) m = v;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        float sum = 0.f;
        for (int j = lane; j < seq; j += 32) { float e = __expf(row[j] - m); row[j] = e; sum += e; }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        float inv = 1.0f / sum;
        for (int j = lane; j < seq; j += 32) row[j] *= inv;
    }
    __syncwarp();

    wmma::fragment<wmma::accumulator, 16,16,16, float> of[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(of[n], 0.0f);
    __shared__ half phalf[16*16];
    for (int jt = 0; jt < seq; jt += 16) {
        for (int idx = lane; idx < 16*16; idx += 32) {
            int r = idx >> 4, cc = idx & 15;
            phalf[idx] = __float2half(sscore[(size_t)r * seq + jt + cc]);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> pf;
        wmma::load_matrix_sync(pf, phalf, 16);
        #pragma unroll
        for (int n = 0; n < 4; n++) {
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::row_major> vf;
            wmma::load_matrix_sync(vf, Vh + (size_t)jt * kvW + n*16, kvW);
            wmma::mma_sync(of[n], pf, vf, of[n]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::store_matrix_sync(Oh + (size_t)q0 * qW + n*16, of[n], qW, wmma::mem_row_major);
}
