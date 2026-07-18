// wmma_attn.cu — fused prefill attention on tensor cores (Tw-FLASHATTN slice 2). One warp per
// (head, 16-query tile) computes O[16,hd] = softmax(scale·Q·Kᵀ + causal)·V in ONE kernel, no
// intermediate global scores (our current path materializes scores to global via cuBLAS-batched
// GEMM — the measured vLLM prefill gap is exactly this fusion). head_dim hd=64 (4 k-fragments).
// Scores for the query tile are staged in shared (16×seq), row-softmaxed, then P·V via mma.
// Layout: Q,K,V are [heads, seq, hd] row-major f16; O is [heads, seq, hd] f32. Causal mask on
// absolute positions. seq must be a multiple of 16 (prefill pads); hd==64.
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

extern "C" __global__ void wmma_attn(const half* Q, const half* K, const half* V, float* O,
                                     int heads, int seq, int hd, float scale) {
    // grid: x = query-tile index (seq/16), y = head. one warp per block.
    int qt = blockIdx.x;          // query tile (16 queries)
    int h  = blockIdx.y;
    int q0 = qt * 16;
    int lane = threadIdx.x & 31;
    const half* Qh = Q + (size_t)h * seq * hd;
    const half* Kh = K + (size_t)h * seq * hd;
    const half* Vh = V + (size_t)h * seq * hd;
    float* Oh = O + (size_t)h * seq * hd;

    extern __shared__ float sscore[]; // [16 * seq]

    // Q fragments for this query tile: Q[q0:q0+16, 0:64] → 4 matrix_a frags over hd.
    wmma::fragment<wmma::matrix_a, 16,16,16, half, wmma::row_major> qf[4];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++)
        wmma::load_matrix_sync(qf[kk], Qh + (size_t)q0 * hd + kk*16, hd);

    // Phase 1: S[16, seq] = scale·Q·Kᵀ, staged to shared with causal mask.
    for (int jt = 0; jt < seq; jt += 16) {
        wmma::fragment<wmma::accumulator, 16,16,16, float> sfrag;
        wmma::fill_fragment(sfrag, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            // Kᵀ tile: B col_major reads K[jt:jt+16, kk*16:+16] as [16(n=key),16(k=dim)].
            wmma::fragment<wmma::matrix_b, 16,16,16, half, wmma::col_major> kf;
            wmma::load_matrix_sync(kf, Kh + (size_t)jt * hd + kk*16, hd);
            wmma::mma_sync(sfrag, qf[kk], kf, sfrag);
        }
        // store to shared [16][jt..jt+16] with scale
        wmma::store_matrix_sync(sscore + jt, sfrag, seq, wmma::mem_row_major);
    }
    __syncwarp();

    // scale + causal mask + row softmax: each lane owns query rows lane%16 across its columns.
    // Simpler: 16 lanes each handle one query row (rows 0..15), lanes 16..31 idle.
    if (lane < 16) {
        int qi = q0 + lane;
        float* row = sscore + (size_t)lane * seq;
        float m = -1e30f;
        for (int j = 0; j < seq; j++) {
            float v = row[j] * scale;
            if (j > qi) v = -1e30f;    // causal
            row[j] = v;
            if (v > m) m = v;
        }
        float sum = 0.f;
        for (int j = 0; j < seq; j++) { float e = __expf(row[j] - m); row[j] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int j = 0; j < seq; j++) row[j] *= inv;
    }
    __syncwarp();

    // Phase 2: O[16,64] = P[16,seq]·V[seq,64] via mma. P from shared (row_major), V (row_major).
    // O = 4 accumulator frags over hd (n=0..63 in 16-col fragments).
    wmma::fragment<wmma::accumulator, 16,16,16, float> of[4];
    #pragma unroll
    for (int n = 0; n < 4; n++) wmma::fill_fragment(of[n], 0.0f);
    // P must be f16 for matrix_a — stage a f16 copy of the softmax weights per ktile.
    __shared__ half phalf[16*16];
    for (int jt = 0; jt < seq; jt += 16) {
        // pack P[0:16, jt:jt+16] into phalf
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
            wmma::load_matrix_sync(vf, Vh + (size_t)jt * hd + n*16, hd);
            wmma::mma_sync(of[n], pf, vf, of[n]);
        }
        __syncwarp();
    }
    #pragma unroll
    for (int n = 0; n < 4; n++)
        wmma::store_matrix_sync(Oh + (size_t)q0 * hd + n*16, of[n], hd, wmma::mem_row_major);
}
