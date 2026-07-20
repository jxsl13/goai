// wmma_paged_decode.cu — tensor-core (mma.h) batched paged DECODE attention. The warp-primitive
// paged_decode_gqa kernel is latency-bound GEMV (~0.64 TFLOP/s, 34% of the A1 step). This does the
// QK and PV as WMMA GEMMs on the GQA group's queries: per (seq, kv head) the `group` q-heads form
// an M=16 (padded) tile, K=hd=64, keys tiled by 16. Paged K/V are staged to shared f16 first (the
// block table scatters rows). Full softmax in shared (seqLen<=128 fits). hd==64, group<=8, blockSize<=16.
// Q,O: f32 [batch, qHeads*hd]; poolK/V: f32 [*, kvHeads*hd]; one block per (kv head, seq); 32 threads.
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;

#define MAXK 128  // max keys per sequence (seqLen)

extern "C" __global__ void wmma_paged_decode(const float* Q, const float* poolK, const float* poolV,
                                             const int* blockTables, const int* seqLens, float* O,
                                             int batch, int qHeads, int kvHeads, int hd, int blockSize,
                                             int maxBlocks, float scale) {
    int kvh = blockIdx.x, seq = blockIdx.y;
    int group = qHeads / kvHeads, WKV = kvHeads * hd, n = seqLens[seq];
    if (n > MAXK) n = MAXK;
    int nkp = (n + 15) & ~15;            // keys padded to a multiple of 16
    const int* bt = blockTables + (size_t)seq * maxBlocks;
    int tid = threadIdx.x, nt = blockDim.x, lane = threadIdx.x & 31;

    extern __shared__ char smem[];
    half* shQ = (half*)smem;             // [16*hd] queries (rows>=group zero)
    half* shK = shQ + 16 * hd;           // [MAXK*hd] K (f16)
    half* shV = shK + MAXK * hd;         // [MAXK*hd] V (f16)
    float* shS = (float*)(shV + MAXK * hd); // [16*MAXK] scores

    // load the group's queries [group,hd] -> shQ, pad rows group..15 with 0 (ALL threads)
    for (int i = tid; i < 16 * hd; i += nt) {
        int r = i / hd, d = i - r * hd;
        shQ[i] = (r < group) ? __float2half(Q[(size_t)seq * qHeads * hd + (size_t)(kvh * group + r) * hd + d]) : (half)0;
    }
    // stage K/V for [0,n) from the paged pool -> shK/shV (f16); pad [n,nkp) with 0 (ALL threads)
    for (int i = tid; i < nkp * hd; i += nt) {
        int j = i / hd, d = i - j * hd;
        if (j < n) {
            size_t phys = (size_t)bt[j / blockSize] * blockSize + (j % blockSize);
            size_t src = phys * WKV + (size_t)kvh * hd + d;
            shK[i] = __float2half(poolK[src]);
            shV[i] = __float2half(poolV[src]);
        } else {
            shK[i] = (half)0;
            shV[i] = (half)0;
        }
    }
    __syncthreads();

    __shared__ float shO[16 * 64];
    if (tid < 32) { // WMMA on warp 0 (staging above used all warps)
    // QK: scores[16, nkp] = shQ[16,hd] * shK[nkp,hd]^T, WMMA (K=hd=64 -> 4 steps)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) wmma::load_matrix_sync(qf[kk], shQ + kk * 16, hd);
    for (int jt = 0; jt < nkp; jt += 16) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> sfrag;
        wmma::fill_fragment(sfrag, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            // K tile [16 keys, 16 of hd] as col_major (so it's K^T): shK row = key, col = hd
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
            wmma::load_matrix_sync(kf, shK + (size_t)jt * hd + kk * 16, hd);
            wmma::mma_sync(sfrag, qf[kk], kf, sfrag);
        }
        wmma::store_matrix_sync(shS + jt, sfrag, MAXK, wmma::mem_row_major);
    }
    __syncwarp();

    // softmax over [0,n) per query row (rows 0..group-1); mask padded keys
    if (lane < group) {
        float* row = shS + (size_t)lane * MAXK;
        float m = -1e30f;
        for (int j = 0; j < n; j++) { float v = row[j] * scale; row[j] = v; if (v > m) m = v; }
        float sum = 0.f;
        for (int j = 0; j < n; j++) { float e = __expf(row[j] - m); row[j] = e; sum += e; }
        for (int j = n; j < nkp; j++) row[j] = 0.f;
        float inv = 1.0f / sum;
        for (int j = 0; j < n; j++) row[j] *= inv;
    }
    __syncwarp();

    // PV: O[16,hd] = P[16,nkp] * V[nkp,hd], WMMA
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[4];
    #pragma unroll
    for (int nn = 0; nn < 4; nn++) wmma::fill_fragment(of[nn], 0.0f);
    __shared__ half phalf[16 * 16];
    for (int jt = 0; jt < nkp; jt += 16) {
        for (int idx = lane; idx < 16 * 16; idx += 32) {
            int r = idx >> 4, cc = idx & 15;
            phalf[idx] = __float2half(shS[(size_t)r * MAXK + jt + cc]);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
        wmma::load_matrix_sync(pf, phalf, 16);
        #pragma unroll
        for (int nn = 0; nn < 4; nn++) {
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
            wmma::load_matrix_sync(vf, shV + (size_t)jt * hd + nn * 16, hd);
            wmma::mma_sync(of[nn], pf, vf, of[nn]);
        }
        __syncwarp();
    }
    // store the WMMA output to shared for the all-thread writeback
    #pragma unroll
    for (int nn = 0; nn < 4; nn++)
        wmma::store_matrix_sync(shO + nn * 16, of[nn], hd, wmma::mem_row_major);
    } // end warp-0 WMMA
    __syncthreads();
    // write O for the group's q-heads (rows 0..group-1), all threads
    for (int i = tid; i < group * hd; i += nt) {
        int r = i / hd, d = i - r * hd;
        O[(size_t)seq * qHeads * hd + (size_t)(kvh * group + r) * hd + d] = shO[(size_t)r * hd + d];
    }
}

// wmma_paged_decode_flash — FlashDecoding form of the WMMA paged decode: online softmax across K/V
// TILES (FTILE keys/iter) so shared memory is O(tile) not O(seqLen). The non-flash kernel above
// materializes ALL K/V (MAXK=128 cap, 42KB shared -> 1 block/SM); this tiles at ~21KB (better
// occupancy) AND handles arbitrary seqLen — the regime where the warp-GQA kernel degrades and vLLM's
// Flash-Decoding wins. Running (max, sum, unnormalized O) carried in shared across tiles; QK/PV WMMA
// per tile on warp 0. hd==64, group<=8, blockSize<=16. Q,O f32 [batch,qHeads*hd]; poolK/V f32.
#define FTILE 32
extern "C" __global__ void wmma_paged_decode_flash(const float* Q, const float* poolK, const float* poolV,
                                                   const int* blockTables, const int* seqLens, float* O,
                                                   int batch, int qHeads, int kvHeads, int hd, int blockSize,
                                                   int maxBlocks, float scale) {
    int kvh = blockIdx.x, seq = blockIdx.y;
    int group = qHeads / kvHeads, WKV = kvHeads * hd, n = seqLens[seq];
    const int* bt = blockTables + (size_t)seq * maxBlocks;
    int tid = threadIdx.x, nt = blockDim.x;

    extern __shared__ char smem[];
    half*  shQ  = (half*)smem;                 // [16*hd]
    half*  shK  = shQ + 16 * hd;               // [FTILE*hd]
    half*  shV  = shK + FTILE * hd;            // [FTILE*hd]
    float* shS  = (float*)(shV + FTILE * hd);  // [16*FTILE]
    half*  shP  = (half*)(shS + 16 * FTILE);   // [16*FTILE]
    float* shO  = (float*)(shP + 16 * FTILE);  // [16*hd] running unnormalized output
    float* shTO = shO + 16 * hd;               // [16*hd] this tile's PV output
    float* rm   = shTO + 16 * hd;              // [16] running max
    float* rl   = rm + 16;                     // [16] running sum
    float* rc   = rl + 16;                     // [16] per-tile correction

    for (int i = tid; i < 16 * hd; i += nt) {
        int r = i / hd, d = i - r * hd;
        shQ[i] = (r < group) ? __float2half(Q[(size_t)seq * qHeads * hd + (size_t)(kvh * group + r) * hd + d]) : (half)0;
    }
    for (int i = tid; i < 16; i += nt) { rm[i] = -1e30f; rl[i] = 0.f; }
    for (int i = tid; i < 16 * hd; i += nt) shO[i] = 0.f;
    __syncthreads();

    for (int base = 0; base < n; base += FTILE) {
        int nk = n - base; if (nk > FTILE) nk = FTILE;
        int nkp = (nk + 15) & ~15;
        for (int i = tid; i < FTILE * hd; i += nt) {
            int j = i / hd, d = i - j * hd;
            if (j < nk) {
                size_t phys = (size_t)bt[(base + j) / blockSize] * blockSize + ((base + j) % blockSize);
                size_t src = phys * WKV + (size_t)kvh * hd + d;
                shK[i] = __float2half(poolK[src]);
                shV[i] = __float2half(poolV[src]);
            } else { shK[i] = (half)0; shV[i] = (half)0; }
        }
        __syncthreads();
        if (tid < 32) { // QK: shS[16,nkp] = shQ[16,hd] * shK[nkp,hd]^T
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[4];
            #pragma unroll
            for (int kk = 0; kk < 4; kk++) wmma::load_matrix_sync(qf[kk], shQ + kk * 16, hd);
            for (int ct = 0; ct < nkp; ct += 16) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> sf;
                wmma::fill_fragment(sf, 0.f);
                #pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> kf;
                    wmma::load_matrix_sync(kf, shK + (size_t)ct * hd + kk * 16, hd);
                    wmma::mma_sync(sf, qf[kk], kf, sf);
                }
                wmma::store_matrix_sync(shS + ct, sf, FTILE, wmma::mem_row_major);
            }
        }
        __syncthreads();
        if (tid < group) { // online softmax update for this row over the tile's nk keys
            float* row = shS + (size_t)tid * FTILE;
            float tmax = -1e30f;
            for (int j = 0; j < nk; j++) { float v = row[j] * scale; row[j] = v; if (v > tmax) tmax = v; }
            float newm = rm[tid] > tmax ? rm[tid] : tmax;
            float corr = __expf(rm[tid] - newm);
            float s = 0.f;
            for (int j = 0; j < nk; j++) { float e = __expf(row[j] - newm); shP[(size_t)tid * FTILE + j] = __float2half(e); s += e; }
            for (int j = nk; j < FTILE; j++) shP[(size_t)tid * FTILE + j] = (half)0;
            rl[tid] = rl[tid] * corr + s; rm[tid] = newm; rc[tid] = corr;
        }
        __syncthreads();
        for (int i = tid; i < 16 * FTILE; i += nt) { if (i / FTILE >= group) shP[i] = (half)0; } // pad unused rows
        __syncthreads();
        if (tid < 32) { // PV: shTO[16,hd] = shP[16,nkp] * shV[nkp,hd]
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> of[4];
            #pragma unroll
            for (int nn = 0; nn < 4; nn++) wmma::fill_fragment(of[nn], 0.f);
            for (int ct = 0; ct < nkp; ct += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pf;
                wmma::load_matrix_sync(pf, shP + ct, FTILE);
                #pragma unroll
                for (int nn = 0; nn < 4; nn++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vf;
                    wmma::load_matrix_sync(vf, shV + (size_t)ct * hd + nn * 16, hd);
                    wmma::mma_sync(of[nn], pf, vf, of[nn]);
                }
            }
            #pragma unroll
            for (int nn = 0; nn < 4; nn++) wmma::store_matrix_sync(shTO + nn * 16, of[nn], hd, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = tid; i < group * hd; i += nt) { int r = i / hd; shO[i] = shO[i] * rc[r] + shTO[i]; }
        __syncthreads();
    }
    for (int i = tid; i < group * hd; i += nt) {
        int r = i / hd, d = i - r * hd;
        float inv = (rl[r] > 0.f) ? 1.f / rl[r] : 0.f;
        O[(size_t)seq * qHeads * hd + (size_t)(kvh * group + r) * hd + d] = shO[i] * inv;
    }
}
