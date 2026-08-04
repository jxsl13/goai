// CUDA/cuBLAS bridge for the optional cuda backend (§T42). Compiled only under
// `-tags cuda` with cgo on linux/windows. Synchronous API: each call copies
// H2D, runs cublasSgemm, copies D2H and syncs before returning (async batching
// and device-resident tensors are a later optimization; §V14 keeps the interface
// stable so that can land without an API break).
#ifndef GOAI_CUDA_BRIDGE_H
#define GOAI_CUDA_BRIDGE_H

// cu_available returns 1 if at least one CUDA-capable GPU is present.
int cu_available(void);
// cu_mem_info: device free/total VRAM in bytes (VRAM-budget probe for T631 offload).
int cu_mem_info(unsigned long long* freeB, unsigned long long* totalB);
// cu_gpu_is_geforce: 1 if device 0 is a GeForce/consumer card (half-rate f32-accum → f16-accum wins).
int cu_gpu_is_geforce(void);
// cu_matmul_i8_mma: dC[M,N](i32) = dA8[M,K]·dW8[K,N] via tiled mma.sync int8 tensor cores.
int cu_matmul_i8_mma(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// cu_gemm_w8a16: C[M,N]f16 = A[M,K]f16 · dequant(W[K,N]int8, per-col f32 Scale). Dequant-in-tile mma.
int cu_gemm_w8a16(const void* dA16, const void* dW8, const void* dScale, void* dC16, int M, int K, int N);
// cu_gemm_w8a16_t: shared-tiled W8A16 (coalesced int8 staging, W read once/block). N%64==0.
int cu_gemm_w8a16_t(const void* dA16, const void* dW8, const void* dScale, void* dC16, int M, int K, int N);
// cu_gemm_w8a16_b: BM-spanning W8A16 (64x64 tile, W read once/N-strip). M%64==0, N%64==0.
int cu_gemm_w8a16_b(const void* dA16, const void* dW8, const void* dScale, void* dC16, int M, int K, int N);
// cu_gemm_w8a16_d: double-buffered (cp.async pipeline) W8A16 — overlaps weight load with compute.
int cu_gemm_w8a16_d(const void* dA16, const void* dW8, const void* dScale, void* dC16, int M, int K, int N);
// cu_gemm_w8a16_sk: split-K W8A16 (occupancy fix). dCacc = scratch f32 [M*N] (zeroed internally).
int cu_gemm_w8a16_sk(const void* dA16, const void* dW8, const void* dScale, void* dCacc, void* dC16, int M, int K, int N, int splitK);
// cu_gemm_w8a16_p3: 3-stage cp.async pipeline + split-K W8A16 (deeper pipeline test). dCacc scratch f32.
int cu_gemm_w8a16_p3(const void* dA16, const void* dW8, const void* dScale, void* dCacc, void* dC16, int M, int K, int N, int splitK);
// cu_matmul_i8_mma_t: shared-tiled int8 mma GEMM (16x64 block, shared A/W staging). N%64==0.
int cu_matmul_i8_mma_t(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mma_rb: register-blocked int8 mma GEMM (64x64 block, 4 MMAs/warp). M%64,N%64,K%32==0.
int cu_matmul_i8_mma_rb(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mma_db: double-buffered cp.async int8 mma GEMM (64x64 block). M%64,N%64,K%32==0.
int cu_matmul_i8_mma_db(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mma_wt: like _db but W is [N][K] native layout -> contiguous B reads (no bank conflict). M%64,N%64,K%32==0.
int cu_matmul_i8_mma_wt(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mma_wp: _wt with 48-byte padded shared stride (dissolves residual 2-way bank conflict). M%64,N%64,K%32==0.
int cu_matmul_i8_mma_wp(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mma_lm: _wp with ldmatrix.x4/x2 fragment loads. M%64,N%64,K%32==0.
int cu_matmul_i8_mma_lm(const void* dA8, const void* dWt8, void* dC32, int M, int K, int N);
// cu_matmul_i8_mmq: true per-32-block MMQ (int8 A/W + per-block f32 scales -> f32 C). M%64,N%64,K%32==0.
int cu_matmul_i8_mmq(const void* dA8, const void* dWt8, const void* daSc, const void* dwSc, void* dCf, int M, int K, int N);
// cu_matmul_i8_mmq_r: MMQ with per-ROW activation scale aSc[M] (hoisted from K-loop). M%64,N%64,K%32==0.
int cu_matmul_i8_mmq_r(const void* dA8, const void* dWt8, const void* daSc, const void* dwSc, void* dCf, int M, int K, int N);
// cu_quant_rows_i8: device per-row int8 quant of f32 activations (A8[M][K]+aSc[M]) for MMQ input.
int cu_quant_rows_i8(const void* dAf, void* dA8, void* daSc, int M, int K);
// CUDA graph capture: begin/end capture on the work stream, launch/sync/free the
// instantiated graph. Caller must LockOSThread across begin→ops→end.
int cu_capture_begin(void);
void* cu_capture_end(void);
int cu_graph_launch(void* exec);
int cu_graph_sync(void);
void cu_graph_free(void* exec);

// cu_matmul_f32 computes C[M,N] = A[M,K]·B[K,N], all row-major float32.
// Returns 0 on success, nonzero on failure (see cuda_bridge.c for codes).
int cu_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N);

// Resident-B matmul (§V14 Phase-1, mirrors the metal §T156 resident-weight seed):
// upload a weight B[K,N] to the GPU ONCE, then reuse it across many matmuls,
// skipping its per-call H2D. This is the transfer lever for inference, where the
// weight is fixed and only the activation A varies.
//
// cu_upload_f32 copies n row-major floats to a fresh device buffer and returns an
// opaque device handle (NULL on failure). cu_free_f32 releases it (NULL-safe).
void* cu_upload_f32(const float* src, int n);
void cu_free_f32(void* dptr);

// cu_matmul_f32_bres computes C[M,N] = A[M,K]·dB[K,N] with A and C host-side and
// dB a resident handle from cu_upload_f32 (its element count must be K*N). A and
// C use the same pooled device buffers as cu_matmul_f32. Returns 0 on success.
int cu_matmul_f32_bres(const float* A, const void* dB, float* C, int M, int K, int N);

// Fully-device matmul (§V14 Phase-2, activation residency): all three operands
// are device handles, so a chain of matmuls keeps its intermediates on the GPU —
// only the first activation upload and the final download touch host memory.
//
// cu_alloc_f32 returns an uninitialized device buffer of n floats (NULL on fail);
// cu_download_f32 copies n floats device→host. cu_matmul_f32_ddd computes
// dC[M,N] = dA[M,K]·dB[K,N] with every operand resident (no H2D/D2H).
void* cu_alloc_f32(int n);
// cu_wmma_gemm: nvcc-compiled WMMA tensor-core GEMM, C[M,N]f32 = A[M,K]f16·B[K,N]f16 (M,N,K %16==0).
int cu_wmma_gemm(const void* fatbin, int fatlen, const void* dA, const void* dB, void* dC, int M, int K, int N);
// cu_wmma_attn: fused prefill attention (softmax(scale·QKᵀ+causal)·V) on tensor cores. hd==64, seq%16==0.
int cu_wmma_attn(const void* fatbin, int fatlen, const void* dQ, const void* dK, const void* dV, void* dO, int heads, int seq, int hd, float scale);
// cu_paged_decode_attn: batched single-query decode attention over a paged KV pool (FRONT B / B2). hd==64.
int cu_paged_decode_attn(const void* dQ, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_paged_decode_attn_gqa: same contract, GQA K/V-shared (one block per (kv head, seq), group warps,
// K/V staged into shared once per tile) — cuts the naive kernel's group× redundant K/V reads. hd==64, group≤8, blockSize≤16.
int cu_paged_decode_attn_gqa(const void* dQ, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_paged_decode_attn_gqa_qio: f32-KV GQA with f16 Q-in/O-out — kills serving-path Q/O converts, no accuracy change.
int cu_paged_decode_attn_gqa_qio(const void* dQ16, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO16, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_paged_decode_attn_gqa_sk: split-K (FlashDecoding) — splitK blocks per (kv head, seq) + merge; parallelizes the online-softmax scan. splitK 1..32.
int cu_paged_decode_attn_gqa_sk(const void* dQ, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale, int splitK);
// cu_wmma_paged_decode: tensor-core (nvcc mma.h fatbin) batched paged decode attention. hd==64, group≤8, blockSize≤16, seqLen≤128.
int cu_wmma_paged_decode(const void* fatbin, int fatlen, const void* dQ, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);

// cu_paged_append_batched: device-side batched paged KV append. Scatter dK/dV [batch,wkv] into each
// sequence's slot seqLens[b] (pre-append length) — the real serving append with no host round-trip.
int cu_paged_append_batched(void* dPoolK, void* dPoolV, const void* dBlockTables, const void* dSeqLens, const void* dK, const void* dV, int batch, int wkv, int blockSize, int maxBlocks);

// cu_wmma_paged_decode_flash: tiled FlashDecoding WMMA paged decode (O(tile) shared, any seqLen). hd==64, group<=8, blockSize<=16.
int cu_wmma_paged_decode_flash(const void* fatbin, int fatlen, const void* dQ, const void* dPoolK, const void* dPoolV, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_cvt_f32_to_f16: convert n device f32 (src32) to device f16/u16 (dst16), stream-ordered.
int cu_cvt_f32_to_f16(void* dst16, const void* src32, long n);
int cu_cvt_f16_to_f32(void* dst32, const void* src16, long n);
int cu_addf16_to_f32(void* dst32, const void* src16, long n); // f32-residual A1: dst32 += f16(src16)
// cu_gemm_int8: C32[M,N] int32 = A8[M,K]·W8[K,N] int8 via cublasGemmEx IMMA (int8 tensor cores).
int cu_gemm_int8(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// A1 fp16-activation elementwise twins — in/out are u16 (f16); gamma/inv stay f32. Math == the f32 kernels.
int cu_rmsnorm_f16(const void* in, void* out, const void* gamma, int rows, int cols, float eps);
int cu_swiglu_f16(void* gate, const void* up, int n);
int cu_rope_f16(void* x, const void* inv, int seq, int heads, int hd, int posOffset, double posDiv);
int cu_add_f16(void* dst, const void* src, int n); // A1 f16 residual add (dst += src, u16)
int cu_rope_f16_dpos(void* x, const void* inv, int seq, int heads, int hd, const void* dPos, double posDiv); // f16 device-position RoPE (graph decode)
int cu_rope_f16_dpos_arr(void* x, const void* inv, int seq, int heads, int hd, const void* dPosArr, double posDiv); // f16 PER-SEQ-position RoPE (continuous batching)
// cu_paged_decode_attn_gqa_f16: f16-KV twin of cu_paged_decode_attn_gqa (poolK16/V16 are u16, half the global bytes).
int cu_paged_decode_attn_gqa_f16(const void* dQ, const void* dPoolK16, const void* dPoolV16, const void* dBlockTables, const void* dSeqLens, void* dO, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_paged_decode_attn_gqa_f16_qio: f16-KV + f16 Q-in/O-out — kills the A1 per-layer Q/O conversions.
int cu_paged_decode_attn_gqa_f16_qio(const void* dQ16, const void* dPoolK16, const void* dPoolV16, const void* dBlockTables, const void* dSeqLens, void* dO16, int batch, int qHeads, int kvHeads, int hd, int blockSize, int maxBlocks, float scale);
// cu_wmma_attn_gqa: fused prefill attention on f32 DEVICE buffers, [seq,heads·hd] GQA layout — drop-in for GroupedQueryAttention.
int cu_wmma_attn_gqa(const void* fatbin, int fatlen, const void* dQ32, const void* dK32, const void* dV32, void* dO32, int seq, int qHeads, int kvHeads, int hd, float scale);
// cu_wmma_attn_gqa_f16: same but Q/K/V already f16 — skips the internal f32->f16 convert. Output f32.
int cu_wmma_attn_gqa_f16(const void* fatbin, int fatlen, const void* dQ16, const void* dK16, const void* dV16, void* dO32, int seq, int qHeads, int kvHeads, int hd, float scale);
void* cu_clone_f32(const void* src, int n);
int cu_zero_f32(void* d, int n); // zero n floats on the stream
// cu_blit: contiguous device→device copy of n floats, src[srcOff:]→dst[dstOff:].
int cu_blit(void* dst, int dstOff, const void* src, int srcOff, int n);
// cu_copy2d: copy a rows×rowFloats sub-matrix with independent src/dst row strides.
int cu_copy2d(void* dst, int dstOff, int dstStride, const void* src, int srcOff, int srcStride, int rows, int rowFloats);
// cu_argmax_f32 returns argmax over x[n] (greedy token) — downloads only the index.
int cu_argmax_f32(const void* x, int n);
int cu_argmax_batched_f16(const void* x16, int* hostOut, int rows, int cols); // per-row greedy argmax over f16 logits (serving sampling)
// cu_upload_i8: upload n signed bytes (Q8 weights) to a fresh device buffer.
void* cu_upload_i8(const signed char* src, int n);
// cu_qmatmul_q8: out[M,N] = a[M,K]·dequant(W), W = transposed Q8 q[N,K] + per-32-block scales[N,nb].
int cu_qmatmul_q8(const void* dA, const void* dQ, const void* dScales, void* dOut, int M, int K, int N, int nb, float beta);
// cu_qmatmul_q4: out[M,N] = a·dequant(W4), W4 = asymmetric Q4 (packed nibbles q[N,K/2] +
// per-32-block f32 scale + f32 min), dequant w = min + nibble·scale. K must be a mult of 256.
int cu_qmatmul_q4(const void* dA, const void* dQ, const void* dScales, const void* dMins, void* dOut, int M, int K, int N, int nb, float beta);
// cu_qmatmul_q4k: out[M,N] = a·dequant(W), W = ggml Q4_K super-blocks stored per output row —
// q[N, K/256 * 144] (144-byte blocks: f16 d + f16 dmin + 12B packed 6-bit scales/mins + 128B
// nibbles), dequant y = d*sc6*nibble - dmin*min6 per 32-sub-block. K%256==0. DECODE-ONLY (GEMV).
int cu_qmatmul_q4k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_dequant_q4k_to_f16: expand Q4_K weight -> contiguous f16 [K,N] for the tensor-core prefill GEMM.
int cu_dequant_q4k_to_f16(const void* dQ, void* dBf16, int K, int N);
// cu_qmatmul_q4k_mt: weight-read-once M-tiled GEMM for M>1 — one warp owns a column and an
// MT-row tile, decoding each Q4_K sub-block ONCE and reusing it across rows (weight BW M/MT×
// lower than the per-(m,n) GEMV). Bit-identical arithmetic to cu_qmatmul_q4k. K%256==0.
int cu_qmatmul_q4k_mt(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q4k_pre: Q4_K with pre-decoded f32 sub-block scales (192-byte blocks: 64B scale
// plane + 128B nibbles). Bit-exact vs cu_qmatmul_q4k, no in-kernel scale unpack (R5). K%256==0.
int cu_qmatmul_q4k_pre(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// SwiGLU-epilogue GEMV variants (Tw55): out = silu(gate) ⊙ (a·dequant(W)), gate in out's [M,N] layout.
int cu_qmatmul_q8_swiglu(const void* dA, const void* dQ, const void* dScales, const void* dGate, void* dOut,
                         int M, int K, int N, int nb);
int cu_qmatmul_q4k_swiglu(const void* dA, const void* dQ, const void* dGate, void* dOut,
                          int M, int K, int N);
// cu_qmatmul_iq4nl / iq4xs: out = a·dequant(W), W = ggml IQ4_NL (18-byte blocks) / IQ4_XS
// (136-byte super-blocks) — 4-bit quants over a nonlinear 16-value codebook. K%32 / K%256.
int cu_qmatmul_iq4nl(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta);
int cu_qmatmul_iq4xs(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq4xs_mt: weight-read-once M-tiled GEMM for M>1 — IQ4_XS twin of cu_qmatmul_q4k_mt.
// Bit-identical arithmetic to cu_qmatmul_iq4xs. K%256==0.
int cu_qmatmul_iq4xs_mt(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_mxfp4: out = a·dequant(MXFP4, gpt-oss), REPACKED into dScale (nblk E8M0 bytes/row)
// + dNib (nblk×16 nibble bytes/row, 16-aligned) for coalesced reads. K%32==0. DECODE GEMV.
int cu_qmatmul_mxfp4(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_mxfp4_mt: weight-read-once M-tiled GEMM for M>1 — MXFP4 (gpt-oss) twin of the M-tile
// (each block's scale + FP4 codebook values decoded once per warp, reused across the row tile). K%32==0.
int cu_qmatmul_mxfp4_mt(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q40: out[M,N] = a·dequant(Q4_0), REPACKED into dScale (nblk f16/row) + dNib
// (nblk×16 nibble bytes/row, 16-aligned) for coalesced reads. y = d·(nibble−8). K%32==0. GEMV.
int cu_qmatmul_q40(const void* dA, const void* dScale, const void* dNib, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q2k: out[M,N] = a·dequant(W), W = ggml Q2_K 84-byte super-blocks per output row
// (asymmetric affine, 4-bit sub-scale+min nibbles, 2-bit quants). K%256==0. DECODE GEMV.
int cu_qmatmul_q2k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q2k_mt: weight-read-once M-tiled GEMM for M>1 — Q2_K twin of cu_qmatmul_q4k_mt.
// Bit-identical arithmetic to cu_qmatmul_q2k. K%256==0.
int cu_qmatmul_q2k_mt(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q3k: out[M,N] = a·dequant(W), W = ggml Q3_K 110-byte super-blocks per output row
// (symmetric, signed 6-bit sub-scales, 3-bit quants via qs low-2 + hmask high-1). K%256==0. GEMV.
int cu_qmatmul_q3k(const void* dA, const void* dMeta, const void* dQs, const void* dHm, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q3k_mt: weight-read-once M-tiled GEMM for M>1 — Q3_K twin of cu_qmatmul_q4k_mt.
// Bit-identical arithmetic to cu_qmatmul_q3k. K%256==0.
int cu_qmatmul_q3k_mt(const void* dA, const void* dMeta, const void* dQs, const void* dHm, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q5k: out[M,N] = a·dequant(W), W = ggml Q5_K 176-byte super-blocks per output row
// (Q4_K's 6-bit scale/min packing + a qh high-bit plane → 5-bit quants). K%256==0. DECODE GEMV.
int cu_qmatmul_q5k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q5k_mt: weight-read-once M-tiled GEMM for M>1 — Q5_K twin of cu_qmatmul_q4k_mt.
// Bit-identical arithmetic to cu_qmatmul_q5k. K%256==0.
int cu_qmatmul_q5k_mt(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq2xxs: out = a·dequant(W), W = ggml IQ2_XXS (66-byte super-blocks) — the first
// GRID-codebook i-quant. dGrid = the shared 256×8 float grid (device buffer). K%256==0. GEMV.
int cu_qmatmul_iq2xxs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq2xxs_mt: weight-read-once M-tiled GEMM for M>1 — IQ2_XXS twin of cu_qmatmul_q4k_mt
// (grid decoded once per warp, reused across the row tile). Bit-identical arithmetic. K%256==0.
int cu_qmatmul_iq2xxs_mt(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq2xs: IQ2_XS (74-byte super-blocks, 512×8 grid + explicit 4-bit scales). K%256==0.
int cu_qmatmul_iq2xs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq3xxs: IQ3_XXS (98-byte super-blocks, 256×4 grid + packed ksigns/scale). K%256==0.
int cu_qmatmul_iq3xxs(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq3xxs_mt: weight-read-once M-tiled GEMM for M>1 — IQ3_XXS twin of cu_qmatmul_q4k_mt
// (grid decoded once per warp, reused across the row tile). Bit-identical arithmetic. K%256==0.
int cu_qmatmul_iq3xxs_mt(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq3s: IQ3_S (110-byte super-blocks, 512×4 grid, 9-bit indices, direct signs). K%256==0.
int cu_qmatmul_iq3s(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq3s_mt: weight-read-once M-tiled GEMM for M>1 — IQ3_S twin of cu_qmatmul_q4k_mt
// (grid decoded once per warp, reused across the row tile). Bit-identical arithmetic. K%256==0.
int cu_qmatmul_iq3s_mt(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq1s: IQ1_S (50-byte super-blocks, 2048×8 ternary grid + ±δ + odd multipliers). K%256==0.
int cu_qmatmul_iq1s(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_iq1m: IQ1_M (56-byte super-blocks, same 2048×8 grid, split-f16 super-scale + sub-scales). K%256==0.
int cu_qmatmul_iq1m(const void* dA, const void* dQ, const void* dGrid, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q6k: out[M,N] = a·dequant(W), W = ggml Q6_K 210-byte super-blocks per output row.
int cu_qmatmul_q6k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q6k_mt: weight-read-once M-tiled GEMM for M>1 — Q6_K twin of cu_qmatmul_q4k_mt,
// decodes each Q6_K sub-block once per warp and reuses across an MT-row tile. Bit-identical. K%256==0.
int cu_qmatmul_q6k_mt(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_upload_f16: upload host f32, convert to a device f16 (u16) buffer of n elements
// (the f32 staging buffer is freed). For resident prefill weights (tensor-core GEMM).
void* cu_upload_f16(const float* src, long n);
// cu_matmul_f16w: dC[M,N] (f32) = dA32[M,K] (f32, converted to f16 in a stream-ordered
// scratch) x dW16[K,N] (resident f16), cublasGemmEx f16 inputs / f32 accumulate —
// the Ampere tensor-core path (~2x the Sgemm rate). beta in {0,1} (1 = residual fuse).
int cu_matmul_f16w(const void* dA32, const void* dW16, void* dC32, int M, int K, int N, float beta);
// cu_matmul_i8_lt: int8 tensor-core GEMM (cublasLt, COMPUTE_32I) — the Tw61 prefill lever probe.
void* cu_alloc_i8(int n);
int cu_matmul_i8_lt(const void* dA8, const void* dW8, void* dC32, int M, int K, int N);
// cu_matmul_f16acc16: f16 GEMM with f16 ACCUMULATE (COMPUTE_16F, ≈1.5-2× on GeForce), f16 out.
int cu_matmul_f16acc16(const void* dA32, const void* dW16, void* dC16, int M, int K, int N);
// cu_ldmatrix_probe: empirically map ldmatrix.x4.b16 fragment layout (dOut = 128 u32).
int cu_ldmatrix_probe(void* dOut);
// cu_ldmatrix_probe2: map ldmatrix.x2.b16 for the B fragment (dOut = 64 u32).
int cu_ldmatrix_probe2(void* dOut);
int cu_download_u16(const void* dsrc, unsigned short* dst, int n);
// cu_matmul_f16w_acc16: drop-in f32-output twin of cu_matmul_f16w with f16 accumulate (+convert).
int cu_matmul_f16w_acc16(const void* dA32, const void* dW16, void* dC32, int M, int K, int N, float beta);
// cu_gemm_f16_pure: pure f16 GEMM (f16 in/out, no per-call conversions) — isolates the A1 conversion cost.
int cu_gemm_f16_pure(const void* dA16, const void* dW16, void* dC16, int M, int K, int N);
// cu_gemm_f16_pure_acc32: pure f16 GEMM but f32 ACCUMULATE (COMPUTE_32F) — vLLM-precision, no conversions.
int cu_gemm_f16_pure_acc32(const void* dA16, const void* dW16, void* dC16, int M, int K, int N);
// cu_gemm_f16_pure_addc: f16 GEMM with beta=1 (C += A·B) — folds a residual add into the GEMM epilogue.
int cu_gemm_f16_pure_addc(const void* dA16, const void* dW16, void* dC16, int M, int K, int N);
// cu_copy_rows: device→device copy nElems floats from src to dst+dstOffset (KV-cache append).
int cu_copy_rows(void* dst, const void* src, int dstOffset, int nElems);
void* cu_upload_i32(const int* src, int n);
// cu_live_bufs: number of caller-owned device buffers currently outstanding (allocated by the
// cu_upload_*/cu_alloc_*/cu_clone_* family and not yet freed via cu_free_f32). A balanced Go
// workload returns this to its starting value; the leak tests assert on that. Not a byte count.
long cu_live_bufs(void);
int cu_update_i32(void* dst, const int* src, int n); // in-place H2D update of an existing device int buffer (persistent view for graph decode)
int cu_bump_i32(void* buf, int n, int delta); // buf[i]+=delta on-device (capturable length-bump for correct in-graph decode)
// cu_upload_i32_async: like cu_upload_i32 but WITHOUT a cudaStreamSynchronize. Safe when the
// uploaded buffer is consumed only by later ops on gStream (stream-ordered) — the pageable H2D
// copy is host-blocking so the source slice is free after return. Lets a decode step upload its
// block-table/seq-len ONCE and feed all layers without a per-layer device sync.
void* cu_upload_i32_async(const int* src, int n);
// Device-position (graph-capturable) op twins: position lives in a device int.
int cu_set_i32(void* d, int val);
int cu_rope_f32_dpos(void* x, const void* inv, int seq, int heads, int hd, const void* dPos, double posDiv);
int cu_attn_softmax_dpos(void* x, int rows, int cols, float scale, const void* dOff, int seqQ);
int cu_append_dpos(void* dst, const void* src, const void* dPos, int wkv);
// Flash decode attention (seqQ==1): GQA K/V-shared split-K online-softmax partials + merge.
int cu_gqa_flash_dpos(const void* dQ, const void* dK, const void* dV, void* dOut,
                      int seqKV, int qHeads, int kvHeads, int hd, float scale, const void* dOff);
// f16 KV-cache twins: u16 storage (half the K/V bytes), f32 compute in shared.
void* cu_alloc_u16(int n);
int cu_zero_u16(void* d, int n);
int cu_append_dpos_f16(void* dst16, const void* src, const void* dPos, int wkv);
int cu_gqa_flash_f16_dpos(const void* dQ, const void* dK16, const void* dV16, void* dOut,
                          int seqKV, int qHeads, int kvHeads, int hd, float scale, const void* dOff);
// out[i,:] = table[ids[i],:] — input embedding row gather (table [vocab,d] resident).
int cu_embed_f32(const void* dTable, const void* dIds, void* dOut, int seq, int d);
int cu_download_f32(const void* dsrc, float* dst, int n);
// cu_upload_into: H2D copy n floats into an existing device buffer (pointer kept).
int cu_upload_into(void* dst, const float* src, int n);
int cu_matmul_f32_ddd(const void* dA, const void* dB, void* dC, int M, int K, int N);
// dC = dA·dB + dC (beta=1): fuses the residual add into the projection matmul.
int cu_matmul_f32_ddd_acc(const void* dA, const void* dB, void* dC, int M, int K, int N);
// dC[M,N] = dA[M,K]·dB[N,K]ᵀ, all resident (attention QKᵀ).
int cu_matmul_f32_ddd_bt(const void* dA, const void* dB, void* dC, int M, int K, int N);

// Multi-head attention (batched strided). Q/K/V are [seq, heads*hd]; scores is
// [heads, seqQ, seqKV]. cu_mha_scores = batched Q·Kᵀ; cu_causal_scale_mh = per-head
// scale+causal-mask; cu_mha_out = batched scores·V into [seqQ, heads*hd].
int cu_mha_scores(const void* dQ, const void* dK, void* dScores, int seq, int heads, int hd);
int cu_causal_scale_mh(void* x, int heads, int seqQ, int seqKV, float scale, int offset);
// cu_attn_softmax: fused scale + causal-mask + softmax over scores[rows, cols]
// (rows = heads·seqQ, cols = seqKV) — one launch replacing scale-mask + softmax.
int cu_attn_softmax(void* x, int rows, int cols, float scale, int offset, int seqQ);
int cu_attn_softmax_cap(void* x, int rows, int cols, float scale, int offset, int seqQ, float cap); // Gemma2 attn-logit soft-cap
int cu_attn_softmax_alibi(void* x, int rows, int cols, float scale, int offset, int seqQ, const void* slopes); // MPT ALiBi position bias
int cu_attn_softmax_bias(void* x, int rows, int cols, float scale, int offset, int seqQ, const void* bias); // T5 per-head relative-position bias [heads,seqQ,seqKV]
int cu_mha_out(const void* dScores, const void* dV, void* dOut, int seq, int heads, int hd);

// GQA: qHeads query heads share kvHeads kv heads (query h → kv head h/group).
// Pointer-array batched Sgemm; Q is [seqQ,WQ], K/V are [seqKV,WKV], scores are
// [qHeads, seqQ, seqKV]. Full prefill passes seqQ==seqKV; a KV-cache decode step
// passes seqQ new query rows against seqKV cached keys/values.
int cu_gqa_scores(const void* dQ, const void* dK, void* dScores, int seqQ, int seqKV, int qHeads, int kvHeads, int hd, int tf32);
int cu_gqa_out(const void* dScores, const void* dV, void* dOut, int seqQ, int seqKV, int qHeads, int kvHeads, int hd, int tf32);

// On-device elementwise op (§V14 Phase-2, breadth beyond matmul). The kernel is
// compiled at runtime from CUDA-C source via nvrtc (no nvcc needed) and launched
// on the same stream as the matmuls, so a matmul→activation→matmul block stays
// fully resident. cu_gelu_f32 applies exact GELU (0.5·x·(1+erf(x/√2))) in-place;
// cu_silu_f32 applies SiLU (x·sigmoid(x)) in-place; cu_add_f32 does dst += src
// (residual). All operate on n floats, in-place, on the stream; return 0 on ok.
int cu_gelu_f32(void* d, int n);
int cu_moe_gate(const void* logits, void* weights, int rows, int E, int K, int raw, float scale); // top-k routing (raw=DeepSeek softmax·scale, else renorm)
int cu_row_axpy(void* dst, const void* src, const void* arow, int rows, int cols); // per-row scalar AXPY (MoE combine)
int cu_ssm_step(const void* u, const void* delta, const void* A, const void* B, const void* C, const void* dskip, void* h, void* y, int D, int N); // Mamba selective-scan decode step
int cu_ssd_step(const void* x, const void* delta, const void* A, const void* B, const void* C, const void* dskip, void* state, void* y, int H, int P, int G, int N); // Mamba-2 SSD (scalar-decay, grouped B/C) decode step
int cu_conv1d_step(const void* x, const void* w, const void* b, void* state, void* out, int D, int K); // Mamba causal depthwise conv decode step
int cu_wkv_step(const void* k, const void* v, const void* w, const void* u, void* aa, void* bb, void* pp, void* out, int D); // RWKV-4 WKV recurrence decode step
int cu_relu2_f32(void* d, int n); // squared ReLU (Nemotron relu2): relu(x)² in-place
int cu_relu_f32(void* d, int n); // plain ReLU (T5 v1.0 FFN): max(x,0) in-place
int cu_silu_f32(void* d, int n);
int cu_sigmoid_f32(void* d, int n); // plain sigmoid (Qwen2-MoE shared-expert gate)
int cu_softplus_f32(void* d, int n); // softplus (Mamba Δ)
int cu_add_f32(void* dst, const void* src, int n);
int cu_mul_f32(void* dst, const void* src, int n);
// gate[i] = SiLU(gate[i])*up[i], fused (SwiGLU) in one pass.
int cu_swiglu_f32(void* gate, const void* up, int n);

// cu_rmsnorm_f32 applies RMSNorm y = x/√(mean(x²)+eps)·gamma in-place over the
// last axis (x is rows×cols row-major; gamma is a resident [cols] weight).
int cu_rmsnorm_f32(const void* in, void* out, const void* gamma, int rows, int cols, float eps);
// cu_layernorm_f32: out = (x−mean)·inv·gamma + beta over the last axis (OpLayerNorm).
int cu_layernorm_f32(const void* in, void* out, const void* gamma, const void* beta, int rows, int cols, float eps);
// cu_addbias_f32: out[r,j] = x[r,j] + bias[j] (row-broadcast, OpAddBias).
int cu_addbias_f32(const void* x, const void* bias, void* out, int rows, int n);

// cu_softmax_f32 applies stable softmax over the last axis in-place (rows×cols).
int cu_softmax_f32(void* x, int rows, int cols);

// cu_causal_scale_f32 scales attention scores[qRows,kCols] by scale and applies
// a causal mask (j > i + offset → −inf) in-place, ready for softmax.
int cu_causal_scale_f32(void* x, int qRows, int kCols, float scale, int offset);

// cu_rope_f32 applies rotary position embeddings in-place to x[seq, heads*hd]
// (HF rotate_half). inv is the resident [hd/2] frequency table and posDiv the
// position divisor (both from backend.RoPEFreqs on the host).
int cu_rope_f32(void* x, const void* inv, int seq, int heads, int hd, int posOffset, double posDiv);
// cu_rope_partial applies PARTIAL rotary in-place: only the first rotaryDim channels of each
// head are rotated (rotate_half), the rest pass through. inv is the resident [rotaryDim/2]
// frequency table. cu_rope_f32 is the rotaryDim==hd case (GPT-NeoX/Phi/StableLM partial rotary).
int cu_rope_partial(void* x, const void* inv, int seq, int heads, int hd, int rotaryDim, int posOffset, double posDiv);
// cu_rope_partial_dpos: device-position twin of cu_rope_partial — partial rotary reading the
// posOffset from device int *dPos (graph-capturable decode of partial-rotary architectures).
int cu_rope_partial_dpos(void* x, const void* inv, int seq, int heads, int hd, int rotaryDim, const void* dPos, double posDiv);
// cu_rope_f32_band: strided-band RoPE — rotate `heads` heads (hd wide) starting at
// float-element column `off` within rows of `stride` floats, in place. Generalises
// cu_rope_f32 for the fused-QKV path (q/k bands of one [seq,stride] buffer).
int cu_rope_f32_band(void* x, const void* inv, int seq, int stride, int off, int heads, int hd, int posOffset, double posDiv);
// cu_rope_partial_band: cu_rope_f32_band with only the first rotaryDim channels of each head
// rotated (partial rotary; the fused-QKV band path for GPT-NeoX/Phi/StableLM). inv=[rotaryDim/2].
int cu_rope_partial_band(void* x, const void* inv, int seq, int stride, int off, int heads, int hd, int rotaryDim, int posOffset, double posDiv);

#endif
