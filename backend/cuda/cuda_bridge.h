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
void* cu_clone_f32(const void* src, int n);
int cu_zero_f32(void* d, int n); // zero n floats on the stream
// cu_blit: contiguous device→device copy of n floats, src[srcOff:]→dst[dstOff:].
int cu_blit(void* dst, int dstOff, const void* src, int srcOff, int n);
// cu_copy2d: copy a rows×rowFloats sub-matrix with independent src/dst row strides.
int cu_copy2d(void* dst, int dstOff, int dstStride, const void* src, int srcOff, int srcStride, int rows, int rowFloats);
// cu_argmax_f32 returns argmax over x[n] (greedy token) — downloads only the index.
int cu_argmax_f32(const void* x, int n);
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
// SwiGLU-epilogue GEMV variants (Tw55): out = silu(gate) ⊙ (a·dequant(W)), gate in out's [M,N] layout.
int cu_qmatmul_q8_swiglu(const void* dA, const void* dQ, const void* dScales, const void* dGate, void* dOut,
                         int M, int K, int N, int nb);
int cu_qmatmul_q4k_swiglu(const void* dA, const void* dQ, const void* dGate, void* dOut,
                          int M, int K, int N);
// cu_qmatmul_q3k: out[M,N] = a·dequant(W), W = ggml Q3_K 110-byte super-blocks per output row
// (symmetric, signed 6-bit sub-scales, 3-bit quants via qs low-2 + hmask high-1). K%256==0. GEMV.
int cu_qmatmul_q3k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q5k: out[M,N] = a·dequant(W), W = ggml Q5_K 176-byte super-blocks per output row
// (Q4_K's 6-bit scale/min packing + a qh high-bit plane → 5-bit quants). K%256==0. DECODE GEMV.
int cu_qmatmul_q5k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
// cu_qmatmul_q6k: out[M,N] = a·dequant(W), W = ggml Q6_K 210-byte super-blocks per output row.
int cu_qmatmul_q6k(const void* dA, const void* dQ, void* dOut, int M, int K, int N, float beta);
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
int cu_download_u16(const void* dsrc, unsigned short* dst, int n);
// cu_matmul_f16w_acc16: drop-in f32-output twin of cu_matmul_f16w with f16 accumulate (+convert).
int cu_matmul_f16w_acc16(const void* dA32, const void* dW16, void* dC32, int M, int K, int N, float beta);
// cu_copy_rows: device→device copy nElems floats from src to dst+dstOffset (KV-cache append).
int cu_copy_rows(void* dst, const void* src, int dstOffset, int nElems);
void* cu_upload_i32(const int* src, int n);
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
int cu_silu_f32(void* d, int n);
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
// cu_rope_f32_band: strided-band RoPE — rotate `heads` heads (hd wide) starting at
// float-element column `off` within rows of `stride` floats, in place. Generalises
// cu_rope_f32 for the fused-QKV path (q/k bands of one [seq,stride] buffer).
int cu_rope_f32_band(void* x, const void* inv, int seq, int stride, int off, int heads, int hd, int posOffset, double posDiv);

#endif
