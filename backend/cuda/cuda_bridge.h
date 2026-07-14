// CUDA/cuBLAS bridge for the optional cuda backend (§T42). Compiled only under
// `-tags cuda` with cgo on linux/windows. Synchronous API: each call copies
// H2D, runs cublasSgemm, copies D2H and syncs before returning (async batching
// and device-resident tensors are a later optimization; §V14 keeps the interface
// stable so that can land without an API break).
#ifndef GOAI_CUDA_BRIDGE_H
#define GOAI_CUDA_BRIDGE_H

// cu_available returns 1 if at least one CUDA-capable GPU is present.
int cu_available(void);

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
void* cu_upload_i32(const int* src, int n);
// out[i,:] = table[ids[i],:] — input embedding row gather (table [vocab,d] resident).
int cu_embed_f32(const void* dTable, const void* dIds, void* dOut, int seq, int d);
int cu_download_f32(const void* dsrc, float* dst, int n);
int cu_matmul_f32_ddd(const void* dA, const void* dB, void* dC, int M, int K, int N);
// dC[M,N] = dA[M,K]·dB[N,K]ᵀ, all resident (attention QKᵀ).
int cu_matmul_f32_ddd_bt(const void* dA, const void* dB, void* dC, int M, int K, int N);

// Multi-head attention (batched strided). Q/K/V are [seq, heads*hd]; scores is
// [heads, seqQ, seqKV]. cu_mha_scores = batched Q·Kᵀ; cu_causal_scale_mh = per-head
// scale+causal-mask; cu_mha_out = batched scores·V into [seqQ, heads*hd].
int cu_mha_scores(const void* dQ, const void* dK, void* dScores, int seq, int heads, int hd);
int cu_causal_scale_mh(void* x, int heads, int seqQ, int seqKV, float scale, int offset);
int cu_mha_out(const void* dScores, const void* dV, void* dOut, int seq, int heads, int hd);

// GQA: qHeads query heads share kvHeads kv heads (query h → kv head h/group).
// Pointer-array batched Sgemm; Q is [seqQ,WQ], K/V are [seqKV,WKV], scores are
// [qHeads, seqQ, seqKV]. Full prefill passes seqQ==seqKV; a KV-cache decode step
// passes seqQ new query rows against seqKV cached keys/values.
int cu_gqa_scores(const void* dQ, const void* dK, void* dScores, int seqQ, int seqKV, int qHeads, int kvHeads, int hd);
int cu_gqa_out(const void* dScores, const void* dV, void* dOut, int seqQ, int seqKV, int qHeads, int kvHeads, int hd);

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
int cu_rmsnorm_f32(void* x, const void* gamma, int rows, int cols, float eps);

// cu_softmax_f32 applies stable softmax over the last axis in-place (rows×cols).
int cu_softmax_f32(void* x, int rows, int cols);

// cu_causal_scale_f32 scales attention scores[qRows,kCols] by scale and applies
// a causal mask (j > i + offset → −inf) in-place, ready for softmax.
int cu_causal_scale_f32(void* x, int qRows, int kCols, float scale, int offset);

// cu_rope_f32 applies rotary position embeddings in-place to x[seq, heads*hd]
// (HF rotate_half). inv is the resident [hd/2] frequency table and posDiv the
// position divisor (both from backend.RoPEFreqs on the host).
int cu_rope_f32(void* x, const void* inv, int seq, int heads, int hd, int posOffset, double posDiv);

#endif
