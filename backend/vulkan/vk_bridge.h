// Vulkan compute bridge for the optional vulkan backend (§T43). Compiled only
// under `-tags vulkan` with cgo. Synchronous API: each call uploads inputs, runs
// the compute shader, and vkQueueWaitIdle before reading the result back
// (async/device-resident tensors are a later optimization; §V14 keeps the
// interface stable so that lands without an API break).
#ifndef GOAI_VK_BRIDGE_H
#define GOAI_VK_BRIDGE_H

#include <stdint.h>

// vk_available returns 1 if a Vulkan instance with a compute-capable physical
// device is present.
int vk_available(void);

// vk_matmul_f32 computes C[M,N] = opA(A)·opB(B), row-major float32, via the SPIR-V module
// in `spv`. transA: A is stored [K,M] and read transposed (opA=Aᵀ); transB: B stored [N,K].
// Returns 0 on success, nonzero on failure (see vk_bridge.c for codes).
int vk_coopmat(void);
int vk_coopmat_gemm_f16_res_tiled(const uint32_t* spv, int spvLen,
                                  void* ah, void* bh, void* ch,
                                  int M, int K, int N);
int vk_coopmat_gemm_f16_res_v3(const uint32_t* spv, int spvLen,
                               void* ah, void* bh, void* ch,
                               int M, int K, int N);
int vk_coopmat_gemm_f16_res_v2(const uint32_t* spv, int spvLen,
                               void* ah, void* bh, void* ch,
                               int M, int K, int N);
int vk_coopmat_gemm_f16_res(const uint32_t* spv, int spvLen,
                            void* ah, void* bh, void* ch,
                            int M, int K, int N);
int vk_coopmat_gemm_f16(const uint32_t* spv, int spvLen,
                        const void* Ah, const void* Bh, float* C,
                        int M, int K, int N);
int vk_matmul_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* C,
                  int M, int K, int N, int transA, int transB);

// vk_rmsnorm_f32 computes O = X/√(mean(X²)+eps)·G over the last axis (RMSNorm), with
// X/O row-major [rows,dim] float32 and G [dim]. One GPU invocation per row. Returns 0
// on success, nonzero on failure.
int vk_rmsnorm_f32(const uint32_t* spv, int spvLen,
                   const float* X, const float* G, float* O,
                   int rows, int dim, float eps);

// vk_rope_f32 applies rotary position embeddings to Q[seq,width] (width=heads·hd),
// row-major float32, using host-precomputed inverse frequencies inv[hd/2] and posDiv.
// Returns 0 on success, nonzero on failure.
int vk_rope_f32(const uint32_t* spv, int spvLen,
                const float* Q, const float* inv, float* O,
                int seq, int width, int heads, int hd, int half, int posOffset, float posDiv);

// vk_softmax_f32 computes a stable softmax over the last axis of X[rows,dim] (row-major
// float32) into O. Returns 0 on success, nonzero on failure.
int vk_softmax_f32(const uint32_t* spv, int spvLen,
                   const float* X, float* O, int rows, int dim);

// vk_crossentropy_backward_f32 computes the cross-entropy gradient DZ[b,c] from logits Z[b,c]
// and float targets T[b], with scalar upstream gv, label-smoothing eps and z-loss zl.
// Returns 0 on success, nonzero on failure.
int vk_crossentropy_backward_f32(const uint32_t* spv, int spvLen,
                                 const float* Z, const float* T, float* DZ,
                                 int b, int c, float gv, float eps, float zl);

// vk_embed_backward_f32 scatter-adds the upstream gradient G[m,d] into DTable[n,d] by the
// float indices Idx[m]: DTable[Idx[i],:] += G[i,:] (float atomics). DTable MUST be uploaded
// zero-initialized. Returns 0, -7 if the device lacks float atomics, else nonzero.
int vk_embed_backward_f32(const uint32_t* spv, int spvLen,
                          const float* Idx, const float* G, float* DTable, int n, int d, int m);

// vk_layernorm_f32 computes O = (X−mean)/√(var+eps)·G + B over the last axis, X/O
// row-major [rows,dim] float32 and G/B [dim]. Returns 0 on success, nonzero on failure.
int vk_layernorm_f32(const uint32_t* spv, int spvLen,
                     const float* X, const float* G, const float* B, float* O,
                     int rows, int dim, float eps);

// vk_rmsnorm_backward_f32 computes the RMSNorm gradient DX[rows,dim] and DGamma[dim] from
// X/Gamma/G (upstream). DGamma is accumulated with float atomics and MUST be uploaded
// zero-initialized. Returns 0 on success, -7 if the device lacks float atomics (caller
// falls back to the reference), other nonzero on failure.
int vk_rmsnorm_backward_f32(const uint32_t* spv, int spvLen,
                            const float* X, const float* Gamma, const float* G,
                            float* DX, float* DGamma, int rows, int dim, float eps);

// vk_layernorm_backward_f32 computes the LayerNorm gradient DX[rows,dim], DGamma[dim] and
// DBeta[dim] from X/Gamma/G (upstream). DGamma/DBeta are accumulated with float atomics and
// MUST be uploaded zero-initialized. Returns 0, -7 if no float atomics, else nonzero.
int vk_layernorm_backward_f32(const uint32_t* spv, int spvLen,
                              const float* X, const float* Gamma, const float* G,
                              float* DX, float* DGamma, float* DBeta, int rows, int dim, float eps);

// vk_unary_f32 applies a unary elementwise op (selector `op`) to the n float32 elements
// of X into O. Returns 0 on success, nonzero on failure.
int vk_unary_f32(const uint32_t* spv, int spvLen,
                 const float* X, float* O, int n, int op);

// vk_batch_dispatch_bench (§T380) runs the unary ReLU kernel `iters`× over one pooled buffer
// pair, either batched into one command buffer (oneBuffer=1, one submit+waitIdle, barriers
// between) or per-op (oneBuffer=0, N submit/waitIdle). Measures the Vulkan/MoltenVK batching win.
int vk_batch_dispatch_bench(const uint32_t* spv, int spvLen, int iters, int oneBuffer, int n);

// vk_crossentropy_f32 (§T403): per-row cross-entropy loss L[b]=lse−z[target] (caller means over b).
int vk_crossentropy_f32(const uint32_t* spv, int spvLen,
                        const float* Z, const float* T, float* L, int b, int c);

// vk_embed_f32 (§T403): row gather O[m,d], O[i,:]=Table[Idx[i],:].
int vk_embed_f32(const uint32_t* spv, int spvLen,
                 const float* Table, const float* Idx, float* O, int n, int m, int d);

// vk_binary_f32 applies a same-shape binary elementwise op (selector `op`) to the n
// float32 elements of A and B into O. Returns 0 on success, nonzero on failure.
int vk_binary_f32(const uint32_t* spv, int spvLen,
                  const float* A, const float* B, float* O, int n, int op);

// vk_addbias_f32 computes O[i,j] = X[rows,n][i,j] + B[n][j] (bias broadcast over rows).
// Returns 0 on success, nonzero on failure.
int vk_addbias_f32(const uint32_t* spv, int spvLen,
                   const float* X, const float* B, float* O, int rows, int n);

// vk_gelu_backward_f32 computes DX[i] = G[i]·gelu'(X[i]) over n elements.
// Returns 0 on success, nonzero on failure.
int vk_gelu_backward_f32(const uint32_t* spv, int spvLen,
                         const float* X, const float* G, float* DX, int n);

// vk_silu_backward_f32 computes DX[i] = G[i]·silu'(X[i]) over n elements.
// Returns 0 on success, nonzero on failure.
int vk_silu_backward_f32(const uint32_t* spv, int spvLen,
                         const float* X, const float* G, float* DX, int n);

// vk_addbias_backward_f32 computes DB[j] = Σ_i G[rows,n][i,j] (column sum).
// Returns 0 on success, nonzero on failure.
int vk_addbias_backward_f32(const uint32_t* spv, int spvLen,
                            const float* G, float* DB, int rows, int n);

// vk_qmatmul_q8_0 computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q8_0-quantized [N,K]
// weight (row-major, K/32 blocks of [f16 scale + 32 int8], §R94), dequantized IN-KERNEL —
// GPU quantized inference (§T137). wBytes is the (4-byte-padded) length of the W buffer, read
// as a uint word array by the shader. K must be a multiple of 32. Accumulation is f32 (the
// reference is f64, so V-CROSS uses a crossTol). Returns 0 on success, nonzero on failure.
int vk_qmatmul_q8_0(const uint32_t* spv, int spvLen,
                    const float* X, const unsigned char* W, float* O,
                    int M, int K, int N, int wBytes);

// vk_qmatmul_q4_0 is the Q4_0 analogue of vk_qmatmul_q8_0 (§T167, the Vulkan twin of Metal §T166):
// same generic {M,K,N} dispatch over three storage buffers (X f32, W bytes-as-words, O f32) — only
// the SPIR-V module differs (the Q4_0 in-kernel dequant: 18 B/block, f16 scale + 32 nibbles offset
// −8, §R94). K must be a multiple of 32. Returns 0 on success, nonzero on failure.
int vk_qmatmul_q4_0(const uint32_t* spv, int spvLen,
                    const float* X, const unsigned char* W, float* O,
                    int M, int K, int N, int wBytes);

// vk_qmatmul_q4k is the Q4_K analogue of vk_qmatmul_q8_0 (§T139): same generic {M,K,N} dispatch
// over three storage buffers (X f32, W bytes-as-words, O f32) — only the SPIR-V module differs
// (the Q4_K in-kernel dequant, §R100). K must be a multiple of 256. Returns 0 on success.
int vk_qmatmul_q4k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes);

// vk_qmatmul_q6k is the Q6_K analogue (§T141): same generic {M,K,N} byte-buffer dispatch, only
// the SPIR-V module differs (the Q6_K in-kernel dequant, §R99). K must be a multiple of 256.
int vk_qmatmul_q6k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes);

// vk_qmatmul_q5k is the Q5_K analogue (§T144): same generic {M,K,N} byte-buffer dispatch, only
// the SPIR-V module differs (the Q5_K in-kernel dequant, §R102). K must be a multiple of 256.
int vk_qmatmul_q5k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes);

// vk_qmatmul_q2k is the Q2_K analogue (§T146): same generic {M,K,N} byte-buffer dispatch, only
// the SPIR-V module differs (the Q2_K in-kernel dequant, §R104). K must be a multiple of 256.
int vk_qmatmul_q2k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes);

// vk_qmatmul_q3k is the Q3_K analogue (§T148): same generic {M,K,N} byte-buffer dispatch, only
// the SPIR-V module differs (the Q3_K in-kernel dequant, §R103). K must be a multiple of 256.
int vk_qmatmul_q3k(const uint32_t* spv, int spvLen,
                   const float* X, const unsigned char* W, float* O,
                   int M, int K, int N, int wBytes);

// vk_qweight_upload uploads a weight blob (wBytes, a multiple of 4) into a DEVICE-RESIDENT VkBuffer
// and returns an opaque handle, reused across matmuls (§T156). NULL on failure; free with
// vk_qweight_free.
void* vk_qweight_upload(const unsigned char* W, int wBytes);

// vk_qweight_free destroys a buffer returned by vk_qweight_upload.
void vk_qweight_free(void* handle);

// Device-resident buffers for the Vulkan recorder (§T381, mirrors Metal §T367). A persistent
// host-visible VkBuffer the batched-decode path chains ops over. Round-trip: upload → download.
void* vk_devbuf_upload(const void* data, int nbytes);
int vk_devbuf_download(void* handle, void* dst, int nbytes);
int vk_devbuf_upload_into(void* handle, const void* src, int nbytes);
void vk_devbuf_free(void* handle);

// Vulkan recorder (§T382, mirrors Metal §T370): one open command buffer records dispatches over
// DevBufs with barriers, submitted once at finish. begin → op(s) → finish → free.
void* vk_recorder_begin(void);
int vk_recorder_unary(void* rec, const uint32_t* spv, int spvLen, void* xh, void* oh, int n, int op);
int vk_recorder_binary(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* oh, int n, int op);
int vk_recorder_matmul(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* ch, int M, int K, int N,
                       int accumulate);
int vk_recorder_rmsnorm(void* rec, const uint32_t* spv, int spvLen, void* xh, void* gh, void* oh, int rows, int dim, float eps);
int vk_recorder_layernorm(void* rec, const uint32_t* spv, int spvLen, void* xh, void* gh, void* bh, void* oh, int rows, int dim, float eps);
int vk_recorder_addbias(void* rec, const uint32_t* spv, int spvLen, void* xh, void* bh, void* oh, int rows, int n);
int vk_recorder_rope(void* rec, const uint32_t* spv, int spvLen, void* qh, void* invh, void* oh,
                     int seq, int width, int heads, int hd, int half, int posOffset, float posDiv,
                     int elemOff);
int vk_recorder_rope2(void* rec, const uint32_t* spv, int spvLen, void* qh, void* invh,
                      int seq, int stride, int headsQ, int offQ, int headsK, int offK,
                      int hd, int half, int posOffset, float posDiv);
int vk_recorder_matmul_strided(void* rec, const uint32_t* spv, int spvLen, void* ah, void* bh, void* ch,
                               int M, int K, int N, int transA, int transB,
                               int lda, int ldb, int ldc, int offA, int offB, int offC, float alpha);
int vk_recorder_softmax_causal(void* rec, const uint32_t* spv, int spvLen, void* sh, int rows, int cols, int causal);
int vk_recorder_mha(void* rec, const uint32_t* spv, int spvLen, void* qh, void* kh, void* vh, void* oh,
                    int sq, int sk, int dm, int heads, int kvHeads, int dk, int causal, int window, float scale,
                    int qElemOff);
int vk_recorder_qmatmul(void* rec, const uint32_t* spv, int spvLen, void* xh, void* wHandle, void* oh,
                        int M, int K, int N, int wBytes);
int vk_recorder_mha_decode(void* rec, const uint32_t* spv, int spvLen, void* qh, void* kh, void* vh, void* oh,
                           int sq, int sk, int dm, int heads, int kvHeads, int dk, int causal, float scale,
                           int qElemOff);
int vk_recorder_blit(void* rec, void* srcH, int srcOff, void* dstH, int dstOff, int nbytes);
int vk_recorder_copy2d(void* rec, void* srcH, int srcOff, int srcStride,
                       void* dstH, int dstOff, int dstStride, int rows, int rowFloats);
int vk_recorder_finish(void* rec);
void vk_recorder_free(void* rec);

// vk_qmatmul_resident is a quantized matmul whose weight is the already-resident buffer `wHandle`
// (from vk_qweight_upload); `spv` selects the per-quant-type shader. Only X is uploaded per call.
int vk_qmatmul_resident(const uint32_t* spv, int spvLen,
                        const float* X, void* wHandle, float* O,
                        int M, int K, int N, int wBytes);

// vk_mha_f32 runs fused multi-head attention (forward) with the SPIR-V module in
// `spv`: Q[sq,dm], K[sk,dkv], V[sk,dkv] → O[sq,dm], dkv = kvHeads*dk, dk = dm/heads.
// Per query head h (sharing KV head h/(heads/kvHeads) for GQA): scores =
// scale·Qₕ·Kₕᵀ, causal −∞ mask when causal!=0, softmax, ·Vₕ. A positive window
// restricts each query to the window most recent keys (sliding-window attention);
// 0 = full attention. dk ≤ 128. Returns 0 on success, nonzero on failure.
int vk_mha_f32(const uint32_t* spv, int spvLen,
               const float* Q, const float* K, const float* V, float* O,
               int sq, int sk, int dm, int heads, int kvHeads, int dk,
               int causal, int window, float scale);

// vk_flashattn_f32 runs FlashAttention-2 forward (Dao 2023) with the SPIR-V module in
// `spv`: Q[seq,dm], K,V[seq,kvHeads*dk] (sq==sk; query head h shares K/V head
// h/(heads/kvHeads) for GQA/MQA) → O[seq,dm], dk = dm/heads. One invocation per (head,
// query row) streams the keys with the online-softmax recurrence, so no [seq,seq] score
// row is materialized. causal!=0 masks keys past the query row. dk ≤ 128. Same result
// as vk_mha_f32 up to float reassociation. Returns 0 on success.
int vk_flashattn_f32(const uint32_t* spv, int spvLen,
                     const float* Q, const float* K, const float* V, float* O,
                     int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);

// vk_retention_f32 computes the RetNet retention forward O[L,dv]=(QKᵀ⊙D)V, D_nm=γ^(n−m), for a single
// head. Value expansion dv≠dk supported: Q,K are [L,dk], V,O are [L,dv] (§T179). One invocation per
// query row streams keys with an incremental γ-decay and a register acc[dv] (dv ≤ 128) — no [L,L]
// score matrix. Same result as the reference up to float reassociation (V-CROSS crossTol). Returns 0.
int vk_retention_f32(const uint32_t* spv, int spvLen,
                     const float* Q, const float* K, const float* V, float* O,
                     int L, int dk, int dv, float gamma);

// vk_retention_backward_f32 is the retention backward (§T175/§T192): (Q,K [L,dk], V,dO [L,dv]) →
// (dQ,dK [L,dk], dV [L,dv]). Value expansion dv≠dk supported. One invocation per query row writes its
// own dQ row and float-atomically accumulates the shared dK/dV (uploaded pre-zeroed; needs
// VK_EXT_shader_atomic_float). dk ≤ 128. Returns 0 on success, -7 if the device lacks atomic-float
// (caller falls back to the reference).
int vk_retention_backward_f32(const uint32_t* spv, int spvLen,
                              const float* Q, const float* K, const float* V, const float* dO,
                              float* dQ, float* dK, float* dV, int L, int dk, int dv, float gamma);

// vk_atomic_float returns 1 if the device enabled VK_EXT_shader_atomic_float,
// which vk_mha_backward_f32 requires (shared dK/dV are accumulated atomically).
int vk_atomic_float(void);

// vk_mha_decode_f32 (§T432): cooperative decode/prefill attention over host slices (per-op path).
int vk_mha_decode_f32(const uint32_t* spv, int spvLen,
                      const float* Q, const float* K, const float* V, float* O,
                      int sq, int sk, int dm, int heads, int kvHeads, int dk,
                      int causal, float scale);

// vk_mha_backward_f32 is the SDPA backward: (Q,K,V,dO) → (dQ,dK,dV). dQ/dK/dV are
// passed in pre-zeroed (dK/dV accumulate atomically). Returns 0 on success, -7 if
// float atomics are unavailable (caller falls back to the reference), else nonzero.
int vk_mha_backward_f32(const uint32_t* spv, int spvLen,
                        const float* Q, const float* K, const float* V, const float* dO,
                        float* dQ, float* dK, float* dV,
                        int sq, int sk, int dm, int heads, int kvHeads, int dk,
                        int causal, int window, float scale);

// vk_conv2d_f32 is 2-D cross-correlation (forward conv) lowered to im2col + the
// tiled GEMM shader + a scatter/bias pass (§T342): X[N,C,H,Wd] ⋆ W[F,C,KH,KW]
// (+ Bias[F], zeros when none) → Out[N,F,ho,wo] with the given stride and
// zero-padding. Takes the im2col, matmul, and colout SPIR-V modules; the
// intermediate column/GEMM buffers stay device-resident across the three stages.
// Returns 0 on success, nonzero on failure.
int vk_conv2d_f32(const uint32_t* spvIm2col, int im2colLen,
                  const uint32_t* spvMatmul, int matmulLen,
                  const uint32_t* spvColout, int coloutLen,
                  const float* X, const float* W, const float* B, float* Out,
                  int N, int C, int H, int Wd, int F, int KH, int KW,
                  int stride, int pad, int ho, int wo);

// vk_conv2d_igemm_f32 is the FUSED implicit-GEMM forward conv (§T620): a single
// dispatch of the conv_igemm.comp kernel that gathers X directly in the GEMM tile
// load (no im2col materialization) and scatters+biases straight to NCHW. Takes only
// the fused SPIR-V module; no col/G intermediate buffers. Same shape contract and
// result as vk_conv2d_f32. Returns 0 on success, nonzero on failure.
int vk_conv2d_igemm_f32(const uint32_t* spv, int spvLen,
                        const float* X, const float* W, const float* B, float* Out,
                        int N, int C, int H, int Wd, int F, int KH, int KW,
                        int stride, int pad, int ho, int wo);

// vk_conv2d_backward_f32 is the conv2d backward: (X,W,dO) → (dX,dW,dBias). dX/dW/dBias
// are passed in pre-zeroed and accumulate with float atomics. Returns 0 on success,
// -7 if float atomics are unavailable (caller falls back to the reference), else
// nonzero.
int vk_conv2d_backward_f32(const uint32_t* spv, int spvLen,
                           const float* X, const float* W, const float* dO,
                           float* dX, float* dW, float* dBias,
                           int N, int C, int H, int Wd, int F, int KH, int KW,
                           int stride, int pad, int ho, int wo);

#endif

// vk_mha_backward_chain (§T528): matmul-decomposed SDPA backward (the vulkan analog of
// metal's MPS path §T399) — seven staged submits over packed per-head buffers using the
// strided-matmul, packed-softmax and softmax-jacobian pipelines. dVt/dKt come back PER
// QUERY HEAD [heads,seq,dk]; the caller reduces GQA groups. Returns 0 on success.
int vk_mha_backward_chain(const uint32_t* spvMatmul, int matmulLen,
                          const uint32_t* spvSoftmax, int softmaxLen,
                          const uint32_t* spvJac, int jacLen,
                          const float* Q, const float* K, const float* V, const float* dO,
                          float* dQ, float* dVt, float* dKt,
                          int seq, int dm, int heads, int kvHeads, int dk,
                          int causal, float scale);

// vk_mha_forward_chain (§T531): matmul-decomposed SDPA forward in the §T528 chain
// structure (three staged submits, one descriptor set each). Returns 0 on success.
int vk_mha_forward_chain(const uint32_t* spvMatmul, int matmulLen,
                         const uint32_t* spvSoftmax, int softmaxLen,
                         const float* Q, const float* K, const float* V, float* O,
                         int seq, int dm, int heads, int kvHeads, int dk,
                         int causal, float scale);
