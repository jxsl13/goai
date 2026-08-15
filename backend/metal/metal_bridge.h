// Metal/MPS bridge for the optional metal backend (§T20). Compiled only under
// `-tags metal` on darwin with cgo. Synchronous API: each call commits and
// waits (async batching is a later optimization; §V14 keeps the interface safe).
#ifndef GOAI_METAL_BRIDGE_H
#define GOAI_METAL_BRIDGE_H

// mtl_available returns 1 if a Metal device with MPS support exists.
int mtl_available(void);

// mtl_available_memory returns the device's remaining recommended working-set
// budget in bytes (recommendedMaxWorkingSetSize-currentAllocatedSize), or 0
// when Metal cannot report one.
unsigned long long mtl_available_memory(void);

// mtl_matmul_f32 computes C[M,N] = A[M,K]·B[K,N], all row-major float32.
// Returns 0 on success, nonzero on failure.
int mtl_matmul_f32(const float* A, const float* B, float* C, int M, int K, int N,
                   int transA, int transB);

// mtl_rmsnorm_f32 computes O = X/√(mean(X²)+eps)·G over the last axis (RMSNorm), with
// X/O row-major [rows,dim] float32 and G [dim]. One GPU thread per row. Returns 0 on
// success, nonzero on failure.
int mtl_rmsnorm_f32(const float* X, const float* G, float* O, int rows, int dim, float eps);

// mtl_rope_f32 applies rotary position embeddings to Q[seq,width] (width=heads·hd),
// row-major float32, using host-precomputed inverse frequencies inv[hd/2] and posDiv.
// Returns 0 on success, nonzero on failure.
int mtl_rope_f32(const float* Q, const float* inv, float* O,
                 int seq, int width, int heads, int hd, int half, int posOffset, float posDiv);

// mtl_rope_backward_f32 applies the RoPE BACKWARD (inverse rotation) to the upstream
// gradient G[seq,width] into O, same layout/params as mtl_rope_f32. Returns 0 / nonzero.
int mtl_rope_backward_f32(const float* G, const float* inv, float* O,
                          int seq, int width, int heads, int hd, int half, int posOffset, float posDiv);

// mtl_softmax_f32 computes a stable softmax over the last axis of X[rows,dim] (row-major
// float32) into O. Returns 0 on success, nonzero on failure.
int mtl_softmax_f32(const float* X, float* O, int rows, int dim);

// mtl_crossentropy_f32 (§T401): per-row cross-entropy loss L[b] = lse − z[target] (caller means
// over b). Basic case only (no label-smoothing/z-loss). Fixes a 20ms silent CPU fallback.
int mtl_crossentropy_f32(const float* Z, const float* T, float* L, int b, int c);

// mtl_embed_f32 (§T402): row gather O[m,d], O[i,:]=Table[Idx[i],:]. The last metal training fallback.
int mtl_embed_f32(const float* Table, const float* Idx, float* O, int n, int m, int d);

// mtl_crossentropy_backward_f32 computes the cross-entropy gradient DZ[b,c] from logits Z[b,c]
// and float targets T[b], with scalar upstream gv, label-smoothing eps and z-loss zl.
// Returns 0 on success, nonzero on failure.
int mtl_crossentropy_backward_f32(const float* Z, const float* T, float* DZ,
                                  int b, int c, float gv, float eps, float zl);

// mtl_embed_backward_f32 scatter-adds G[m,d] into DTable[n,d] by float indices Idx[m]:
// DTable[Idx[i],:] += G[i,:] (float atomics, DTable uploaded zero). Returns 0 / nonzero.
int mtl_embed_backward_f32(const float* Idx, const float* G, float* DTable, int n, int d, int m);

// mtl_layernorm_f32 computes O = (X−mean)/√(var+eps)·G + B over the last axis, X/O
// row-major [rows,dim] float32 and G/B [dim]. Returns 0 on success, nonzero on failure.
int mtl_layernorm_f32(const float* X, const float* G, const float* B, float* O,
                      int rows, int dim, float eps);

// mtl_rmsnorm_backward_f32 computes the RMSNorm gradient DX[rows,dim] and DGamma[dim] from
// X/Gamma/G (upstream). DGamma is accumulated with float atomics (uploaded zero). Returns
// 0 on success, nonzero on failure.
int mtl_rmsnorm_backward_f32(const float* X, const float* Gamma, const float* G,
                             float* DX, float* DGamma, int rows, int dim, float eps);

// mtl_layernorm_backward_f32 computes the LayerNorm gradient DX[rows,dim], DGamma[dim] and
// DBeta[dim] from X/Gamma/G (upstream). DGamma/DBeta are float-atomic accumulators (uploaded
// zero). Returns 0 on success, nonzero on failure.
int mtl_layernorm_backward_f32(const float* X, const float* Gamma, const float* G,
                               float* DX, float* DGamma, float* DBeta, int rows, int dim, float eps);

// mtl_unary_f32 applies a unary elementwise op (selector `op`) to the n float32 elements
// of X into O. Returns 0 on success, nonzero on failure.
int mtl_unary_f32(const float* X, float* O, int n, int op);

// mtl_binary_f32 applies a same-shape binary elementwise op (selector `op`) to the n
// float32 elements of A and B into O. Returns 0 on success, nonzero on failure.
int mtl_binary_f32(const float* A, const float* B, float* O, int n, int op);

// mtl_addbias_f32 computes O[i,j] = X[rows,n][i,j] + B[n][j] (bias broadcast over rows).
// Returns 0 on success, nonzero on failure.
int mtl_addbias_f32(const float* X, const float* B, float* O, int rows, int n);

// mtl_gelu_backward_f32 computes DX[i] = G[i]·gelu'(X[i]) over n float32 elements.
// Returns 0 on success, nonzero on failure.
int mtl_gelu_backward_f32(const float* X, const float* G, float* DX, int n);

// mtl_silu_backward_f32 computes DX[i] = G[i]·silu'(X[i]) over n float32 elements.
// Returns 0 on success, nonzero on failure.
int mtl_silu_backward_f32(const float* X, const float* G, float* DX, int n);

// mtl_addbias_backward_f32 computes DB[j] = Σ_i G[rows,n][i,j] (column sum).
// Returns 0 on success, nonzero on failure.
int mtl_addbias_backward_f32(const float* G, float* DB, int rows, int n);

// mtl_qmatmul_q8_0 computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q8_0-quantized
// [N,K] weight (row-major, K/32 blocks per row of [f16 scale + 32 int8], §R94) — a
// quantized linear layer run on the GPU with the weights dequantized IN-KERNEL (no full-
// precision weight matrix materialized). One thread per output (mi,ni) streams the row's
// blocks; accumulation is f32 (vs the reference's f64, so V-CROSS uses a crossTol). K must
// be a multiple of 32. Returns 0 on success, nonzero on failure.
int mtl_qmatmul_q8_0(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// mtl_retention_f32 computes the RetNet retention forward O[L,dv] = (QKᵀ ⊙ D)·V for a single head,
// D_nm=γ^(n−m) (n≥m else 0), Q,K [L,dk], V,O [L,dv] (value expansion dv≠dk supported, §T178). One
// thread per query row streams keys with an incremental γ-decay and a register acc[dv] (dv must be
// ≤128) — no [L,L] score matrix. f32; 0 on success.
int mtl_retention_f32(const float* Q, const float* K, const float* V, float* O, int L, int dk, int dv, float gamma);

// mtl_retention_backward_f32 is the retention backward (§T174/§T191): (Q,K [L,dk], V,dO [L,dv]) →
// (dQ,dK [L,dk], dV [L,dv]). Value expansion dv≠dk supported. One thread per query row writes its own
// dQ row and atomically accumulates into the shared dK/dV (pre-zeroed). dk ≤ 128. 0 on success.
int mtl_retention_backward_f32(const float* Q, const float* K, const float* V, const float* dO,
                               float* dQ, float* dK, float* dV, int L, int dk, int dv, float gamma);

// mtl_qmatmul_q4_0 computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q4_0-quantized [N,K]
// weight (row-major, K/32 blocks per row of [f16 scale + 16 bytes of 32 4-bit nibbles], 18 B/
// block, §R94): low nibble → element i, high nibble → element i+16, value = d·(nibble−8). One
// thread per output (mi,ni) streams the row's blocks, dequantizing IN-KERNEL; f32 accumulation
// (V-CROSS crossTol vs the f64 reference). K must be a multiple of 32. 0 on success.
int mtl_qmatmul_q4_0(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// mtl_qmatmul_q4k computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q4_K-quantized [N,K]
// weight (row-major, K/256 super-blocks per row of 144 bytes, §R100) — the DOMINANT real-world
// quant (the bulk of Q4_K_M models), dequantized IN-KERNEL: asymmetric affine
// y = d·sc6·nibble − dmin·min6 with the 6-bit scale/min unpacked per get_scale_min_k4. One
// thread per output. K must be a multiple of 256. Returns 0 on success, nonzero on failure.
int mtl_qmatmul_q4k(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// Selects the simdgroup-cooperative Q4_K kernel (on, default) or the historical
// one-thread-per-output kernel (off) for same-process forced-off A/B measurement.
// Returns the previous setting. Unsupported devices transparently retain scalar.
int mtl_q2k_cooperative_set(int on);
int mtl_q8_0_cooperative_set(int on);
int mtl_q3k_cooperative_set(int on);
int mtl_q4k_cooperative_set(int on);
int mtl_q5k_cooperative_set(int on);

// mtl_qmatmul_q6k computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q6_K-quantized [N,K]
// weight (row-major, K/256 super-blocks per row of 210 bytes, §R99) — the higher-precision
// tensors of Q4_K_M models (attn_v/ffn_down/output), dequantized IN-KERNEL: SYMMETRIC
// y = d·scale·(q6−32) with a signed int8 sub-block scale and the 6-bit quant assembled from
// ql (low/high nibble) + qh (2 high bits). One thread per output. K must be a multiple of 256.
// Returns 0 on success, nonzero on failure.
int mtl_qmatmul_q6k(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// Selects the SIMD-group-cooperative Q6_K M=1 kernel (on, default) or the
// historical scalar-K kernel (off) for same-process forced-off measurement.
// Returns the previous setting. Unsupported devices transparently retain scalar.
int mtl_q6k_cooperative_set(int on);

// mtl_qmatmul_q5k computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q5_K-quantized [N,K]
// weight (row-major, K/256 super-blocks per row of 176 bytes, §R102) — the Q5_K_M weight
// format, dequantized IN-KERNEL: Q4_K's asymmetric affine y = d·sc6·q5 − dmin·min6 plus a 5th
// bit per quant from the qh plane (q5 = nibble | highbit<<4). One thread per output. K must be a
// multiple of 256. Returns 0 on success, nonzero on failure.
int mtl_qmatmul_q5k(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// mtl_qmatmul_q2k computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q2_K-quantized [N,K]
// weight (row-major, K/256 super-blocks per row of 84 bytes, §R104) — the smallest quant (Q2_K,
// for very large models), dequantized IN-KERNEL: asymmetric affine y = d·(sc&0xF)·q2 −
// dmin·(sc>>4) with q2 ∈ [0,3] 2-bit and plain 4-bit nibble scale/min (no get_scale_min_k4, no
// hmask). One thread per output. K must be a multiple of 256. Returns 0 on success, nonzero on
// failure.
int mtl_qmatmul_q2k(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// mtl_qmatmul_q3k computes O[M,N] = X[M,K] · dequant(W)ᵀ where W is a Q3_K-quantized [N,K]
// weight (row-major, K/256 super-blocks per row of 110 bytes, d LAST, §R103) — dequantized
// IN-KERNEL: SYMMETRIC y = d·(sc6−32)·(q3−4) with a signed 6-bit sub-block scale (from the
// aux/kmask splice) and the 3-bit quant assembled from qs (2 low bits) + hmask (1 high bit,
// INVERTED). One thread per output. K must be a multiple of 256. Returns 0 on success.
int mtl_qmatmul_q3k(const float* X, const unsigned char* W, float* O, int M, int K, int N);

// mtl_qweight_upload copies a quantized weight blob into a DEVICE-RESIDENT MTLBuffer (retained
// beyond the call) and returns an opaque handle, so the weight is uploaded ONCE and reused across
// many matmuls — the decode loop, which reuses every weight for every token, is where this pays
// off (§T153). Returns NULL on failure. Free with mtl_qweight_free.
void* mtl_qweight_upload(const unsigned char* W, int len);

// mtl_qweight_free releases a buffer returned by mtl_qweight_upload.
void mtl_qweight_free(void* handle);

// Device-resident tensor primitive (ADR-0019 Phase 1, §T367): upload/download/free a
// general host-visible device buffer. mtl_devbuf_upload retains a Shared MTLBuffer of
// nbytes and returns its handle (NULL on failure); mtl_devbuf_download copies nbytes
// back to dst (0 ok, <0 error); mtl_devbuf_free releases the handle.
int mtl_batch_dispatch_bench(int iters, int oneBuffer);
int mtl_chain_matmul_relu(const float* A, const float* B, float* Out, int M, int K, int N);
void* mtl_recorder_begin(void);
int mtl_recorder_unary(void* rec, void* xh, void* oh, int n, int op);
int mtl_recorder_binary(void* rec, void* ah, void* bh, void* oh, int n, int op);
int mtl_recorder_blit(void* rec, void* srcH, int srcOff, void* dstH, int dstOff, int nbytes);
int mtl_recorder_copy2d(void* rec, void* srcH, int srcOff, int srcStride,
                        void* dstH, int dstOff, int dstStride, int rows, int rowFloats);
int mtl_recorder_matmul(void* rec, void* ah, void* bh, void* ch, int M, int K, int N,
                        int accumulate);
int mtl_recorder_rmsnorm(void* rec, void* xh, void* gh, void* oh, int rows, int dim, float eps);
int mtl_recorder_layernorm(void* rec, void* xh, void* gh, void* bh, void* oh, int rows, int dim, float eps);
int mtl_recorder_addbias(void* rec, void* xh, void* bh, void* oh, int rows, int n);
int mtl_recorder_rope(void* rec, void* qh, void* invh, void* oh,
                      int seq, int width, int heads, int hd, int half, int posOffset, float posDiv,
                      int elemOff);
int mtl_recorder_rope2(void* rec, void* qh, void* invh,
                       int seq, int stride, int headsQ, int offQ, int headsK, int offK,
                       int hd, int half, int posOffset, float posDiv);
int mtl_recorder_flashattn(void* rec, void* qh, void* kh, void* vh, void* oh,
                           int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);
int mtl_recorder_mha(void* rec, void* qh, void* kh, void* vh, void* oh,
                     int sq, int sk, int dm, int heads, int kvHeads, int dk,
                     int causal, int window, float scale, int qElemOff);
int mtl_recorder_finish(void* rec);
int mtl_recorder_commit(void* rec);
int mtl_recorder_wait(void* rec);
void mtl_recorder_free(void* rec);
void* mtl_devbuf_upload(const void* data, int nbytes);
int mtl_devbuf_download(void* handle, void* dst, int nbytes);
int mtl_devbuf_upload_into(void* handle, const void* src, int nbytes);
void mtl_devbuf_free(void* handle);

// mtl_qmatmul_resident is a quantized matmul whose weight is the already-resident buffer `wbuf`
// (from mtl_qweight_upload) instead of per-call bytes — only X is uploaded each call. qtype is the
// ggml code (8=Q8_0, 10=Q2_K, 11=Q3_K, 12=Q4_K, 13=Q5_K, 14=Q6_K); it selects the matching cached
// pipeline (§T155). Returns 0 on success, -7 for an unknown qtype, nonzero otherwise.
int mtl_qmatmul_resident(const float* X, void* wbuf, float* O, int M, int K, int N, int qtype);

// mtl_recorder_qmatmul (§T413): record-mode quantized matmul over a DeviceBuffer X/O and a resident
// quantized weight — the quantized batched-decode building block. Same qtype codes as above.
int mtl_recorder_qmatmul(void* rec, void* xh, void* wbuf, void* oh, int M, int K, int N, int qtype);

// mtl_mha_f32 computes fused multi-head scaled-dot-product attention (forward):
// Q[sq,dm], K[sk,dkv], V[sk,dkv] → O[sq,dm], dkv = kvHeads*dk, dk = dm/heads.
// Per query head h (sharing KV head h/(heads/kvHeads) for GQA): scores =
// scale·Qₕ·Kₕᵀ, causal −∞ mask for key > query abs-pos when causal!=0, softmax,
// ·Vₕ. A positive window restricts each query to the window most recent keys
// (sliding-window attention); 0 = full attention. Requires dk ≤ 128. Returns 0 on
// success, nonzero on failure.
int mtl_mha_f32(const float* Q, const float* K, const float* V, float* O,
                int sq, int sk, int dm, int heads, int kvHeads, int dk,
                int causal, int window, float scale);

// mtl_flashattn_f32 is FlashAttention-2 forward (Dao 2023): exact multi-head
// scaled-dot-product attention over Q[seq,dm], K,V[seq,kvHeads*dk] (sq==sk; query head
// h shares K/V head h/(heads/kvHeads) for GQA/MQA), computed with the online-softmax
// tiling so no [seq,seq] score row is ever materialized — one thread per (head, query
// row) keeps only a running max m, normalizer ℓ and an O(dk) output accumulator,
// streaming the keys once. causal!=0 masks keys past the query row. Requires dk ≤ 128.
// Same result as mtl_mha_f32 up to float reassociation. Returns 0 on success, nonzero
// on failure.
int mtl_flashattn_f32(const float* Q, const float* K, const float* V, float* O,
                      int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);

// mtl_mha_mps (§T394): attention forward via MPS(Q·Kᵀ)→causal softmax→MPS(P·V) — ~15× faster than
// the hand-written flash kernel (§T393). Q,O are [seq,dm]; K,V are [seq,kvHeads·dk].
int mtl_mha_mps(const float* Q, const float* K, const float* V, float* O,
                int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);

// mtl_mha_mpsgraph (§T621, EXPERIMENT): attention forward via a single cached MPSGraph
// (reshape→transpose→matmul QKᵀ→scale→+causal-mask→softmax→matmul PV→transpose back). All heads
// batched in ONE graph run (vs mtl_mha_mps's per-head serialized MPS loop). Q,O are [seq,dm];
// K,V are [seq,kvHeads·dk]. GQA via explicit KV broadcast. A/B toggle only — see SetMHAUseMPSGraph.
int mtl_mha_mpsgraph(const float* Q, const float* K, const float* V, float* O,
                     int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);

// mtl_attn_cache_cap_set (§T622) sets the effective MPSGraph attention cache cap (clamped to
// [1,16]), clears all resident graphs, and returns the previous cap. cap==1 reproduces the old
// single last-shape cache (per-length recompile) for the variable-length prefill A/B. Test-only.
int mtl_attn_cache_cap_set(int cap);

// mtl_mha_decode_host (§T432): cooperative decode/prefill attention over host slices — the per-op
// path's route onto the §T428/431 kernel (the two-pass kernel is ~serial in sk at small sq).
int mtl_mha_decode_host(const float* Q, const float* K, const float* V, float* O,
                        int sq, int sk, int dm, int heads, int kvHeads, int dk,
                        int causal, float scale);

// mtl_mha_backward_mps (§T399/T400): attention backward via MPS matmuls (dV=Pᵀ·dO, dP=dO·Vᵀ,
// dQ=scale·dS·K, dK=scale·dSᵀ·Q) — the backward analog of mtl_mha_mps. Supports GQA (kvHeads<heads,
// dK/dV accumulate via MPS beta=1). window==0. Q/dQ/dO [seq,dm]; K/V/dK/dV [seq,kvHeads·dk].
int mtl_mha_backward_mps(const float* Q, const float* K, const float* V, const float* dO,
                         float* dQ, float* dK, float* dV,
                         int seq, int dm, int heads, int dk, int causal, int kvHeads, float scale);

// mtl_mha_backward_f32 is the SDPA backward: (Q,K,V,dO)[sq,·] → (dQ,dK,dV). One
// thread per (head, query row): dQ is written exclusively (own head slice + row),
// while dK/dV are accumulated with atomic float adds because query heads in a GQA
// group and all query rows share them. Requires dk ≤ 128 and float-atomic support;
// returns 0 on success, -5 if the GPU lacks the atomic pipeline (caller falls back
// to the reference), nonzero otherwise.
int mtl_mha_backward_f32(const float* Q, const float* K, const float* V, const float* dO,
                         float* dQ, float* dK, float* dV,
                         int sq, int sk, int dm, int heads, int kvHeads, int dk,
                         int causal, int window, float scale);

// mtl_conv2d_f32 is 2-D cross-correlation (forward conv): X[N,C,H,W] ⋆ W[F,C,KH,KW]
// (+ per-filter Bias[F]) → Out[N,F,ho,wo] with the given stride and zero-padding.
// One thread per output element (no accumulation races). B is a length-F buffer of
// zeros when the layer has no bias. Returns 0 on success, nonzero on failure.
int mtl_conv2d_f32(const float* X, const float* W, const float* B, float* Out,
                   int N, int C, int H, int Wd, int F, int KH, int KW,
                   int stride, int pad, int ho, int wo);

// mtl_conv2d_mps_f32 is the same 2-D cross-correlation forward via Apple's native
// MPSGraph convolution2D primitive (implicit-GEMM/Winograd internally) instead of
// the im2col+MPS-GEMM lowering. Same arguments and semantics as mtl_conv2d_f32.
// Returns 0 on success, nonzero on failure.
int mtl_conv2d_mps_f32(const float* X, const float* W, const float* B, float* Out,
                       int N, int C, int H, int Wd, int F, int KH, int KW,
                       int stride, int pad, int ho, int wo);

// mtl_conv_cache_cap_set (§T622) sets the effective MPSGraph conv cache cap (clamped to [1,16]),
// clears all resident graphs, and returns the previous cap. cap==1 reproduces the old single
// last-shape cache (per-call recompile for a multi-shape CNN) for the A/B benchmark. Test-only.
int mtl_conv_cache_cap_set(int cap);

// mtl_matmul_nocopy_set selects zero-copy operand wrapping (1, default) or the
// pooled-copy path (0) for mtl_matmul_f32; returns the previous value. A/B hook.
int mtl_matmul_nocopy_set(int on);

// mtl_conv2d_backward_f32 is the conv2d backward: (X,W,dO) → (dX,dW,dBias). One
// thread per output-gradient element (n,f,oy,ox) scatters into the shared dX/dW/dBias
// with atomic float adds. dX/dW/dBias are passed in pre-zeroed. Returns 0 on success,
// nonzero on failure.
int mtl_conv2d_backward_f32(const float* X, const float* W, const float* dO,
                            float* dX, float* dW, float* dBias,
                            int N, int C, int H, int Wd, int F, int KH, int KW,
                            int stride, int pad, int ho, int wo);

#endif
