// Package llamagpu decodes nlp.Llama models on the GPU with batched command buffers (ADR-0019):
// each per-token step records the whole layer stack into ONE command buffer over device-resident
// weights and KV cache, instead of paying a dispatch round-trip per op. Measured on a real model
// (D=512, GQA 8:2, 6 layers): 24× faster than nlp.Llama.DecodeStep on Metal and 21× on Vulkan,
// with token-for-token identical greedy output (§T404/§T409).
//
// Architecture coverage (CUDA): the same batched Decoder core drives 20 model families beyond Llama,
// each a New*CUDA constructor + greedy parity test, built from one composable set of gated
// generalizations (norm type/placement — pre / one-norm-parallel / two-norm-parallel / post-norm /
// sandwich; RMSNorm / LayerNorm±bias / (1+w); MLP — SwiGLU / GELU / squared-ReLU / GeGLU; position —
// full RoPE / partial RoPE / ALiBi; attention width — MHA / GQA / MQA / decoupled-head_dim; per-head
// & full-width QK-norm; attention soft-cap; config scalars; logit scale; √dim embed; biased
// projections/lm_head). Dense: StableLM, StarCoder2, Phi, GPT-NeoX, Qwen2, Qwen3, Phi-3, Granite,
// Cohere, Nemotron, Gemma, OLMo2, Falcon, Gemma2, MPT. Sparse Mixture-of-Experts (route → experts →
// combine, all in the pre-recorded command buffer via on-device routing): Mixtral, Qwen3-MoE, OLMoE,
// GraniteMoE, Qwen2-MoE (shared expert). These New*CUDA constructors are cuda-only where partial
// rotary / soft-cap / ALiBi / MoE kernels are involved; metal/vulkan return the corresponding stubs.
//
// Build [New] (Metal, darwin+cgo) or [NewVulkan] (vulkan build tag) from any *nlp.Llama — including
// one loaded via nlp.LlamaFromGGUF — then call [Decoder.Generate] with any nlp.TokenSampler
// (nlp.Sampler or nlp.Mirostat), or drive
// [Decoder.Step] / [Decoder.StepN] yourself. Beyond plain generation the package provides:
//
//   - [Decoder.StepN]: a whole multi-token window in one recorded step. Generate uses it to prefill
//     the prompt in a single dispatch round-trip — measured 41× faster than token-at-a-time (§T418).
//   - [NewQuant] / [NewQuantVulkan]: decode a quantized model (nlp.QuantizeLlama or a quantized
//     GGUF) with every projection held device-resident in its 4-8× smaller ggml form (§T415).
//   - [SpeculativeGenerate]: lossless speculative decoding — a small draft Decoder proposes, the
//     target verifies the window in one StepN, and the output stays exactly target-distributed
//     (§T419; ~2-3× at typical acceptance rates with a trained draft).
//   - [PromptLookupGenerate]: draft-model-free speculative decoding — candidate continuations are
//     copied from the sequence's own history by n-gram matching (§T426); lossless, and effective
//     when the output repeats the input (summarization, RAG, code editing).
//   - [MedusaGenerate]: Medusa chain drafting (§T446/§T447) — nlp.MedusaHeads trained on the base's
//     own rollouts draft host-side for free, one batched StepN verifies the window. Measured 1.81×
//     at 97% acceptance where draft-model speculative managed 1.12× (drafting cost is the
//     difference on dispatch-bound decoders); greedy-anchored, not distribution-exact.
//
// The Decoder core is backend-agnostic over small buffer/recorder interfaces; the two adapters plug
// in the concrete backends, whose exported Recorder/DeviceBuffer APIs are identical by construction
// (§T391/§T408).
package llamagpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// unarySiLU/binaryAdd/binaryMul are the shared kernel selectors (identical on both backends —
// they must match shaders/unary.comp / metal_bridge.m's unary switch and the binary op tables).
const (
	unarySiLU     = 6
	unaryGELU     = 9
	unaryReLU2    = 10 // squared ReLU (Nemotron); cuda-only
	unarySigmoid  = 11 // plain sigmoid (Qwen2-MoE shared-expert gate); cuda-only
	unarySoftplus = 12 // softplus (Mamba Δ); cuda-only
	binaryAdd     = 0
	binaryMul     = 2
	binarySwiGLU  = 6 // fused silu(a)·b — one dispatch instead of SiLU+Mul (§T613)
)

// buffer is a device-resident f32 buffer (metal.DeviceBuffer / vulkan.DeviceBuffer).
type buffer interface {
	UploadF32([]float32) error
	DownloadF32([]float32) error
	Release()
}

// qweight is a backend-resident quantized weight (metal/vulkan *ResidentQWeight) — opaque to the
// core except for Close.
type qweight interface {
	Close() error
}

// linear records o[m,N] = x[m,K]·W for one projection — either an f32 device weight (MatMul) or a
// resident quantized weight (QMatMulResident). This is what lets ONE Decoder core serve both the
// f32 and the quantized model (§T415).
type linear interface {
	record(r recorder, x, o buffer, m int) error
	// recordAdd records dst += x·W (the residual-add epilogue, §T613). The f32 weight fuses
	// the add into the matmul (one dispatch); the quantized weight has no accumulate path,
	// so it multiplies into scratch and adds — the pre-fusion two-dispatch shape.
	recordAdd(r recorder, x, scratch, dst buffer, m int) error
}

type f32Linear struct {
	w    buffer
	k, n int
}

func (l f32Linear) record(r recorder, x, o buffer, m int) error {
	return r.MatMul(x, l.w, o, m, l.k, l.n)
}

func (l f32Linear) recordAdd(r recorder, x, _, dst buffer, m int) error {
	return r.MatMulAcc(x, l.w, dst, m, l.k, l.n)
}

type quantLinear struct{ w qweight }

func (l quantLinear) record(r recorder, x, o buffer, m int) error {
	return r.QMatMulResident(x, l.w, o, m)
}

func (l quantLinear) recordAdd(r recorder, x, scratch, dst buffer, m int) error {
	return firstErr(
		r.QMatMulResident(x, l.w, scratch, m),
		r.Binary(dst, scratch, dst, binaryAdd),
	)
}

// recorder is one open batched command buffer (metal.Recorder / vulkan.Recorder); the adapter
// asserts the buffer args back to its concrete type.
type recorder interface {
	RMSNorm(x, g, o buffer, rows, dim int, eps float32) error
	LayerNorm(x, g, b, o buffer, rows, dim int, eps float32) error
	AddBias(x, b, o buffer, rows, n int) error
	MatMul(a, b, c buffer, m, k, n int) error
	// MatMulAcc records c += a·b — the residual-add epilogue (§T613): the projection lands
	// directly in the running residual stream, saving the separate elementwise-add dispatch.
	MatMulAcc(a, b, c buffer, m, k, n int) error
	RoPE(q, inv, o buffer, seq, width, heads, hd, half, pos int, posDiv float32) error
	// RoPEAt rotates a sub-row view living at float-element offset `off` inside a wider
	// buffer — the q/k bands of a fused QKV projection output (§T613). width acts as the
	// ROW STRIDE; only the first heads·hd columns of each row are rotated, in place.
	RoPEAt(q, inv, o buffer, off, seq, width, heads, hd, half, pos int, posDiv float32) error
	// RoPEPair rotates the q band (headsQ heads at offQ) AND the k band (headsK at offK) of
	// a fused QKV buffer in ONE dispatch (§T613), rows `stride` floats wide, in place.
	RoPEPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, half, pos int, posDiv float32) error
	// RoPEPartialPair is the partial-rotary sibling of RoPEPair: only the first rotaryDim
	// channels of each head in the q and k bands rotate (GPT-NeoX/Phi/StableLM partial
	// rotary). inv must be the [rotaryDim/2] frequency table. Implemented on cuda; metal and
	// vulkan return an unsupported error until a partial-rotary decoder targets them.
	RoPEPartialPair(qkv, inv buffer, seq, stride, headsQ, offQ, headsK, offK, hd, rotaryDim, pos int, posDiv float32) error
	Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error
	// Copy2D moves a strided rows×rowFloats sub-matrix (the fused-QKV band extraction, §T613):
	// row r copies from src[srcOff+r·srcStride:] to dst[dstOff+r·dstStride:].
	Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error
	MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error
	// MHACap is MHA with a Gemma-2 attention-logit soft-cap (cap·tanh(score·scale/cap) before the
	// mask+softmax). Implemented on cuda; metal/vulkan return unsupported until a softcap decoder
	// targets them (the RoPEPartialPair pattern).
	MHACap(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale, cap float32) error
	// MHAALiBi is MHA with an ALiBi position bias (MPT): slopeₕ·(j−qabs) added to each scaled score
	// before the mask+softmax; slopes holds `heads` per-head slopes. cuda-only (metal/vulkan stub).
	MHAALiBi(q, k, v, o, slopes buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error
	// MoEGate writes Mixtral top-k renormalized routing weights [rows·e] from router logits [rows·e].
	// RowAxpy accumulates dst[r,:] += arow[r]·src[r,:] (the MoE weighted combine). Both cuda-only.
	MoEGate(logits, weights buffer, rows, e, k, raw int, scale float32) error
	RowAxpy(dst, src, arow buffer, rows, cols int) error
	// MHARect is multi-head attention with a rectangular head shape: query/key share head width dqk,
	// value has a different head width dv (o is heads·dv wide). The DeepSeek-V2 MLA shape. cuda-only.
	MHARect(q, k, v, o buffer, sq, sk, heads, kvHeads, dqk, dv, causal, window int, scale float32) error
	// SSMStep advances ONE Mamba selective-scan timestep (decode): given u/Δ/A/B/C[d,·] and the skip
	// D[d], it updates the [d,n] recurrent state h IN PLACE and writes y[d]. Conv1DStep advances one
	// causal depthwise-conv step: out[c]=Σ_k w[c,k]·window[c,k]+b[c], then shifts the [d,k-1] state and
	// appends x. Both carry per-block state across calls (no KV cache) — the GPU Mamba block. cuda-only.
	SSMStep(u, delta, a, b, c, dskip, h, y buffer, d, n int) error
	Conv1DStep(x, w, b, state, out buffer, d, k int) error
	// SSDStep advances ONE Mamba-2 SSD timestep: H heads with a SCALAR per-head decay a=exp(Δ[h]·A[h])
	// and B/C shared across a group (g=h/(H/G)); updates the per-head state[H,N,P] in place and writes
	// y[H·P]. x/y are [H·headDim]=intermediate; delta/a/dskip are [H]; b/c are [G·N]. cuda-only.
	SSDStep(x, delta, a, b, c, dskip, state, y buffer, heads, headDim, groups, n int) error
	// WKVStep advances ONE RWKV-4 WKV recurrence timestep: k/v/w/u/out are [d] (w = per-channel decay
	// exp(WLog), u = current-token bonus); the running state aa/bb/pp [d] is updated in place (fresh
	// sequence: aa=bb=0, pp=−1e38). No KV cache — RWKV's O(1) recurrent inference. cuda-only.
	WKVStep(k, v, w, u, aa, bb, pp, out buffer, d int) error
	Unary(x, o buffer, op int) error
	Binary(a, b, o buffer, op int) error
	QMatMulResident(x buffer, w qweight, o buffer, m int) error
	// Commit submits without waiting; Wait blocks until done (§T614 encode-overlap split).
	// Backends without async submit implement Commit as a deferral and Wait as Finish.
	Commit() error
	Wait() error
	Finish() error
	Free()
}

// backendOps is the per-backend constructor vtable the adapters fill in.
type backendOps struct {
	name          string
	newBuffer     func([]float32) (buffer, error)
	newRecorder   func() (recorder, error)
	uploadQWeight func(weight []byte, qt uint32, n, k int) (qweight, error) // resident quantized upload
	// asyncEncode: the backend can encode a second recorder while one executes (§T614).
	// Metal command buffers are independent objects → true; the vulkan bridge keeps ONE
	// global recording context → false (pre-encoding there would clobber the in-flight one).
	asyncEncode bool
}

// moeFFN is one sparse-MoE expert: a SwiGLU FFN (gate/up/down) with its own weights.
type moeFFN struct{ wG, wU, wD linear }

type block struct {
	wq, wk, wv, wo, wG, wU, wD linear
	// wqkv is the fused QKV projection (§T613): one [in, D+2·kvDim] weight whose output row
	// is [q | k | v]. Non-nil = Step/StepN run the fused chain (one matmul instead of three);
	// nil = fall back to wq/wk/wv (quantized blocks whose projections mix quant types).
	wqkv                linear
	gAttn, gFFN, kC, vC buffer
	bAttn, bFFN         buffer // LayerNorm betas (nil ⇒ RMSNorm, the Llama default)
	// projection biases (nil ⇒ no bias): the fused-QKV bias [D+2·kvDim], the o-proj bias [D],
	// and the GELU-MLP fc [hidden] / proj [D] biases — GPT-NeoX/Phi/StarCoder2 carry them.
	qkvBias, oBias, fcBias, projBias                      buffer
	qN, kN                                                buffer   // per-head query/key RMSNorm gains [dk] (Qwen3); nil ⇒ no QK-norm
	gAttn2, gFFN2                                         buffer   // sandwich-norm OUTPUT gains (Gemma2 post_attention/post_feedforward); nil otherwise
	moeRouter                                             linear   // sparse-MoE gate: dim → nExperts logits (Mixtral); nil unless d.moe
	moeExperts                                            []moeFFN // the per-expert SwiGLU FFNs (Mixtral); len == d.nExperts
	moeShared                                             moeFFN   // Qwen2-MoE shared expert (SwiGLU on every token); zero-value when absent
	moeSharedGate                                         linear   // Qwen2-MoE shared-expert gate: dim → 1 logit, sigmoid-scaled; nil when absent
	wqA, wqB, wkvA, wkvB                                  linear   // DeepSeek-V2 MLA low-rank projections (q_a/q_b/kv_a/kv_b); nil unless d.mla
	gQA, gKvA                                             buffer   // MLA latent RMSNorm gains (q_a_layernorm / kv_a_layernorm)
	mInX, mInZ, mDtLow, mDtProj, mBProj, mCProj, mOutProj linear   // Mamba block projections; nil unless d.mamba
	mConvW, mConvB, mA, mDskip, mDtBias                   buffer   // Mamba conv filters+bias, A=−exp(A_log) [dInner,N], D skip [dInner], Δ bias [dInner]
	mConvState, mSsmState                                 buffer   // per-block Mamba decode state: conv window [dInner,dConv-1], ssm state [dInner,N]
	m2InProj, m2OutProj                                   linear   // Mamba-2 fused in_proj [d,projSize] / out_proj [inter,d]; nil unless d.mamba2
	m2ConvW, m2ConvB, m2A, m2DtBias, m2Dskip, m2NormW     buffer   // Mamba-2 conv filters+bias [convDim,·], A/Δbias/D [numHeads], gated-RMSNorm gain [inter]
	m2ConvState, m2SsdState                               buffer   // Mamba-2 state: conv window [convDim,dConv-1], SSD state [numHeads·N·headDim]
	// RWKV-4 (nil unless d.rwkv): time-mix reuses wq/wk/wv/wo = Wr/Wk/Wv/Wo, channel-mix reuses
	// wG/wD = CWk/CWv; LN1 = gAttn/bAttn, LN2 = gFFN/bFFN. rwCWr is the channel-mix receptance.
	rwCWr                                linear // channel-mix receptance CWr [Dim,Dim]
	rwMuR, rwMuK, rwMuV, rwCMuR, rwCMuK  buffer // token-shift interpolators μ [Dim]
	rwOmR, rwOmK, rwOmV, rwOmCR, rwOmCK  buffer // (1−μ) precomputed [Dim]
	rwW, rwU                             buffer // WKV per-channel decay w=exp(WLog) [Dim], bonus u [Dim]
	rwPrevTM, rwPrevCM, rwAA, rwBB, rwPP buffer // per-block RWKV recurrent state [Dim] (token-shift + WKV aa/bb/pp)
}

// Decoder holds a Llama's weights + KV cache as device-resident buffers and runs one batched decode
// step per token. Not safe for concurrent use — one Decoder per goroutine. Release when done.
type Decoder struct {
	ops                                           backendOps
	d, h, kvH, dk, kvDim, half, hidden, v, maxLen int
	qDim                                          int     // attention Q/O width = h·dk; == d (model dim) unless head_dim is decoupled (Gemma)
	rotaryDim                                     int     // rotated channels/head; == dk for full RoPE (Llama), < dk for partial rotary (GPT-NeoX/Phi/StableLM)
	lnBias                                        bool    // true ⇒ LayerNorm-with-bias norms (StableLM/Phi/StarCoder2); false ⇒ RMSNorm (Llama)
	ffnGELU                                       bool    // true ⇒ 2-layer GELU MLP (fc→GELU→proj: GPT-NeoX/Phi/StarCoder2); false ⇒ SwiGLU (Llama/StableLM)
	ffnReLU2                                      bool    // true ⇒ 2-layer squared-ReLU MLP (fc→relu²→proj: Nemotron); mutually exclusive with ffnGELU
	ffnGEGLU                                      bool    // true ⇒ GeGLU MLP: down(GELU(gate)⊙up) — Gemma's GELU-gated variant of SwiGLU
	parallelRes                                   bool    // true ⇒ parallel residual: attn+FFN both read norm(x0), sum onto x (GPT-NeoX/Phi/Cohere); false ⇒ sequential (Llama)
	parallelTwoNorm                               bool    // parallel residual with SEPARATE attn/FFN norms both over x0 (GPT-NeoX); needs the FFN norm computed pre-attn-add
	qkNorm                                        bool    // true ⇒ RMSNorm on Q and K before RoPE (Qwen3 per-head, OLMo2 full-width); false otherwise
	qkNormFull                                    bool    // with qkNorm: true ⇒ one RMSNorm over the whole q/k projection (OLMo2), false ⇒ per-head (Qwen3)
	postNorm                                      bool    // true ⇒ post-norm blocks (OLMo2): sublayer reads the raw residual, its OUTPUT is normed before the add
	sandwich                                      bool    // true ⇒ sandwich norms (Gemma2): pre-norm the input (gAttn/gFFN) AND post-norm the output (gAttn2/gFFN2)
	attnCap                                       float32 // >0 ⇒ Gemma2 attention-logit soft-cap magnitude (via MHACap); 0 ⇒ plain MHA
	noRope                                        bool    // true ⇒ no rotary position embedding (MPT: position enters via ALiBi only)
	aliBiSlopes                                   buffer  // non-nil ⇒ per-head ALiBi slopes [heads]; attention runs via MHAALiBi (MPT)
	moe                                           bool    // true ⇒ the FFN sublayer is a sparse Mixture-of-Experts (Mixtral): route → experts → combine
	nExperts, topK                                int     // MoE expert count and experts-per-token (when moe)
	moeRaw                                        bool    // true ⇒ DeepSeek-V2 raw-softmax·routedScale gating (no top-k renormalization); false ⇒ Mixtral renorm
	routedScale                                   float32 // DeepSeek-V2 routed-expert weight multiplier (routed_scaling_factor); 1 otherwise
	mla                                           bool    // true ⇒ DeepSeek-V2 Multi-head Latent Attention (rectangular, low-rank latent KV, decoupled RoPE)
	mamba                                         bool    // true ⇒ Mamba selective-state-space blocks (no attention, no KV cache; linear-time recurrent decode)
	dInner, mambaN, dConv, dtRank                 int     // Mamba block dims: inner width (e·d), SSM state size, conv kernel, Δ low-rank
	mamba2                                        bool    // true ⇒ Mamba-2 SSD blocks (scalar-per-head decay, grouped B/C, fused in_proj, gated RMSNorm)
	m2H, m2P, m2G, m2N, m2Conv, m2Inter, m2CD     int     // Mamba-2 dims: num_heads, head_dim, n_groups, state_size, conv kernel, intermediate, conv_dim
	rwkv                                          bool    // true ⇒ RWKV-4 blocks (WKV time-mix + gated squared-ReLU channel-mix; no attention/KV cache; O(1) recurrent)
	rwHidden                                      int     // RWKV channel-mix hidden width (CWk: Dim→Hidden, CWv: Hidden→Dim)
	qkNope, qkRope, vHead, qkHead                 int     // MLA per-head dims: nope/rope query-key parts, value width, qkHead=qkNope+qkRope
	qLoRA, kvLoRA                                 int     // MLA query/kv latent compression ranks (q_lora_rank / kv_lora_rank)
	mlaScale, mlaPosDiv                           float32 // MLA pre-softmax scale (1/√qkHead default) and the QKRope rope posDiv
	eps, posDiv, scale                            float32
	embMult                                       float32 // gathered embeddings ×= embMult (IBM Granite EmbeddingMult); 1 for everything else

	blocks                                                         []block
	out                                                            linear
	gFinal, bFinal, outBias, dinv                                  *bufSlot
	dx, xn, xn2, q, k, v_, attn, ao, gate, up, mo, logits          *bufSlot
	moeGate, moeW, moeCol                                          *bufSlot       // sparse-MoE scratch: router logits [·,E], routing weights [·,E], one weight column [·]
	mlaCQ, mlaQ, mlaComp, mlaLatent, mlaKV, mlaAttn, mlaInv        *bufSlot       // MLA scratch: compressed query, fused Q, kv_a out, normed kv latent, kv_b out, attn out, QKRope freqs
	mbXin, mbZ, mbXc, mbDtLow, mbDelta, mbB, mbC, mbY              *bufSlot       // Mamba scratch (rows=1): in_x, gate branch, conv+SiLU out, Δ_low, Δ, B, C, scan out
	m2Proj, m2Z, m2XBC, m2Dt, m2Xc, m2X, m2B, m2C, m2Y             *bufSlot       // Mamba-2 scratch (rows=1): in_proj out, gate z, conv input xBC, Δ pre-act, conv+SiLU out, value x, B, C, SSD out
	rwPreLNg, rwPreLNb                                             *bufSlot       // RWKV ln0 (pre-block-0 LayerNorm) γ/β
	rwMix, rwR, rwK, rwV, rwWkv, rwO, rwT1, rwT2, rwCr, rwCk, rwCv *bufSlot       // RWKV scratch (rows=1): token-shift mix, r/k/v, WKV out, time-mix out, 2 mix temps, chan-mix cr, ck [Hidden], cv
	qkv                                                            *bufSlot       // fused QKV output rows [·, d+2·kvDim] (§T613)
	pending                                                        recorder       // pre-encoded next-step command buffer (§T614)
	pendingPos                                                     int            // position pending encodes; -1 = none
	table                                                          *tensor.Tensor // token embedding, host-side gather source
	invHost                                                        []float32      // RoPE inverse freqs (uploaded by allocScratch)
	all                                                            []buffer
	qweights                                                       []qweight // resident quantized weights (quant decoder)
}

// bufSlot wraps a buffer so the struct fields read naturally while sharing the release list.
type bufSlot struct{ b buffer }

// newDecoderCommon sets the geometry, RoPE frequencies and scratch buffers shared by the f32 and
// quantized constructors.
func newDecoderCommon(cfg nlp.LlamaConfig, tokEmb *tensor.Tensor, ops backendOps) (*Decoder, error) {
	d := &Decoder{
		ops: ops,
		d:   cfg.Dim, h: cfg.Heads, kvH: cfg.KVHeads, hidden: cfg.Hidden, v: cfg.Vocab,
		maxLen: cfg.Ctx, eps: float32(cfg.Eps), table: tokEmb,
		pendingPos: -1, embMult: 1,
	}
	if d.kvH <= 0 {
		d.kvH = d.h
	}
	if d.h <= 0 || d.d%d.h != 0 {
		return nil, fmt.Errorf("llamagpu(%s): dim %d not divisible by heads %d", ops.name, d.d, d.h)
	}
	d.dk = d.d / d.h
	d.qDim = d.h * d.dk // = d.d for standard models; decoupled-head_dim constructors (Gemma) override
	d.kvDim = d.kvH * d.dk
	d.half = d.dk / 2
	d.rotaryDim = d.dk // full RoPE by default; partial-rotary constructors override before allocScratch
	d.scale = float32(1.0 / math.Sqrt(float64(d.dk)))

	invF64, posDiv64 := backend.RoPEFreqs(d.dk, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.half)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)
	return d, nil
}

// ropeQK records RoPE over the q and k bands of the fused QKV buffer in one pass: full rotary
// (RoPEPair) when rotaryDim == dk (Llama), else partial rotary (RoPEPartialPair — only the first
// rotaryDim channels of each head rotate; GPT-NeoX/Phi/StableLM). inv (d.dinv) is sized to
// rotaryDim/2 so the same buffer feeds both.
// recordFFN records the feed-forward sublayer with the residual-add epilogue (dx += ffn(xn2)):
// a 2-layer GELU MLP (fc → GELU → proj, GPT-NeoX/Phi/StarCoder2) when ffnGELU, else the SwiGLU
// (silu(gate)·up → down, Llama/StableLM). wG/wU/wD are c_fc/·/c_proj or gate/up/down accordingly.
// recordQKVProj records the fused QKV projection (xn·Wqkv) plus its optional bias — the bias is
// added BEFORE RoPE, per the [q|k|v] band layout (GPT-NeoX/Phi/StarCoder2 have biased q/k/v).
// recordAttnNorm produces the attention sublayer's input in d.xn. Pre-norm (Llama & most): xn =
// norm(x0) (plus norm2(x0)→xn2 for two-norm parallel). Post-norm (OLMo2): attention reads the RAW
// residual, so xn is just a copy of x0 and the norm happens later on the attention output.
func (d *Decoder) recordAttnNorm(r recorder, b block, rows int) error {
	if d.postNorm {
		return r.Blit(d.dx.b, 0, d.xn.b, 0, rows*d.d)
	}
	e := d.norm(r, d.dx.b, b.gAttn, b.bAttn, d.xn.b, rows)
	if d.parallelTwoNorm { // norm2(x0) BEFORE the attn add, for the FFN branch
		e = firstErr(e, d.norm(r, d.dx.b, b.gFFN, b.bFFN, d.xn2.b, rows))
	}
	return e
}

func (d *Decoder) recordQKVProj(r recorder, b block, rows int) error {
	e := b.wqkv.record(r, d.xn.b, d.qkv.b, rows)
	if b.qkvBias != nil {
		e = firstErr(e, r.AddBias(d.qkv.b, b.qkvBias, d.qkv.b, rows, d.qDim+2*d.kvDim))
	}
	return e
}

// recordQKNorm records Qwen3's per-head RMSNorm on the Q and K bands of the fused QKV buffer,
// applied BEFORE RoPE. Each head's dk channels are normed independently with a shared gain — the
// reference reshapes [seq, heads·dk] → [seq·heads, dk], RMSNorms, reshapes back. The heads of one
// token are contiguous but tokens are stride-separated inside qkv, so we Copy2D the band out to a
// packed [seq, band] scratch (reusing d.q / d.k, unused in the fused path until after RoPE),
// RMSNorm it as seq·heads rows of dk, and Copy2D it back. No-op when qkNorm is unset.
func (d *Decoder) recordQKNorm(r recorder, b block, seq, stride int) error {
	if !d.qkNorm {
		return nil
	}
	// rows×dim for the RMSNorm: OLMo2 norms the WHOLE q/k projection as one vector (rows=seq,
	// dim=qDim/kvDim, gain [qDim]/[kvDim]); Qwen3 norms each head (rows=seq·heads, dim=dk, gain [dk]).
	qRows, qCols, kRows, kCols := seq*d.h, d.dk, seq*d.kvH, d.dk
	if d.qkNormFull {
		qRows, qCols, kRows, kCols = seq, d.qDim, seq, d.kvDim
	}
	e := r.Copy2D(d.qkv.b, 0, stride, d.q.b, 0, d.qDim, seq, d.qDim) // extract Q band [seq, qDim]
	e = firstErr(e, r.RMSNorm(d.q.b, b.qN, d.q.b, qRows, qCols, d.eps))
	e = firstErr(e, r.Copy2D(d.q.b, 0, d.qDim, d.qkv.b, 0, stride, seq, d.qDim)) // write Q band back
	e = firstErr(e, r.Copy2D(d.qkv.b, d.qDim, stride, d.k.b, 0, d.kvDim, seq, d.kvDim))
	e = firstErr(e, r.RMSNorm(d.k.b, b.kN, d.k.b, kRows, kCols, d.eps))
	e = firstErr(e, r.Copy2D(d.k.b, 0, d.kvDim, d.qkv.b, d.qDim, stride, seq, d.kvDim))
	return e
}

// recordOProj records the attention output projection with its residual-add epilogue (dx +=
// attn·Wo) plus the optional o-bias broadcast onto the residual stream.
// recordMHA runs the attention core, dispatching to the soft-capped kernel (Gemma2) when attnCap>0.
func (d *Decoder) recordMHA(r recorder, q, kC, vC, o buffer, sq, sk int) error {
	if d.aliBiSlopes != nil {
		return r.MHAALiBi(q, kC, vC, o, d.aliBiSlopes, sq, sk, d.qDim, d.h, d.kvH, d.dk, 1, 0, d.scale)
	}
	if d.attnCap > 0 {
		return r.MHACap(q, kC, vC, o, sq, sk, d.qDim, d.h, d.kvH, d.dk, 1, 0, d.scale, d.attnCap)
	}
	return r.MHA(q, kC, vC, o, sq, sk, d.qDim, d.h, d.kvH, d.dk, 1, 0, d.scale)
}

func (d *Decoder) recordOProj(r recorder, b block, rows int) error {
	if d.postNorm || d.sandwich {
		// Output-normed residual: ao = attn·Wo → norm(ao) → dx += ao. OLMo2 post-norm reuses gAttn;
		// Gemma2 sandwich has a SEPARATE post_attention_layernorm gain (gAttn2), the input already
		// having been pre-normed with gAttn.
		g, bta := b.gAttn, b.bAttn
		if d.sandwich {
			g, bta = b.gAttn2, nil
		}
		return firstErr(
			b.wo.record(r, d.attn.b, d.ao.b, rows),
			d.norm(r, d.ao.b, g, bta, d.ao.b, rows),
			r.Binary(d.dx.b, d.ao.b, d.dx.b, binaryAdd),
		)
	}
	e := b.wo.recordAdd(r, d.attn.b, d.ao.b, d.dx.b, rows)
	if b.oBias != nil {
		e = firstErr(e, r.AddBias(d.dx.b, b.oBias, d.dx.b, rows, d.d))
	}
	return e
}

// recordDownProj records the FFN down-projection's residual epilogue: dx += act·Wdown, either fused
// (pre-norm) or, for OLMo2 post-norm, as mo = act·Wdown → post_feedforward_layernorm(mo) → dx += mo.
func (d *Decoder) recordDownProj(r recorder, b block, rows int) error {
	if d.postNorm || d.sandwich {
		g, bta := b.gFFN, b.bFFN
		if d.sandwich {
			g, bta = b.gFFN2, nil // Gemma2 post_feedforward_layernorm (input pre-normed with gFFN)
		}
		return firstErr(
			b.wD.record(r, d.gate.b, d.mo.b, rows),
			d.norm(r, d.mo.b, g, bta, d.mo.b, rows),
			r.Binary(d.dx.b, d.mo.b, d.dx.b, binaryAdd),
		)
	}
	return b.wD.recordAdd(r, d.gate.b, d.mo.b, d.dx.b, rows)
}

// recordLogits records the final projection to logits (xn·Out) plus the optional output bias
// (Phi's biased untied lm_head; nil ⇒ no bias, as for Llama/StableLM/StarCoder2).
func (d *Decoder) recordLogits(r recorder, rows int) error {
	e := d.out.record(r, d.xn.b, d.logits.b, rows)
	if d.outBias != nil {
		e = firstErr(e, r.AddBias(d.logits.b, d.outBias.b, d.logits.b, rows, d.v))
	}
	return e
}

// recordFFNSublayer records the FFN norm + the FFN with its residual-add epilogue. Sequential
// (Llama/StableLM/StarCoder2): norm the post-attention residual, then FFN(that). PARALLEL residual
// (Phi/Cohere one-norm): the FFN reuses the SAME normed input as attention (d.xn = norm(x0), which
// survives attention since recordOProj writes dx, not xn) and adds onto the residual in parallel —
// no second norm. Both leave dx += FFN(·).
func (d *Decoder) recordFFNSublayer(r recorder, b block, rows int) error {
	if b.moeRouter != nil {
		// sparse MoE — per BLOCK (DeepSeek-V2 mixes dense and MoE layers): pre-norm, then route →
		// per-expert SwiGLU → weighted combine (+ optional shared expert).
		if e := d.norm(r, d.dx.b, b.gFFN, b.bFFN, d.xn2.b, rows); e != nil {
			return e
		}
		return d.recordMoE(r, b, d.xn2.b, rows)
	}
	if d.postNorm {
		return d.recordFFN(r, b, d.dx.b, rows) // post-norm: FFN reads the raw residual; recordDownProj norms its output
	}
	if d.parallelRes {
		return d.recordFFN(r, b, d.xn.b, rows) // parallel: FFN(norm(x0)), same norm as attn
	}
	if d.parallelTwoNorm {
		return d.recordFFN(r, b, d.xn2.b, rows) // two-norm parallel: xn2 = norm2(x0), computed pre-attn-add
	}
	if e := d.norm(r, d.dx.b, b.gFFN, b.bFFN, d.xn2.b, rows); e != nil {
		return e
	}
	return d.recordFFN(r, b, d.xn2.b, rows)
}

// recordMoE records the sparse Mixture-of-Experts FFN sublayer over the normed input `in`:
// gate logits = in·Router → top-k renormalized weights (MoEGate) → for every expert, SwiGLU(in)
// scaled by that expert's per-token weight and accumulated straight onto the residual dx (RowAxpy).
// This is DENSE eval (all experts run); the non-selected experts get weight 0, so it is exact —
// the sparse top-k gather is a later optimization. The batched command buffer stays valid because
// the routing weights are computed on-device, not by host control flow.
func (d *Decoder) recordMoE(r recorder, b block, in buffer, rows int) error {
	e := b.moeRouter.record(r, in, d.moeGate.b, rows) // [rows, nExperts] router logits
	e = firstErr(e, r.MoEGate(d.moeGate.b, d.moeW.b, rows, d.nExperts, d.topK, boolToInt(d.moeRaw), d.routedScale))
	for ex := range b.moeExperts {
		xp := b.moeExperts[ex]
		e = firstErr(e,
			xp.wG.record(r, in, d.gate.b, rows),
			xp.wU.record(r, in, d.up.b, rows),
			r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU),            // silu(gate)·up
			xp.wD.record(r, d.gate.b, d.mo.b, rows),                       // expert output → mo (non-accumulating)
			r.Copy2D(d.moeW.b, ex, d.nExperts, d.moeCol.b, 0, 1, rows, 1), // extract weight column [rows]
			r.RowAxpy(d.dx.b, d.mo.b, d.moeCol.b, rows, d.d),              // dx += w_ex · expert_ex
		)
	}
	if b.moeShared.wG != nil {
		// Shared expert run on every token. Qwen2-MoE gates it by sigmoid(in·gate); DeepSeek-V2 runs
		// it unconditionally (no gate), so its SwiGLU accumulates straight onto the residual.
		s := b.moeShared
		e = firstErr(e,
			s.wG.record(r, in, d.gate.b, rows),
			s.wU.record(r, in, d.up.b, rows),
			r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU), // shared SwiGLU → gate
		)
		if b.moeSharedGate != nil { // Qwen2-MoE: sigmoid-gated combine
			e = firstErr(e,
				s.wD.record(r, d.gate.b, d.mo.b, rows),
				b.moeSharedGate.record(r, in, d.moeCol.b, rows),
				r.Unary(d.moeCol.b, d.moeCol.b, unarySigmoid),
				r.RowAxpy(d.dx.b, d.mo.b, d.moeCol.b, rows, d.d), // dx += sigmoid(gate)·shared_out
			)
		} else { // DeepSeek-V2: ungated — dx += shared·Wdown directly
			e = firstErr(e, s.wD.recordAdd(r, d.gate.b, d.mo.b, d.dx.b, rows))
		}
	}
	return e
}

// recordMLAAttention records DeepSeek-V2 Multi-head Latent Attention for `rows` new tokens at
// position pos, leaving dx += attn·Wo. It fuses the reference's head-by-head build: input_layernorm →
// low-rank q/kv latent projections (each with its own RMSNorm) → decoupled RoPE on the per-head q_pe
// and the SHARED k_pe → assemble the rectangular per-head Q [qkHead] / K [qkHead] (k_pe broadcast to
// every head) and the VHead-wide V straight into the KV cache → MHARect (rectangular) → o_proj.
func (d *Decoder) recordMLAAttention(r recorder, b block, pos, rows int) error {
	H, qkH, nope, rope, vH := d.h, d.qkHead, d.qkNope, d.qkRope, d.vHead
	kvH := nope + vH // kv_b per-head width (k_nope + value)
	qDim, vDim := H*qkH, H*vH
	compW := d.kvLoRA + rope // kv_a output width: [kv_latent | k_pe]

	e := d.norm(r, d.dx.b, b.gAttn, b.bAttn, d.xn.b, rows) // input_layernorm → xn

	// Query: q = q_b_proj(q_a_layernorm(q_a_proj(xn))), then decoupled RoPE on each head's q_pe.
	e = firstErr(e,
		b.wqA.record(r, d.xn.b, d.mlaCQ.b, rows),
		r.RMSNorm(d.mlaCQ.b, b.gQA, d.mlaCQ.b, rows, d.qLoRA, d.eps),
		b.wqB.record(r, d.mlaCQ.b, d.mlaQ.b, rows), // [rows, H·qkH]
	)
	for h := 0; h < H; h++ {
		e = firstErr(e, r.RoPEAt(d.mlaQ.b, d.mlaInv.b, d.mlaQ.b, h*qkH+nope, rows, qDim, 1, rope, rope/2, pos, d.mlaPosDiv))
	}

	// KV: compressed = kv_a_proj(xn) → [kv_latent | k_pe]; RoPE the shared k_pe in place; norm the
	// latent and expand kv = kv_b_proj(kv_a_layernorm(latent)).
	e = firstErr(e,
		b.wkvA.record(r, d.xn.b, d.mlaComp.b, rows),
		r.RoPEAt(d.mlaComp.b, d.mlaInv.b, d.mlaComp.b, d.kvLoRA, rows, compW, 1, rope, rope/2, pos, d.mlaPosDiv),
		r.Copy2D(d.mlaComp.b, 0, compW, d.mlaLatent.b, 0, d.kvLoRA, rows, d.kvLoRA), // extract kv_latent
		r.RMSNorm(d.mlaLatent.b, b.gKvA, d.mlaLatent.b, rows, d.kvLoRA, d.eps),
		b.wkvB.record(r, d.mlaLatent.b, d.mlaKV.b, rows), // [rows, H·kvH]
	)

	// Assemble the new tokens' K/V into the cache at row pos: K[h] = [k_nope_h | k_pe(shared)],
	// V[h] = value_h. Copy2D writes rows-many rows at cache-row offset pos.
	for h := 0; h < H; h++ {
		e = firstErr(e,
			r.Copy2D(d.mlaKV.b, h*kvH, H*kvH, b.kC, pos*qDim+h*qkH, qDim, rows, nope),           // k_nope
			r.Copy2D(d.mlaComp.b, d.kvLoRA, compW, b.kC, pos*qDim+h*qkH+nope, qDim, rows, rope), // shared k_pe → every head
			r.Copy2D(d.mlaKV.b, h*kvH+nope, H*kvH, b.vC, pos*vDim+h*vH, vDim, rows, vH),         // value
		)
	}

	// Rectangular attention over the cache [0 : pos+rows], then the output projection onto dx.
	return firstErr(e,
		r.MHARect(d.mlaQ.b, b.kC, b.vC, d.mlaAttn.b, rows, pos+rows, H, H, qkH, vH, 1, 0, d.mlaScale),
		b.wo.recordAdd(r, d.mlaAttn.b, d.mo.b, d.dx.b, rows), // dx += attn·Wo
	)
}

// recordMambaBlock records ONE Mamba selective-state-space block for a single decode token
// (rows==1), leaving dx += mixer(RMSNorm(dx)). It mirrors nn.MambaBlock.Forward exactly, but the
// full-sequence OpConv1D/OpSSM are replaced by their per-timestep decode kernels (Conv1DStep,
// SSMStep) that carry the block's conv-window and SSM state across Step calls — no attention, no KV
// cache, O(1) work per token. The per-token recurrence is why Mamba records rows==1 only (StepN
// loops it): each token's state update depends on the previous token's. Only DtProj and the conv
// carry a bias (the HF checkpoint's other projections are bias-free); the Δ bias lands inside
// softplus. A = −exp(A_log) is precomputed on the host (b.mA).
func (d *Decoder) recordMambaBlock(r recorder, b block) error {
	dI, N, K := d.dInner, d.mambaN, d.dConv
	// pre-norm → xn = RMSNorm(dx); the up-projection splits into the x branch and the z gate.
	e := d.norm(r, d.dx.b, b.gAttn, nil, d.xn.b, 1) // RMSNorm (lnBias false for Mamba)
	e = firstErr(e,
		b.mInX.record(r, d.xn.b, d.mbXin.b, 1), // xin = xn·InX  [dInner]
		b.mInZ.record(r, d.xn.b, d.mbZ.b, 1),   // z   = xn·InZ  [dInner]
		// local mixing: one causal depthwise-conv step (state carried), then SiLU.
		r.Conv1DStep(d.mbXin.b, b.mConvW, b.mConvB, b.mConvState, d.mbXc.b, dI, K),
		r.Unary(d.mbXc.b, d.mbXc.b, unarySiLU),
		// input-dependent Δ = softplus(dt_proj(dt_low_proj(xc)) + dt_bias).
		b.mDtLow.record(r, d.mbXc.b, d.mbDtLow.b, 1),     // Δ_low [dtRank]
		b.mDtProj.record(r, d.mbDtLow.b, d.mbDelta.b, 1), // → [dInner]
		r.AddBias(d.mbDelta.b, b.mDtBias, d.mbDelta.b, 1, dI),
		r.Unary(d.mbDelta.b, d.mbDelta.b, unarySoftplus),
		// input-dependent SSM params B, C.
		b.mBProj.record(r, d.mbXc.b, d.mbB.b, 1), // B [N]
		b.mCProj.record(r, d.mbXc.b, d.mbC.b, 1), // C [N]
		// selective scan: one recurrent timestep (state carried), y[dInner].
		r.SSMStep(d.mbXc.b, d.mbDelta.b, b.mA, d.mbB.b, d.mbC.b, b.mDskip, b.mSsmState, d.mbY.b, dI, N),
		// gate y ⊙ SiLU(z), then down-project onto the residual: dx += y·OutProj.
		r.Unary(d.mbZ.b, d.mbZ.b, unarySiLU),
		r.Binary(d.mbY.b, d.mbZ.b, d.mbY.b, binaryMul),
		b.mOutProj.recordAdd(r, d.mbY.b, d.mo.b, d.dx.b, 1),
	)
	return e
}

// resetMambaState zeroes every block's conv-window and SSM state — called at the start of a fresh
// sequence (pos==0), since the recurrence carries state across Step calls with no KV cache to reset.
func (d *Decoder) resetMambaState() error {
	zc := make([]float32, d.dInner*(d.dConv-1))
	zs := make([]float32, d.dInner*d.mambaN)
	for _, b := range d.blocks {
		if e := b.mConvState.UploadF32(zc); e != nil {
			return e
		}
		if e := b.mSsmState.UploadF32(zs); e != nil {
			return e
		}
	}
	return nil
}

// allocMambaScratch allocates the rows==1 Mamba decode scratch. Mamba has no attention or KV cache,
// so it skips the q/k/v/attn/qkv/gate/up buffers allocScratch would make; it keeps only the residual
// (dx), the pre-norm output (xn), the down-projection scratch (mo), the logits, and the per-block
// intermediates. Everything is one row wide — the sequential recurrence never batches.
func (d *Decoder) allocMambaScratch(mk func(data []float32) *bufSlot) {
	dI, N, dtR := d.dInner, d.mambaN, d.dtRank
	d.dx = mk(make([]float32, d.d))
	d.xn = mk(make([]float32, d.d))
	d.mo = mk(make([]float32, d.d))
	d.logits = mk(make([]float32, d.v))
	d.mbXin = mk(make([]float32, dI))
	d.mbZ = mk(make([]float32, dI))
	d.mbXc = mk(make([]float32, dI))
	d.mbDtLow = mk(make([]float32, dtR))
	d.mbDelta = mk(make([]float32, dI))
	d.mbB = mk(make([]float32, N))
	d.mbC = mk(make([]float32, N))
	d.mbY = mk(make([]float32, dI))
}

// recordMamba2Block records ONE Mamba-2 SSD block for a single decode token (rows==1), leaving
// dx += mixer(RMSNorm(dx)). It mirrors nlp.Mamba2Mixer.forward: fused in_proj → split [z | xBC | dt]
// → causal-conv step + SiLU over conv_dim → split [x | B | C] → Δ=softplus(dt+dt_bias) → the SSD scan
// (cu_ssd_step: scalar per-head decay a=exp(Δ·A), grouped B/C, +D·x skip) → gated RMSNorm norm(y·SiLU(z))
// → out_proj onto the residual. A=−exp(A_log) precomputed on the host. Like Mamba-1 it is rows==1
// (sequential recurrence). Splits are contiguous rows==1 sub-ranges, extracted with Blit.
func (d *Decoder) recordMamba2Block(r recorder, b block) error {
	I, CD, H2 := d.m2Inter, d.m2CD, d.m2H
	gN := d.m2G * d.m2N
	// pre-norm → xn = RMSNorm(dx); fused in_proj → proj = [z(I) | xBC(CD) | dt(H2)].
	e := d.norm(r, d.dx.b, b.gAttn, nil, d.xn.b, 1) // RMSNorm (lnBias false)
	e = firstErr(e,
		b.m2InProj.record(r, d.xn.b, d.m2Proj.b, 1),
		r.Blit(d.m2Proj.b, 0, d.m2Z.b, 0, I),      // gate z
		r.Blit(d.m2Proj.b, I, d.m2XBC.b, 0, CD),   // conv input xBC
		r.Blit(d.m2Proj.b, I+CD, d.m2Dt.b, 0, H2), // Δ pre-activation
		// local mixing over conv_dim: causal depthwise-conv step (state carried) + SiLU.
		r.Conv1DStep(d.m2XBC.b, b.m2ConvW, b.m2ConvB, b.m2ConvState, d.m2Xc.b, CD, d.m2Conv),
		r.Unary(d.m2Xc.b, d.m2Xc.b, unarySiLU),
		// split conv output → x(value, I) | B(gN) | C(gN).
		r.Blit(d.m2Xc.b, 0, d.m2X.b, 0, I),
		r.Blit(d.m2Xc.b, I, d.m2B.b, 0, gN),
		r.Blit(d.m2Xc.b, I+gN, d.m2C.b, 0, gN),
		// Δ = softplus(dt + dt_bias) [num_heads].
		r.AddBias(d.m2Dt.b, b.m2DtBias, d.m2Dt.b, 1, H2),
		r.Unary(d.m2Dt.b, d.m2Dt.b, unarySoftplus),
		// SSD scan: scalar per-head decay, grouped B/C, +D·x skip → y[I].
		r.SSDStep(d.m2X.b, d.m2Dt.b, b.m2A, d.m2B.b, d.m2C.b, b.m2Dskip, b.m2SsdState, d.m2Y.b, H2, d.m2P, d.m2G, d.m2N),
		// gated RMSNorm: norm(y · SiLU(z)) · normW, over the intermediate width.
		r.Unary(d.m2Z.b, d.m2Z.b, unarySiLU),
		r.Binary(d.m2Y.b, d.m2Z.b, d.m2Y.b, binaryMul),
		r.RMSNorm(d.m2Y.b, b.m2NormW, d.m2Y.b, 1, I, d.eps),
		// out_proj onto the residual: dx += y · OutProj.
		b.m2OutProj.recordAdd(r, d.m2Y.b, d.mo.b, d.dx.b, 1),
	)
	return e
}

// resetMamba2State zeroes every block's conv-window and SSD state (fresh sequence, pos==0).
func (d *Decoder) resetMamba2State() error {
	zc := make([]float32, d.m2CD*(d.m2Conv-1))
	zs := make([]float32, d.m2H*d.m2N*d.m2P)
	for _, b := range d.blocks {
		if e := b.m2ConvState.UploadF32(zc); e != nil {
			return e
		}
		if e := b.m2SsdState.UploadF32(zs); e != nil {
			return e
		}
	}
	return nil
}

// allocMamba2Scratch allocates the rows==1 Mamba-2 decode scratch (no attention/KV buffers).
func (d *Decoder) allocMamba2Scratch(mk func(data []float32) *bufSlot) {
	I, CD, H2 := d.m2Inter, d.m2CD, d.m2H
	gN := d.m2G * d.m2N
	projSize := I + CD + H2
	d.dx = mk(make([]float32, d.d))
	d.xn = mk(make([]float32, d.d))
	d.mo = mk(make([]float32, d.d))
	d.logits = mk(make([]float32, d.v))
	d.m2Proj = mk(make([]float32, projSize))
	d.m2Z = mk(make([]float32, I))
	d.m2XBC = mk(make([]float32, CD))
	d.m2Dt = mk(make([]float32, H2))
	d.m2Xc = mk(make([]float32, CD))
	d.m2X = mk(make([]float32, I))
	d.m2B = mk(make([]float32, gN))
	d.m2C = mk(make([]float32, gN))
	d.m2Y = mk(make([]float32, I))
}

// rwMix records the RWKV token-shift interpolation μ⊙x + (1−μ)⊙shift into out (rows==1). The
// reference forms it as shift + μ⊙(x−shift); with only add/mul recorder ops we use the algebraically
// identical μ⊙x + (1−μ)⊙shift, so mu and its precomputed complement omu are both uploaded.
func (d *Decoder) rwMixInto(r recorder, x, shift, mu, omu, out buffer) error {
	return firstErr(
		r.Binary(x, mu, d.rwT1.b, binaryMul),      // μ⊙x
		r.Binary(shift, omu, d.rwT2.b, binaryMul), // (1−μ)⊙shift
		r.Binary(d.rwT1.b, d.rwT2.b, out, binaryAdd),
	)
}

// recordRWKVBlock records ONE RWKV-4 block for a single decode token (rows==1), updating dx in place
// through both residuals. It mirrors nn.RWKVBlock.Step: a WKV TIME-MIX sublayer (LN1 → token-shift →
// r/k/v projections → the stabilized WKV recurrence via cu_wkv_step → receptance gate σ(r)⊙wkv →
// output projection → residual) and a gated squared-ReLU CHANNEL-MIX (LN2 → token-shift → σ(CWr·)
// gate over relu(CWk·)²·CWv → residual). No attention, no KV cache — the per-block token-shift and
// WKV states (rwPrevTM/rwPrevCM/rwAA/rwBB/rwPP) carry the whole history in O(1). Time-mix uses
// wq/wk/wv/wo as Wr/Wk/Wv/Wo; channel-mix uses wG/wD as CWk/CWv.
func (d *Decoder) recordRWKVBlock(r recorder, b block) error {
	// TIME-MIX: xn = LN1(dx); r=σ(Wr·mix_r), k=Wk·mix_k, v=Wv·mix_v.
	e := r.LayerNorm(d.dx.b, b.gAttn, b.bAttn, d.xn.b, 1, d.d, d.eps)
	e = firstErr(e,
		d.rwMixInto(r, d.xn.b, b.rwPrevTM, b.rwMuR, b.rwOmR, d.rwMix.b),
		b.wq.record(r, d.rwMix.b, d.rwR.b, 1),
		r.Unary(d.rwR.b, d.rwR.b, unarySigmoid),
		d.rwMixInto(r, d.xn.b, b.rwPrevTM, b.rwMuK, b.rwOmK, d.rwMix.b),
		b.wk.record(r, d.rwMix.b, d.rwK.b, 1),
		d.rwMixInto(r, d.xn.b, b.rwPrevTM, b.rwMuV, b.rwOmV, d.rwMix.b),
		b.wv.record(r, d.rwMix.b, d.rwV.b, 1),
		// WKV recurrence (state carried), then receptance gate and output projection onto the residual.
		r.WKVStep(d.rwK.b, d.rwV.b, b.rwW, b.rwU, b.rwAA, b.rwBB, b.rwPP, d.rwWkv.b, d.d),
		r.Binary(d.rwWkv.b, d.rwR.b, d.rwWkv.b, binaryMul), // σ(r)⊙wkv (o==a: Binary clobbers b, not a)
		b.wo.record(r, d.rwWkv.b, d.rwO.b, 1),
		r.Binary(d.dx.b, d.rwO.b, d.dx.b, binaryAdd), // residual
		r.Blit(d.xn.b, 0, b.rwPrevTM, 0, d.d),        // prevTM ← xn (token-shift state)
	)
	// CHANNEL-MIX: yn = LN2(dx); cr=σ(CWr·mix_r); ck=relu(CWk·mix_k)²; cv=CWv·ck; dx += cr⊙cv.
	e = firstErr(e,
		r.LayerNorm(d.dx.b, b.gFFN, b.bFFN, d.xn.b, 1, d.d, d.eps),
		d.rwMixInto(r, d.xn.b, b.rwPrevCM, b.rwCMuR, b.rwOmCR, d.rwMix.b),
		b.rwCWr.record(r, d.rwMix.b, d.rwCr.b, 1),
		r.Unary(d.rwCr.b, d.rwCr.b, unarySigmoid),
		d.rwMixInto(r, d.xn.b, b.rwPrevCM, b.rwCMuK, b.rwOmCK, d.rwMix.b),
		b.wG.record(r, d.rwMix.b, d.rwCk.b, 1),            // CWk: Dim→Hidden
		r.Unary(d.rwCk.b, d.rwCk.b, unaryReLU2),           // relu(·)²
		b.wD.record(r, d.rwCk.b, d.rwCv.b, 1),             // CWv: Hidden→Dim
		r.Binary(d.rwCv.b, d.rwCr.b, d.rwCv.b, binaryMul), // σ(CWr)⊙cv (o==a: Binary clobbers b, not a)
		r.Binary(d.dx.b, d.rwCv.b, d.dx.b, binaryAdd),     // residual
		r.Blit(d.xn.b, 0, b.rwPrevCM, 0, d.d),             // prevCM ← yn
	)
	return e
}

// resetRWKVState zeroes every block's token-shift + WKV state for a fresh sequence (pos==0): prevTM/
// prevCM/aa/bb start at 0 and pp at −1e38 (the pre-first-token WKV running-max, matching OpWKV).
func (d *Decoder) resetRWKVState() error {
	zero := make([]float32, d.d)
	ninf := make([]float32, d.d)
	for i := range ninf {
		ninf[i] = -1e38
	}
	for _, b := range d.blocks {
		for _, st := range []buffer{b.rwPrevTM, b.rwPrevCM, b.rwAA, b.rwBB} {
			if e := st.UploadF32(zero); e != nil {
				return e
			}
		}
		if e := b.rwPP.UploadF32(ninf); e != nil {
			return e
		}
	}
	return nil
}

// allocRWKVScratch allocates the rows==1 RWKV decode scratch (no attention/KV buffers).
func (d *Decoder) allocRWKVScratch(mk func(data []float32) *bufSlot) {
	d.dx = mk(make([]float32, d.d))
	d.xn = mk(make([]float32, d.d))
	d.logits = mk(make([]float32, d.v))
	d.rwMix = mk(make([]float32, d.d))
	d.rwR = mk(make([]float32, d.d))
	d.rwK = mk(make([]float32, d.d))
	d.rwV = mk(make([]float32, d.d))
	d.rwWkv = mk(make([]float32, d.d))
	d.rwO = mk(make([]float32, d.d))
	d.rwT1 = mk(make([]float32, d.d))
	d.rwT2 = mk(make([]float32, d.d))
	d.rwCr = mk(make([]float32, d.d))
	d.rwCk = mk(make([]float32, d.rwHidden))
	d.rwCv = mk(make([]float32, d.d))
}

func (d *Decoder) recordFFN(r recorder, b block, in buffer, rows int) error {
	if d.ffnGELU || d.ffnReLU2 {
		act := unaryGELU // 2-layer MLP activation: GELU (GPT-NeoX/Phi/StarCoder2) or squared ReLU (Nemotron)
		if d.ffnReLU2 {
			act = unaryReLU2
		}
		e := b.wG.record(r, in, d.gate.b, rows) // fc = in·Wfc
		if b.fcBias != nil {
			e = firstErr(e, r.AddBias(d.gate.b, b.fcBias, d.gate.b, rows, d.hidden))
		}
		e = firstErr(e,
			r.Unary(d.gate.b, d.gate.b, act), // act(fc)
			d.recordDownProj(r, b, rows),     // dx += act(fc)·Wproj
		)
		if b.projBias != nil {
			e = firstErr(e, r.AddBias(d.dx.b, b.projBias, d.dx.b, rows, d.d))
		}
		return e
	}
	if d.ffnGEGLU {
		// GeGLU (Gemma): down(GELU(gate)⊙up). Same 3-matrix gated shape as SwiGLU, but the gate
		// activation is GELU and the elementwise product is a separate mul (no fused silu·b op).
		return firstErr(
			b.wG.record(r, in, d.gate.b, rows),              // gate = in·Wgate
			b.wU.record(r, in, d.up.b, rows),                // up = in·Wup
			r.Unary(d.gate.b, d.gate.b, unaryGELU),          // GELU(gate)
			r.Binary(d.gate.b, d.up.b, d.gate.b, binaryMul), // GELU(gate)⊙up
			d.recordDownProj(r, b, rows),                    // dx += (·)·Wdown
		)
	}
	return firstErr(
		b.wG.record(r, in, d.gate.b, rows),
		b.wU.record(r, in, d.up.b, rows),
		r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU),
		d.recordDownProj(r, b, rows), // dx += swiglu·Wdown
	)
}

// norm records the pre-sublayer normalization: LayerNorm-with-bias when lnBias (StableLM/Phi),
// else RMSNorm (Llama). gamma is the weight; beta is the LayerNorm bias (ignored under RMSNorm).
func (d *Decoder) norm(r recorder, x, gamma, beta, o buffer, rows int) error {
	if d.lnBias {
		return r.LayerNorm(x, gamma, beta, o, rows, d.d, d.eps)
	}
	return r.RMSNorm(x, gamma, o, rows, d.d, d.eps)
}

// bFinalBeta is the final-norm LayerNorm bias, or nil under RMSNorm (bFinal unset).
func (d *Decoder) bFinalBeta() buffer {
	if d.bFinal != nil {
		return d.bFinal.b
	}
	return nil
}

func (d *Decoder) ropeQK(r recorder, qkv, inv buffer, seq, stride, pos int) error {
	if d.noRope { // MPT: position enters through the ALiBi attention bias, not rotary
		return nil
	}
	if d.rotaryDim < d.dk {
		return r.RoPEPartialPair(qkv, inv, seq, stride, d.h, 0, d.kvH, d.d, d.dk, d.rotaryDim, pos, d.posDiv)
	}
	return r.RoPEPair(qkv, inv, seq, stride, d.h, 0, d.kvH, d.d, d.dk, d.half, pos, d.posDiv)
}

// allocScratch uploads the RoPE frequencies and allocates the per-step scratch buffers. They are
// sized for up to maxLen rows so StepN can process a whole prompt (or a speculative draft window)
// in one recorded step (§T418) — a few MB at typical configs.
func (d *Decoder) allocScratch(mk func(data []float32) *bufSlot) {
	c := d.maxLen
	d.dinv = mk(d.invHost)
	d.dx = mk(make([]float32, c*d.d))
	d.xn = mk(make([]float32, c*d.d))
	d.xn2 = mk(make([]float32, c*d.d))
	d.q = mk(make([]float32, c*d.qDim))
	d.k = mk(make([]float32, c*d.kvDim))
	d.v_ = mk(make([]float32, c*d.kvDim))
	d.qkv = mk(make([]float32, c*(d.qDim+2*d.kvDim)))
	d.attn = mk(make([]float32, c*d.qDim))
	d.ao = mk(make([]float32, c*d.d))
	d.gate = mk(make([]float32, c*d.hidden))
	d.up = mk(make([]float32, c*d.hidden))
	d.mo = mk(make([]float32, c*d.d))
	d.logits = mk(make([]float32, c*d.v))
	if d.moe {
		d.moeGate = mk(make([]float32, c*d.nExperts))
		d.moeW = mk(make([]float32, c*d.nExperts))
		d.moeCol = mk(make([]float32, c))
	}
}

// mkBuf returns a bufSlot allocator that records the first error and tracks buffers for Release.
func (d *Decoder) mkBuf(err *error) func(data []float32) *bufSlot {
	return func(data []float32) *bufSlot {
		if *err != nil {
			return &bufSlot{}
		}
		b, e := d.ops.newBuffer(data)
		if e != nil {
			*err = e
			return &bufSlot{}
		}
		d.all = append(d.all, b)
		return &bufSlot{b: b}
	}
}

// newDecoder uploads m's f32 weights via ops into device buffers and prepares a KV cache up to Ctx.
func newDecoder(m *nlp.Llama, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	d, derr := newDecoderCommon(cfg, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear { // f32 [in,out] device weight
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	// IBM Granite config scalars (all identity when 0 or 1, so no-ops for Llama/Qwen/Mistral). Each
	// folds into the upload rather than a runtime op: AttentionMult overrides the pre-softmax scale,
	// EmbeddingMult scales gathered embeddings (d.embMult), ResidualMult scales each sublayer output
	// (baked into Wo and Wdown, since their matmul IS the residual-add epilogue — Granite carries no
	// projection bias, so nothing else in the add needs scaling), and LogitsScale divides the logits
	// (baked as 1/LogitsScale into the untied lm_head). Separate uploads, so tied embed/lm_head is fine.
	if cfg.AttentionMult != 0 {
		d.scale = float32(cfg.AttentionMult)
	}
	if cfg.EmbeddingMult != 0 && cfg.EmbeddingMult != 1 {
		d.embMult = float32(cfg.EmbeddingMult)
	}
	resMult := float32(1)
	if cfg.ResidualMult != 0 && cfg.ResidualMult != 1 {
		resMult = float32(cfg.ResidualMult)
	}
	outScale := float32(1)
	if cfg.LogitsScale != 0 && cfg.LogitsScale != 1 {
		outScale = float32(1 / cfg.LogitsScale)
	}
	linS := func(w *tensor.Tensor, s float32) linear { // f32 weight with every element pre-scaled by s
		in, out := w.Shape()[0], w.Shape()[1]
		f := flat2D(w)
		if s != 1 {
			for i := range f {
				f[i] *= s
			}
		}
		return f32Linear{w: mk(f).b, k: in, n: out}
	}
	// fused QKV weight (§T613): weights are [in,out] with out along the row, so the fusion
	// concatenates the three output bands PER INPUT ROW — one [in, D+2·kvDim] matrix whose
	// product row is [q | k | v]. The separate wq/wk/wv uploads are dropped (no dup storage).
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	// fused q/k/v bias [Bq | Bk | Bv] = [D+2·kvDim], for the Qwen2/Qwen2.5 family (q/k/v carry a
	// projection bias, o_proj does not); nil for Llama/Mistral, which leaves recordQKVProj's
	// AddBias unrecorded so those models stay byte-identical.
	fusedBias := func(bq, bk, bv *tensor.Tensor) buffer {
		if bq == nil && bk == nil && bv == nil {
			return nil
		}
		fb := append(append(append([]float32{}, flat1D(bq)...), flat1D(bk)...), flat1D(bv)...)
		return mk(fb).b
	}
	// per-head query/key RMSNorm gain [dk] (Qwen3); nil for Llama/Qwen2. Presence flips d.qkNorm,
	// which arms recordQKNorm between the QKV projection and RoPE.
	qkGain := func(n *nn.RMSNorm) buffer {
		if n == nil {
			return nil
		}
		d.qkNorm = true
		return mk(flat1D(n.Gamma)).b
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), qkvBias: fusedBias(b.Bq, b.Bk, b.Bv), wo: linS(b.Wo, resMult),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			qN: qkGain(b.QNorm), kN: qkGain(b.KNorm),
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: linS(b.FFN.Wdown, resMult),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = linS(m.Out, outScale) // LogitsScale folded into the lm_head (1/LogitsScale)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newStableLMDecoder builds a device Decoder for an nlp.StableLM: the Llama-shaped SwiGLU/GQA
// core, but with LayerNorm-with-bias (lnBias) and PARTIAL rotary (rotaryDim < dk). StableLM has
// no attention bias and its rotary is split-half (matches OpRoPE), so weights upload directly —
// only the norms carry a bias and the rope is partial. Currently cuda-only (metal/vulkan
// RoPEPartialPair are unimplemented). §T757-class new-architecture GPU decode.
func newStableLMDecoder(m *nlp.StableLM, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	// StableLMConfig's kvHeads()/headDim()/rotaryDim() are unexported, so replicate here.
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = cfg.Dim / cfg.Heads
	}
	rotaryDim := cfg.RotaryDim
	if rotaryDim <= 0 {
		pct := cfg.RotaryPct
		if pct == 0 {
			pct = 0.25
		}
		rotaryDim = int(float64(headDim) * pct)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	// LayerNorm-bias + partial rotary: rebuild the inverse-frequency table over rotaryDim
	// (newDecoderCommon sized it to dk for full rope).
	d.lnBias = true
	d.rotaryDim = rotaryDim
	if d.rotaryDim <= 0 || d.rotaryDim > d.dk || d.rotaryDim%2 != 0 {
		return nil, fmt.Errorf("llamagpu(%s): StableLM rotaryDim %d invalid for headDim %d", ops.name, d.rotaryDim, d.dk)
	}
	invF64, posDiv64 := backend.RoPEFreqs(d.rotaryDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.rotaryDim/2)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			gFFN: mk(flat1D(b.PostAttnNorm.Gamma)).b, bFFN: mk(flat1D(b.PostAttnNorm.Beta)).b,
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: lin(b.FFN.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.bFinal = mk(flat1D(m.Norm.Beta))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newCohereDecoder builds a device Decoder for an nlp.Cohere (Command-R). Cohere composes existing
// generalizations: ONE-norm parallel residual (parallelRes — input_layernorm feeds both attention
// and FFN, summed onto x0, exactly like Phi), weight-only mean-centered LayerNorm (lnBias with a
// zero β), SwiGLU, GQA and FULL rope. Its interleaved (GPT-J) rotary is already baked into a q/k
// weight-row permutation by CohereFromHF, so standard split-half RoPE reproduces it. The only extra
// is logit_scale: the tied lm_head's logits are multiplied by a constant, folded into the Out
// upload. cuda-only (parallel-residual + LayerNorm reuse the Phi plumbing).
func newCohereDecoder(m *nlp.Cohere, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	if cfg.HeadDim != 0 && cfg.HeadDim != cfg.Dim/cfg.Heads {
		return nil, fmt.Errorf("llamagpu(%s): Cohere decoupled head_dim %d (≠ dim/heads %d) not supported on the GPU decoder yet", ops.name, cfg.HeadDim, cfg.Dim/cfg.Heads)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true      // weight-only mean-centered LayerNorm (β is a zero vector)
	d.parallelRes = true // one-norm parallel residual: input_layernorm feeds attn AND FFN

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	logitScale := float32(1)
	if cfg.LogitScale != 0 && cfg.LogitScale != 1 {
		logitScale = float32(cfg.LogitScale)
	}
	linS := func(w *tensor.Tensor, s float32) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		f := flat2D(w)
		if s != 1 {
			for i := range f {
				f[i] *= s
			}
		}
		return f32Linear{w: mk(f).b, k: in, n: out}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			// gFFN/bFFN unused: parallel one-norm reuses the attn norm (d.xn) for the FFN.
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: lin(b.FFN.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.bFinal = mk(flat1D(m.Norm.Beta))
	d.out = linS(m.Out, logitScale) // logit_scale folded into the tied lm_head
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newNemotronDecoder builds a device Decoder for an nlp.Nemotron. Nemotron is a SEQUENTIAL
// two-norm decoder (input_layernorm feeds attention, post_attention_layernorm feeds the MLP, each
// on its own residual add) with three departures from Llama, all reusing existing plumbing:
// LayerNorm1P (an ordinary mean-centered LayerNorm — lnBias — whose γ=w+1 and β are folded in by
// NemotronFromHF), a squared-ReLU 2-layer MLP (ffnReLU2: down(relu²(up(x))), no gate, no bias), and
// PARTIAL rotary. cuda-only (partial rotary + the relu² unary are cuda-only).
func newNemotronDecoder(m *nlp.Nemotron, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = cfg.Dim / cfg.Heads
	}
	if headDim != cfg.Dim/cfg.Heads {
		return nil, fmt.Errorf("llamagpu(%s): Nemotron decoupled head_dim %d (≠ dim/heads %d) not supported on the GPU decoder yet", ops.name, headDim, cfg.Dim/cfg.Heads)
	}
	rotaryDim := cfg.RotaryDim
	if rotaryDim <= 0 {
		pct := cfg.RotaryPct
		if pct == 0 {
			pct = 0.5
		}
		rotaryDim = int(float64(headDim) * pct)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true   // LayerNorm1P = mean-centered LayerNorm with a bias
	d.ffnReLU2 = true // squared-ReLU 2-layer MLP
	d.rotaryDim = rotaryDim
	if d.rotaryDim <= 0 || d.rotaryDim > d.dk || d.rotaryDim%2 != 0 {
		return nil, fmt.Errorf("llamagpu(%s): Nemotron rotaryDim %d invalid for headDim %d", ops.name, d.rotaryDim, d.dk)
	}
	invF64, posDiv64 := backend.RoPEFreqs(d.rotaryDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.rotaryDim/2)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			gFFN: mk(flat1D(b.PostAttnNorm.Gamma)).b, bFFN: mk(flat1D(b.PostAttnNorm.Beta)).b,
			wG: lin(b.Wup), wD: lin(b.Wdown), // 2-layer relu² MLP: wG = up (fc), wD = down (proj); no gate
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.bFinal = mk(flat1D(m.Norm.Beta))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newGemmaDecoder builds a device Decoder for an nlp.Gemma (Gemma v1). Gemma is close to Llama —
// pre-norm RMSNorm, RoPE, GQA — with three departures, all reusing existing plumbing: the token
// embeddings are scaled by √dim right after lookup (d.embMult), RMSNorm uses (1+w) as the gain
// (folded in at load, so nn.RMSNorm applies directly), and the FFN is GeGLU rather than SwiGLU
// (ffnGEGLU — GELU gate + a plain elementwise product). The lm_head is tied to the embedding
// (Out = TokEmbᵀ). cuda-only for consistency with the rest of the new-arch family.
func newGemmaDecoder(m *nlp.Gemma, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = cfg.Dim / cfg.Heads
	}
	// Gemma infers head_dim from q_proj (heads·head_dim rows) and it is DECOUPLED (head_dim need not
	// equal dim/heads), so read it back from the loaded weights rather than trusting the config —
	// GemmaFromHF may have left cfg.HeadDim at 0.
	if qOut := m.Blocks[0].Wq.Shape()[1]; qOut%cfg.Heads == 0 {
		headDim = qOut / cfg.Heads
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.FFN, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.ffnGEGLU = true                                // GELU-gated FFN
	d.embMult = float32(math.Sqrt(float64(cfg.Dim))) // Gemma's √dim embedding normalizer
	// Decoupled head_dim: override the geometry newDecoderCommon derived as dim/heads. qDim (the
	// attention Q/O width) becomes h·head_dim, which for Gemma differs from the model dim d.
	if headDim != d.dk {
		d.dk = headDim
		d.qDim = d.h * headDim
		d.kvDim = d.kvH * headDim
		d.half = headDim / 2
		d.rotaryDim = headDim
		d.scale = float32(1.0 / math.Sqrt(float64(headDim)))
		invF64, posDiv64 := backend.RoPEFreqs(headDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
		d.invHost = make([]float32, headDim/2)
		for i := range d.invHost {
			d.invHost[i] = float32(invF64[i])
		}
		d.posDiv = float32(posDiv64)
	}

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: lin(b.FFN.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	// Tied lm_head: Out = TokEmbᵀ. TokEmb is [vocab, dim]; the projection wants [dim, vocab].
	vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
	outW := make([]float32, dim*vocab)
	for j := range vocab {
		for i := range dim {
			outW[i*vocab+j] = float32(m.TokEmb.AtF64(j, i))
		}
	}
	d.out = f32Linear{w: mk(outW).b, k: dim, n: vocab}
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newDeepSeekV2Decoder builds a device Decoder for an nlp.DeepSeekV2 — a stack of Multi-head Latent
// Attention blocks, each with a dense SwiGLU or a DeepSeekMoE sparse FFN (first_k_dense_replace
// split). MLA is the
// only new attention machinery (recordMLAAttention + rectangular MHARect); the FFN is a per-block
// dense SwiGLU or the DeepSeekMoE sparse FFN (raw-softmax·routedScale top-k routed experts + an
// ungated shared expert), dispatched by b.moeRouter. cuda-only (MHARect is cuda-only).
func newDeepSeekV2Decoder(m *nlp.DeepSeekV2, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): DeepSeek-V2 has no blocks", ops.name)
	}
	qkHead := cfg.QKNope + cfg.QKRope
	kvHead := cfg.QKNope + cfg.VHead
	// hidden must cover the widest SwiGLU across every block (dense, routed-expert, shared) since the
	// gate/up scratch is shared; nExperts/topK/routedScale come from the first MoE block if any.
	hidden, nExperts, topK, hasMoE := 0, 0, 0, false
	wider := func(sw *nn.SwiGLU) {
		if sw != nil {
			if h := sw.Wgate.Shape()[1]; h > hidden {
				hidden = h
			}
		}
	}
	for _, b := range m.Blocks {
		wider(b.Dense)
		if b.MoE != nil {
			hasMoE = true
			for _, e := range b.MoE.Routed.Experts {
				wider(e)
			}
			for _, s := range b.MoE.Shared {
				wider(s)
			}
			if nExperts == 0 {
				nExperts, topK = len(b.MoE.Routed.Experts), b.MoE.Routed.TopK
			}
		}
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: cfg.Heads,
		Layers: cfg.Layers, Hidden: hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.mla = true
	if hasMoE {
		d.moe, d.moeRaw = true, true // DeepSeek-V2 raw-softmax·routedScale gating
		d.nExperts, d.topK = nExperts, topK
		d.routedScale = 1
		if cfg.RoutedScale > 0 {
			d.routedScale = float32(cfg.RoutedScale)
		}
	}
	d.qkNope, d.qkRope, d.vHead, d.qkHead = cfg.QKNope, cfg.QKRope, cfg.VHead, qkHead
	d.qLoRA, d.kvLoRA = cfg.QLoraRank, cfg.KVLoraRank
	scale := cfg.SoftmaxScale
	if scale <= 0 {
		scale = 1.0 / math.Sqrt(float64(qkHead))
	}
	d.mlaScale = float32(scale)
	// decoupled RoPE frequencies over QKRope (single-head convention, Heads:1)
	invF64, posDiv64 := backend.RoPEFreqs(cfg.QKRope, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: 1})
	d.mlaPosDiv = float32(posDiv64)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	// MLA scratch + the QKRope frequency table.
	c := d.maxLen
	mlaInv := make([]float32, cfg.QKRope/2)
	for i := range mlaInv {
		mlaInv[i] = float32(invF64[i])
	}
	d.mlaInv = mk(mlaInv)
	d.mlaCQ = mk(make([]float32, c*cfg.QLoraRank))
	d.mlaQ = mk(make([]float32, c*cfg.Heads*qkHead))
	d.mlaComp = mk(make([]float32, c*(cfg.KVLoraRank+cfg.QKRope)))
	d.mlaLatent = mk(make([]float32, c*cfg.KVLoraRank))
	d.mlaKV = mk(make([]float32, c*cfg.Heads*kvHead))
	d.mlaAttn = mk(make([]float32, c*cfg.Heads*cfg.VHead))

	for _, b := range m.Blocks {
		gb := block{
			wqA: lin(b.WqA), wqB: lin(b.WqB), wkvA: lin(b.WkvA), wkvB: lin(b.WkvB), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, gFFN: mk(flat1D(b.PostAttnNorm.Gamma)).b,
			gQA: mk(flat1D(b.QANorm.Gamma)).b, gKvA: mk(flat1D(b.KvANorm.Gamma)).b,
			// MLA cache: K is heads·qkHead wide, V is heads·VHead wide (rectangular).
			kC: mk(make([]float32, c*cfg.Heads*qkHead)).b, vC: mk(make([]float32, c*cfg.Heads*cfg.VHead)).b,
		}
		if b.Dense != nil { // dense SwiGLU FFN
			gb.wG, gb.wU, gb.wD = lin(b.Dense.Wgate), lin(b.Dense.Wup), lin(b.Dense.Wdown)
		} else { // DeepSeekMoE: routed top-k experts + ungated shared expert(s)
			experts := make([]moeFFN, len(b.MoE.Routed.Experts))
			for i, ex := range b.MoE.Routed.Experts {
				experts[i] = moeFFN{wG: lin(ex.Wgate), wU: lin(ex.Wup), wD: lin(ex.Wdown)}
			}
			gb.moeRouter, gb.moeExperts = lin(b.MoE.Routed.Router.W), experts
			if len(b.MoE.Shared) > 0 { // realized as a single fused SwiGLU; ungated (moeSharedGate nil)
				s := b.MoE.Shared[0]
				gb.moeShared = moeFFN{wG: lin(s.Wgate), wU: lin(s.Wup), wD: lin(s.Wdown)}
			}
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	d.out = lin(m.LmHead)
	d.allocScratch(mk) // standard scratch (dx/xn/xn2/gate/up/mo/logits reused; Llama-shaped q/qkv unused)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newMambaDecoder builds a device Decoder for an nlp.Mamba — the FIRST non-transformer GPU decoder.
// Each layer is a pre-RMSNorm + selective-state-space mixer (nn.MambaBlock): no attention, no KV
// cache, no FFN. Decode is a linear-time recurrence, so the block records the per-timestep conv/SSM
// decode kernels (Conv1DStep/SSMStep, cuda-only) carrying per-block state across Step calls. Only
// DtProj and the conv carry a bias (the HF checkpoint is otherwise bias-free); A = −exp(A_log) is
// precomputed on the host. The LM head is tied (logits = hidden · Embedᵀ). cuda-only.
func newMambaDecoder(m *nlp.Mamba, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	if len(m.Layers) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): Mamba has no layers", ops.name)
	}
	mix0 := m.Layers[0].Mixer
	// Mamba has no attention: Heads=1 satisfies newDecoderCommon's geometry (all attention/KV scratch
	// is skipped by allocMambaScratch). Hidden is unused (no FFN). Ctx is a pos bound only — the
	// recurrence has no cache, so allocMambaScratch never scales with it; pick a generous limit.
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: 1 << 20, Dim: cfg.DModel, Heads: 1, KVHeads: 1,
		Layers: cfg.Layers, Hidden: 1, Eps: cfg.Eps,
	}
	d, derr := newDecoderCommon(lc, m.Embed, ops)
	if derr != nil {
		return nil, derr
	}
	d.mamba = true
	d.dInner, d.mambaN, d.dConv, d.dtRank = mix0.DInner, mix0.N, mix0.DConv, mix0.DtRank

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	for _, ly := range m.Layers {
		mx := ly.Mixer
		// A = −exp(A_log), precomputed on the host (ALog is [dInner, N], row-major — the SSMStep layout).
		alog := flat2D(mx.ALog)
		A := make([]float32, len(alog))
		for i, v := range alog {
			A[i] = float32(-math.Exp(float64(v)))
		}
		gb := block{
			gAttn: mk(flat1D(ly.Norm.Gamma)).b, // pre-mixer RMSNorm γ (stored in gAttn; recordMambaBlock norms dx→xn)
			mInX:  lin(mx.InX.W), mInZ: lin(mx.InZ.W),
			mDtLow: lin(mx.DtLow.W), mDtProj: lin(mx.DtProj.W),
			mBProj: lin(mx.BProj.W), mCProj: lin(mx.CProj.W),
			mOutProj: lin(mx.OutProj.W),
			mConvW:   mk(flat2D(mx.ConvW)).b, mConvB: mk(flat1D(mx.ConvB)).b,
			mA: mk(A).b, mDskip: mk(flat1D(mx.Dskip)).b, mDtBias: mk(flat1D(mx.DtProj.B)).b,
			mConvState: mk(make([]float32, d.dInner*(d.dConv-1))).b,
			mSsmState:  mk(make([]float32, d.dInner*d.mambaN)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = lin(m.Head) // tied head [d_model, vocab]: logits = hidden · Embedᵀ
	d.allocMambaScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newMamba2Decoder builds a device Decoder for an nlp.Mamba2 — the state-space-duality sibling of
// Mamba. Each layer is a pre-RMSNorm + SSD mixer: fused in_proj → [z | xBC | dt] → causal-conv + SiLU
// → [x | B | C] → Δ=softplus(dt+dt_bias) → the SSD scan (scalar per-head decay, B/C shared across a
// group, +D·x skip; via cu_ssd_step) → gated RMSNorm norm(y·SiLU(z)) → out_proj. No attention, no KV
// cache — decode is a linear-time recurrence, so it reuses the Mamba rows==1 sequential Step/StepN
// plumbing. in_proj/out_proj are torch [out,in] (transposed on upload); A=−exp(A_log); tied LM head.
func newMamba2Decoder(m *nlp.Mamba2, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	if len(m.Layers) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): Mamba2 has no layers", ops.name)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: 1 << 20, Dim: cfg.DModel, Heads: 1, KVHeads: 1,
		Layers: cfg.Layers, Hidden: 1, Eps: cfg.Eps,
	}
	d, derr := newDecoderCommon(lc, m.Embed, ops)
	if derr != nil {
		return nil, derr
	}
	d.mamba2 = true
	d.m2H, d.m2P, d.m2G, d.m2N = cfg.NumHeads, cfg.HeadDim, cfg.NGroups, cfg.N
	d.m2Conv, d.m2Inter, d.m2CD = cfg.DConv, cfg.Intermediate, cfg.Intermediate+2*cfg.NGroups*cfg.N

	var err error
	mk := d.mkBuf(&err)
	// linT uploads a torch [out,in] weight transposed to the [in,out] the recorder matmul expects.
	linT := func(w *tensor.Tensor) linear {
		out, in := w.Shape()[0], w.Shape()[1]
		f := make([]float32, in*out)
		for o := 0; o < out; o++ {
			for i := 0; i < in; i++ {
				f[i*out+o] = float32(w.AtF64(o, i))
			}
		}
		return f32Linear{w: mk(f).b, k: in, n: out}
	}
	for _, ly := range m.Layers {
		mx := ly.Mixer
		aLog := flat1D(mx.ALog)
		A := make([]float32, len(aLog))
		for i, v := range aLog {
			A[i] = float32(-math.Exp(float64(v))) // A = −exp(A_log), per head
		}
		gb := block{
			gAttn:    mk(flat1D(ly.Norm.Gamma)).b, // pre-mixer RMSNorm γ
			m2InProj: linT(mx.InProj), m2OutProj: linT(mx.OutProj),
			m2ConvW: mk(flat2D(mx.ConvW)).b, m2ConvB: mk(flat1D(mx.ConvB)).b,
			m2A: mk(A).b, m2DtBias: mk(flat1D(mx.DtBias)).b, m2Dskip: mk(flat1D(mx.D)).b,
			m2NormW:     mk(flat1D(mx.NormW)).b,
			m2ConvState: mk(make([]float32, d.m2CD*(d.m2Conv-1))).b,
			m2SsdState:  mk(make([]float32, d.m2H*d.m2N*d.m2P)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	// tied head: m.Head is already [d_model, vocab] = Embedᵀ, my [in=d_model, out=vocab] orientation.
	d.out = f32Linear{w: mk(flat2D(m.Head)).b, k: cfg.DModel, n: cfg.Vocab}
	d.allocMamba2Scratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newRWKVDecoder builds a device Decoder for an nlp.RWKV — GoAI's third recurrent family (after the
// Mamba SSMs). Each layer is an nn.RWKVBlock: a WKV time-mixing sublayer (token-shift → r/k/v → the
// stabilized WKV recurrence via cu_wkv_step → receptance gate → output proj) and a gated squared-ReLU
// channel-mixing sublayer, each with its own LayerNorm + residual. No attention, no KV cache, no
// positional encoding — decode is O(1) recurrent, reusing the rows==1 sequential Step/StepN plumbing.
// An extra ln0 LayerNorm transforms the embedding before block 0; the head is UNTIED. cuda-only.
func newRWKVDecoder(m *nlp.RWKV, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): RWKV has no blocks", ops.name)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: 1 << 20, Dim: cfg.Dim, Heads: 1, KVHeads: 1,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps,
	}
	d, derr := newDecoderCommon(lc, m.Embed, ops)
	if derr != nil {
		return nil, derr
	}
	d.rwkv = true
	d.lnBias = true // final norm (ln_out) is a full LayerNorm
	d.rwHidden = cfg.Hidden

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear { // nn.Linear.W is already [in,out] (transposed at HF load)
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	// mu uploads the interpolator and its 1−μ complement (token-shift compose needs both).
	comp := func(mu *tensor.Tensor) buffer {
		v := flat1D(mu)
		o := make([]float32, len(v))
		for i, x := range v {
			o[i] = 1 - x
		}
		return mk(o).b
	}
	expv := func(t *tensor.Tensor) buffer { // per-channel decay w = exp(WLog)
		v := flat1D(t)
		o := make([]float32, len(v))
		for i, x := range v {
			o[i] = float32(math.Exp(float64(x)))
		}
		return mk(o).b
	}
	zeros := func() buffer { return mk(make([]float32, d.d)).b }
	for _, bl := range m.Blocks {
		gb := block{
			gAttn: mk(flat1D(bl.LN1.Gamma)).b, bAttn: mk(flat1D(bl.LN1.Beta)).b, // LN1
			gFFN: mk(flat1D(bl.LN2.Gamma)).b, bFFN: mk(flat1D(bl.LN2.Beta)).b, // LN2
			wq: lin(bl.Wr.W), wk: lin(bl.Wk.W), wv: lin(bl.Wv.W), wo: lin(bl.Wo.W), // time-mix Wr/Wk/Wv/Wo
			wG: lin(bl.CWk.W), wD: lin(bl.CWv.W), rwCWr: lin(bl.CWr.W), // channel-mix CWk/CWv/CWr
			rwMuR: mk(flat1D(bl.MuR)).b, rwMuK: mk(flat1D(bl.MuK)).b, rwMuV: mk(flat1D(bl.MuV)).b,
			rwCMuR: mk(flat1D(bl.CMuR)).b, rwCMuK: mk(flat1D(bl.CMuK)).b,
			rwOmR: comp(bl.MuR), rwOmK: comp(bl.MuK), rwOmV: comp(bl.MuV),
			rwOmCR: comp(bl.CMuR), rwOmCK: comp(bl.CMuK),
			rwW: expv(bl.WLog), rwU: mk(flat1D(bl.U)).b,
			rwPrevTM: zeros(), rwPrevCM: zeros(), rwAA: zeros(), rwBB: zeros(), rwPP: zeros(),
		}
		d.blocks = append(d.blocks, gb)
	}
	d.rwPreLNg = mk(flat1D(m.PreLN.Gamma)) // ln0 (before block 0)
	d.rwPreLNb = mk(flat1D(m.PreLN.Beta))
	d.gFinal = mk(flat1D(m.LNOut.Gamma)) // final LayerNorm ln_out
	d.bFinal = mk(flat1D(m.LNOut.Beta))
	d.out = lin(m.Head) // untied head [dim, vocab]
	d.allocRWKVScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newFalconDecoder builds a device Decoder for an nlp.Falcon (falcon-7b class: parallel_attn=True,
// multi_query=True, new_decoder_architecture=False). Every departure is an existing generalization:
// SINGLE-NORM parallel residual (parallelRes — one input_layernorm feeds attention AND the MLP, both
// summed onto x0, like Cohere), full LayerNorm WITH bias (lnBias), a 2-layer GELU MLP (ffnGELU), and
// multi-query attention (MQA = GQA with one KV head, kvH=1, from the fused query_key_value split).
// Full split-half rope, no linear biases, untied lm_head. cuda-only (parallel residual + LayerNorm).
func newFalconDecoder(m *nlp.Falcon, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = 1 // Falcon MQA
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true      // LayerNorm with bias
	d.ffnGELU = true     // 2-layer GELU MLP
	d.parallelRes = true // one-norm parallel residual: input_layernorm feeds attn AND MLP

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			// gFFN/bFFN unused: parallel one-norm reuses the attn norm (d.xn) for the MLP.
			wG: lin(b.Wh), wD: lin(b.Wout), // 2-layer GELU MLP: wG = dense_h_to_4h, wD = dense_4h_to_h
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	d.bFinal = mk(flat1D(m.FinalNorm.Beta))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newOLMo2Decoder builds a device Decoder for an nlp.OLMo2 (Allen AI). OLMo2 shares Llama's SwiGLU/
// GQA/RoPE core but reshuffles the residual structure in two ways, both new gated generalizations:
//   - POST-NORM (postNorm): there is no input_layernorm; each sublayer reads the RAW residual and
//     its OUTPUT is normed (post_attention_layernorm / post_feedforward_layernorm) before the add.
//   - FULL-WIDTH QK-norm (qkNorm+qkNormFull): an RMSNorm over the ENTIRE q_proj / k_proj output
//     (not per-head like Qwen3), applied before RoPE.
//
// The lm_head is untied. cuda-only.
func newOLMo2Decoder(m *nlp.OLMo2, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.postNorm = true   // sublayer output is normed before the residual add
	d.qkNorm = true     // RMSNorm on the q/k projections before RoPE...
	d.qkNormFull = true // ...over the full width, not per head
	// decoupled head_dim (read back from q_proj; harmless when head_dim == dim/heads)
	if qOut := m.Blocks[0].Wq.Shape()[1]; qOut%cfg.Heads == 0 && qOut/cfg.Heads != d.dk {
		hd := qOut / cfg.Heads
		d.dk, d.qDim, d.kvDim, d.half, d.rotaryDim = hd, d.h*hd, d.kvH*hd, hd/2, hd
		d.scale = float32(1.0 / math.Sqrt(float64(hd)))
		invF64, posDiv64 := backend.RoPEFreqs(hd, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
		d.invHost = make([]float32, hd/2)
		for i := range d.invHost {
			d.invHost[i] = float32(invF64[i])
		}
		d.posDiv = float32(posDiv64)
	}

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			// post-norm: gAttn = post_attention_layernorm (on the attn output), gFFN = post_feedforward_layernorm.
			gAttn: mk(flat1D(b.PostAttnNorm.Gamma)).b, gFFN: mk(flat1D(b.PostFFNNorm.Gamma)).b,
			qN: mk(flat1D(b.QNorm.Gamma)).b, kN: mk(flat1D(b.KNorm.Gamma)).b, // full-width [qDim]/[kvDim]
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: lin(b.FFN.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newGemma2Decoder builds a device Decoder for an nlp.Gemma2. Gemma2 extends Gemma with three more
// departures, all reusing the generalizations added for its siblings:
//   - SANDWICH norms (sandwich): each sublayer is pre-normed AND its output post-normed — four
//     (1+w) RMSNorms per block (input/post_attention, pre_feedforward/post_feedforward).
//   - attention-logit SOFT-CAP (attnCap → MHACap): scaled scores pass through cap·tanh(·/cap)
//     before the mask+softmax.
//   - query_pre_attn_scalar: the pre-softmax scale is 1/√scalar (not 1/√head_dim in general).
//
// Plus Gemma's √dim embedding normalizer, (1+w) RMSNorm, GeGLU, decoupled head_dim and tied lm_head.
// The final-logit soft-cap is a MONOTONIC map, so it never changes the greedy argmax and is omitted
// (greedy generation is identical with or without it). cuda-only (the soft-cap kernel is cuda-only).
func newGemma2Decoder(m *nlp.Gemma2, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = cfg.Dim / cfg.Heads
	}
	if qOut := m.Blocks[0].Wq.Shape()[1]; qOut%cfg.Heads == 0 {
		headDim = qOut / cfg.Heads
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.FFN, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.ffnGEGLU = true
	d.sandwich = true
	d.embMult = float32(math.Sqrt(float64(cfg.Dim)))
	if cfg.AttnLogitCap > 0 {
		d.attnCap = float32(cfg.AttnLogitCap)
	}
	// decoupled head_dim geometry override (as Gemma)
	if headDim != d.dk {
		d.dk, d.qDim, d.kvDim, d.half, d.rotaryDim = headDim, d.h*headDim, d.kvH*headDim, headDim/2, headDim
		invF64, posDiv64 := backend.RoPEFreqs(headDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
		d.invHost = make([]float32, headDim/2)
		for i := range d.invHost {
			d.invHost[i] = float32(invF64[i])
		}
		d.posDiv = float32(posDiv64)
	}
	// query_pre_attn_scalar sets the pre-softmax scale to 1/√scalar (0 → head_dim = standard).
	qpa := cfg.QueryPreAttnScalar
	if qpa == 0 {
		qpa = float64(headDim)
	}
	d.scale = float32(1.0 / math.Sqrt(qpa))

	var err error
	mk := d.mkBuf(&err)
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, gAttn2: mk(flat1D(b.PostAttnNorm.Gamma)).b,
			gFFN: mk(flat1D(b.PreFFNNorm.Gamma)).b, gFFN2: mk(flat1D(b.PostFFNNorm.Gamma)).b,
			wG: lin(b.FFN.Wgate), wU: lin(b.FFN.Wup), wD: lin(b.FFN.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	// tied lm_head: Out = TokEmbᵀ [dim, vocab]
	vocab, dim := m.TokEmb.Shape()[0], m.TokEmb.Shape()[1]
	outW := make([]float32, dim*vocab)
	for j := range vocab {
		for i := range dim {
			outW[i*vocab+j] = float32(m.TokEmb.AtF64(j, i))
		}
	}
	d.out = f32Linear{w: mk(outW).b, k: dim, n: vocab}
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newMPTDecoder builds a device Decoder for an nlp.MPT (MosaicML). MPT is the first ALiBi decoder:
// position enters ONLY through a per-head linear bias on the attention scores (no RoPE, no positional
// embedding), so newMPTDecoder sets noRope and uploads the backend.ALiBiSlopes so recordMHA routes
// attention through MHAALiBi. Otherwise it is a sequential-residual block with weight-only LayerNorm
// (lnBias, β=0), standard MHA (no GQA), a bias-free 2-layer GELU MLP (ffnGELU) and a tied lm_head.
// cuda-only (the ALiBi attention kernel is cuda-only).
func newMPTDecoder(m *nlp.MPT, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: cfg.Heads,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: 10000,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true  // weight-only LayerNorm (β = 0)
	d.ffnGELU = true // bias-free 2-layer GELU MLP
	d.noRope = true  // ALiBi, no rotary

	var err error
	mk := d.mkBuf(&err)
	slopes := make([]float32, d.h)
	for i, s := range backend.ALiBiSlopes(d.h) {
		slopes[i] = float32(s)
	}
	d.aliBiSlopes = mk(slopes).b

	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.Norm1.Gamma)).b, bAttn: mk(flat1D(b.Norm1.Beta)).b,
			gFFN: mk(flat1D(b.Norm2.Gamma)).b, bFFN: mk(flat1D(b.Norm2.Beta)).b,
			wG: lin(b.Wup), wD: lin(b.Wdown),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	d.bFinal = mk(flat1D(m.FinalNorm.Beta))
	d.out = lin(m.Out) // tied wteᵀ, provided [dim, vocab] by the loader
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newMixtralDecoder builds a device Decoder for an nlp.Mixtral (sparse Mixture-of-Experts). The
// attention/norm stack is the plain Llama core (RMSNorm, fused QKV, full rope, GQA, and the optional
// per-head QK-norm shared with Qwen3-MoE); only the FFN sublayer becomes a sparse MoE: a router
// scores the experts, MoEGate picks the renormalized top-k weights, and every expert's SwiGLU is
// weighted-combined onto the residual (recordMoE). Dense expert eval — correct but not yet the sparse
// top-k gather. cuda-only (the MoE routing/combine kernels are cuda-only).
func newMixtralDecoder(m *nlp.Mixtral, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	if len(m.Blocks) == 0 || m.Blocks[0].MoE == nil || len(m.Blocks[0].MoE.Experts) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): Mixtral has no MoE experts", ops.name)
	}
	// Derive the expert count / top-k from the loaded SparseMoE (MixtralFromHF infers Experts and
	// may not write it back into Config).
	nExperts := len(m.Blocks[0].MoE.Experts)
	topK := m.Blocks[0].MoE.TopK
	if topK <= 0 {
		topK = 2
	}
	hidden := cfg.Hidden
	if hidden <= 0 { // inferred, may not be written back to Config — read it from an expert's Wgate [dim, ffn]
		hidden = m.Blocks[0].MoE.Experts[0].Wgate.Shape()[1]
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.moe = true
	d.nExperts = nExperts
	d.topK = topK

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	qkGain := func(n *nn.RMSNorm) buffer { // optional per-head QK-norm (Qwen3-MoE); nil for Mixtral
		if n == nil {
			return nil
		}
		d.qkNorm = true
		return mk(flat1D(n.Gamma)).b
	}
	for _, b := range m.Blocks {
		experts := make([]moeFFN, len(b.MoE.Experts))
		for i, ex := range b.MoE.Experts {
			experts[i] = moeFFN{wG: lin(ex.Wgate), wU: lin(ex.Wup), wD: lin(ex.Wdown)}
		}
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			qN: qkGain(b.QNorm), kN: qkGain(b.KNorm),
			moeRouter: lin(b.MoE.Router.W), moeExperts: experts,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newQwen2MoEDecoder builds a device Decoder for an nlp.Qwen2MoE. Qwen2-MoE is a routed sparse MoE
// PLUS a shared expert: the FFN output is sparse_moe(x) + sigmoid(x·gate)·shared_expert(x), a single
// SwiGLU run on every token and scaled by a per-token sigmoid gate (recordMoE's shared-expert tail).
// Attention is the Llama core with Qwen2's q/k/v projection biases (o_proj bias-free). cuda-only.
func newQwen2MoEDecoder(m *nlp.Qwen2MoE, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	if len(m.Blocks) == 0 || m.Blocks[0].MoE == nil || len(m.Blocks[0].MoE.Experts) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): Qwen2MoE has no MoE experts", ops.name)
	}
	b0 := m.Blocks[0]
	nExperts := len(b0.MoE.Experts)
	topK := b0.MoE.TopK
	if topK <= 0 {
		topK = 2
	}
	hidden := b0.MoE.Experts[0].Wgate.Shape()[1]
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.moe = true
	d.nExperts = nExperts
	d.topK = topK

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	fusedBias := func(bq, bk, bv *tensor.Tensor) buffer {
		if bq == nil && bk == nil && bv == nil {
			return nil
		}
		return mk(append(append(append([]float32{}, flat1D(bq)...), flat1D(bk)...), flat1D(bv)...)).b
	}
	for _, b := range m.Blocks {
		experts := make([]moeFFN, len(b.MoE.Experts))
		for i, ex := range b.MoE.Experts {
			experts[i] = moeFFN{wG: lin(ex.Wgate), wU: lin(ex.Wup), wD: lin(ex.Wdown)}
		}
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), qkvBias: fusedBias(b.Bq, b.Bk, b.Bv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			moeRouter: lin(b.MoE.Router.W), moeExperts: experts,
			moeShared:     moeFFN{wG: lin(b.Shared.Wgate), wU: lin(b.Shared.Wup), wD: lin(b.Shared.Wdown)},
			moeSharedGate: lin(b.SharedGate),
			kC:            mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newGraniteMoEDecoder builds a device Decoder for an nlp.GraniteMoE — the MoE sibling of dense
// Granite: a plain Llama attention core + a sparse Mixture-of-Experts FFN, wrapped in the four
// Granite config scalars (all identity when 0/1, so this is newMixtralDecoder + the load-time folds
// from newDecoder's Granite handling). AttentionMult overrides the softmax scale; EmbeddingMult
// scales the gathered embedding; ResidualMult is baked into Wo AND every expert's Wdown (their
// matmul feeds the residual); LogitsScale is baked as 1/scale into the untied lm_head. cuda-only.
func newGraniteMoEDecoder(m *nlp.GraniteMoE, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	if len(m.Blocks) == 0 || m.Blocks[0].MoE == nil || len(m.Blocks[0].MoE.Experts) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): GraniteMoE has no MoE experts", ops.name)
	}
	b0 := m.Blocks[0]
	nExperts := len(b0.MoE.Experts)
	topK := b0.MoE.TopK
	if topK <= 0 {
		topK = 2
	}
	hidden := b0.MoE.Experts[0].Wgate.Shape()[1]
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.moe = true
	d.nExperts = nExperts
	d.topK = topK
	if cfg.AttentionMult != 0 {
		d.scale = float32(cfg.AttentionMult)
	}
	if cfg.EmbeddingMult != 0 && cfg.EmbeddingMult != 1 {
		d.embMult = float32(cfg.EmbeddingMult)
	}
	resMult := float32(1)
	if cfg.ResidualMult != 0 && cfg.ResidualMult != 1 {
		resMult = float32(cfg.ResidualMult)
	}
	outScale := float32(1)
	if cfg.LogitsScale != 0 && cfg.LogitsScale != 1 {
		outScale = float32(1 / cfg.LogitsScale)
	}

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	linS := func(w *tensor.Tensor, s float32) linear { // weight pre-scaled by s (ResidualMult fold)
		in, out := w.Shape()[0], w.Shape()[1]
		f := flat2D(w)
		if s != 1 {
			for i := range f {
				f[i] *= s
			}
		}
		return f32Linear{w: mk(f).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		experts := make([]moeFFN, len(b.MoE.Experts))
		for i, ex := range b.MoE.Experts {
			experts[i] = moeFFN{wG: lin(ex.Wgate), wU: lin(ex.Wup), wD: linS(ex.Wdown, resMult)}
		}
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: linS(b.Wo, resMult),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			moeRouter: lin(b.MoE.Router.W), moeExperts: experts,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = linS(m.Out, outScale)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newOLMoEDecoder builds a device Decoder for an nlp.OLMoE (Allen AI sparse-MoE). OLMoE is the
// pre-norm sparse-MoE composition: the plain Llama attention core with FULL-WIDTH q/k RMSNorm
// (qkNorm+qkNormFull, one RMSNorm over the whole q/k projection — same as OLMo2) plus a sparse
// Mixture-of-Experts FFN (recordMoE) and an untied lm_head. cuda-only (MoE + full-width QK-norm).
func newOLMoEDecoder(m *nlp.OLMoE, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	if len(m.Blocks) == 0 || m.Blocks[0].MoE == nil || len(m.Blocks[0].MoE.Experts) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): OLMoE has no MoE experts", ops.name)
	}
	b0 := m.Blocks[0]
	nExperts := len(b0.MoE.Experts)
	topK := b0.MoE.TopK
	if topK <= 0 {
		topK = 2
	}
	hidden := cfg.Hidden
	if hidden <= 0 {
		hidden = b0.MoE.Experts[0].Wgate.Shape()[1]
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.moe = true
	d.nExperts = nExperts
	d.topK = topK
	d.qkNorm = true     // full-width q/k RMSNorm before RoPE...
	d.qkNormFull = true // ...over the whole projection (OLMo2/OLMoE)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	for _, b := range m.Blocks {
		experts := make([]moeFFN, len(b.MoE.Experts))
		for i, ex := range b.MoE.Experts {
			experts[i] = moeFFN{wG: lin(ex.Wgate), wU: lin(ex.Wup), wD: lin(ex.Wdown)}
		}
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			qN: mk(flat1D(b.QNorm.Gamma)).b, kN: mk(flat1D(b.KNorm.Gamma)).b, // full-width [qDim]/[kvDim]
			moeRouter: lin(b.MoE.Router.W), moeExperts: experts,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newStarCoder2Decoder builds a device Decoder for an nlp.StarCoder2: LayerNorm-with-bias +
// biased q/k/v/o projections + a biased 2-layer GELU MLP (c_fc → GELU → c_proj) + FULL rope +
// GQA + untied head. Every one of those is a core generalization (lnBias, biased projections,
// ffnGELU); rotary stays full (rotaryDim == dk). cuda-only.
func newStarCoder2Decoder(m *nlp.StarCoder2, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true  // LayerNorm-with-bias
	d.ffnGELU = true // 2-layer GELU MLP
	// rotary stays full (rotaryDim == dk, the newDecoderCommon default) — no inv override.

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	fusedBias := func(bq, bk, bv *tensor.Tensor) buffer { // [Bq | Bk | Bv] = [D+2·kvDim]
		fb := append(append(append([]float32{}, flat1D(bq)...), flat1D(bk)...), flat1D(bv)...)
		return mk(fb).b
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), qkvBias: fusedBias(b.Bq, b.Bk, b.Bv),
			wo: lin(b.Wo), oBias: mk(flat1D(b.Bo)).b,
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			gFFN: mk(flat1D(b.PostAttnNorm.Gamma)).b, bFFN: mk(flat1D(b.PostAttnNorm.Beta)).b,
			wG: lin(b.Wfc), fcBias: mk(flat1D(b.Bfc)).b,
			wD: lin(b.Wproj), projBias: mk(flat1D(b.Bproj)).b,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.bFinal = mk(flat1D(m.Norm.Beta))
	d.out = lin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newPhiDecoder builds a device Decoder for an nlp.Phi: the hardest new-arch shape so far — ONE
// input LayerNorm feeding BOTH attention and the MLP (one-norm parallel residual), biased q/k/v/dense
// projections, a biased 2-layer GELU MLP (fc1 → GELU → fc2), PARTIAL rotary, MHA, a biased untied
// lm_head and a final LayerNorm. Every one of those is a core generalization (parallelRes, biased
// projections, ffnGELU, lnBias, rotaryDim, outBias). cuda-only.
func newPhiDecoder(m *nlp.Phi, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.HeadDim
	if headDim <= 0 {
		headDim = cfg.Dim / cfg.Heads
	}
	rotaryDim := cfg.RotaryDim
	if rotaryDim <= 0 {
		pct := cfg.RotaryPct
		if pct == 0 {
			pct = 0.25
		}
		rotaryDim = int(float64(headDim) * pct)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true      // LayerNorm-with-bias
	d.ffnGELU = true     // 2-layer GELU MLP
	d.parallelRes = true // one-norm parallel residual (input_layernorm feeds attn AND mlp)
	d.rotaryDim = rotaryDim
	if d.rotaryDim <= 0 || d.rotaryDim > d.dk || d.rotaryDim%2 != 0 {
		return nil, fmt.Errorf("llamagpu(%s): Phi rotaryDim %d invalid for headDim %d", ops.name, d.rotaryDim, d.dk)
	}
	invF64, posDiv64 := backend.RoPEFreqs(d.rotaryDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.rotaryDim/2)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	fusedBias := func(bq, bk, bv *tensor.Tensor) buffer {
		fb := append(append(append([]float32{}, flat1D(bq)...), flat1D(bk)...), flat1D(bv)...)
		return mk(fb).b
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), qkvBias: fusedBias(b.Bq, b.Bk, b.Bv),
			wo: lin(b.Wdense), oBias: mk(flat1D(b.Bdense)).b,
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			// gFFN/bFFN unused: parallel one-norm reuses the attn norm (d.xn) for the MLP.
			wG: lin(b.Wfc1), fcBias: mk(flat1D(b.Bfc1)).b,
			wD: lin(b.Wfc2), projBias: mk(flat1D(b.Bfc2)).b,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	d.bFinal = mk(flat1D(m.FinalNorm.Beta))
	d.out = lin(m.Out)
	d.outBias = mk(flat1D(m.OutBias))
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newGPTNeoXDecoder uploads an nlp.GPTNeoX onto the batched Decoder core. GPT-NeoX is the TWO-norm
// parallel residual: a separate input_layernorm feeds attention and post_attention_layernorm feeds
// the MLP, but BOTH read the raw residual x0 and both outputs sum onto it. Beyond that it is
// LayerNorm-with-bias, biased q/k/v/dense, a biased GELU MLP, (partial or full) rotary and an
// untied lm_head with no bias. The fourth new-arch GPU graph decoder — and the one that exercises
// parallelTwoNorm (the FFN norm must be taken from x0 before the attention output is added back).
func newGPTNeoXDecoder(m *nlp.GPTNeoX, ops backendOps) (*Decoder, error) {
	cfg := m.Config
	kvH := cfg.KVHeads
	if kvH <= 0 {
		kvH = cfg.Heads
	}
	headDim := cfg.Dim / cfg.Heads
	rotaryDim := cfg.RotaryDim
	if rotaryDim <= 0 {
		pct := cfg.RotaryPct
		if pct == 0 {
			pct = 1 // GPT-NeoX default is full rope
		}
		rotaryDim = int(float64(headDim) * pct)
	}
	lc := nlp.LlamaConfig{
		Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim, Heads: cfg.Heads, KVHeads: kvH,
		Layers: cfg.Layers, Hidden: cfg.Hidden, Eps: cfg.Eps, RopeBase: cfg.RopeBase,
	}
	d, derr := newDecoderCommon(lc, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	d.lnBias = true          // LayerNorm-with-bias
	d.ffnGELU = true         // 2-layer GELU MLP
	d.parallelTwoNorm = true // two-norm parallel residual (input_ln → attn, post_attn_ln → mlp, both over x0)
	d.rotaryDim = rotaryDim
	if d.rotaryDim <= 0 || d.rotaryDim > d.dk || d.rotaryDim%2 != 0 {
		return nil, fmt.Errorf("llamagpu(%s): GPT-NeoX rotaryDim %d invalid for headDim %d", ops.name, d.rotaryDim, d.dk)
	}
	invF64, posDiv64 := backend.RoPEFreqs(d.rotaryDim, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.rotaryDim/2)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)

	var err error
	mk := d.mkBuf(&err)
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	fused := func(wq, wk, wv *tensor.Tensor) linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		nt := nq + nk + nv
		w := make([]float32, in*nt)
		for i := range in {
			row := w[i*nt : (i+1)*nt]
			copy(row[:nq], fq[i*nq:(i+1)*nq])
			copy(row[nq:nq+nk], fk[i*nk:(i+1)*nk])
			copy(row[nq+nk:], fv[i*nv:(i+1)*nv])
		}
		return f32Linear{w: mk(w).b, k: in, n: nt}
	}
	fusedBias := func(bq, bk, bv *tensor.Tensor) buffer {
		fb := append(append(append([]float32{}, flat1D(bq)...), flat1D(bk)...), flat1D(bv)...)
		return mk(fb).b
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), qkvBias: fusedBias(b.Bq, b.Bk, b.Bv),
			wo: lin(b.Wo), oBias: mk(flat1D(b.Bo)).b,
			gAttn: mk(flat1D(b.InputNorm.Gamma)).b, bAttn: mk(flat1D(b.InputNorm.Beta)).b,
			gFFN: mk(flat1D(b.PostAttnNorm.Gamma)).b, bFFN: mk(flat1D(b.PostAttnNorm.Beta)).b,
			wG: lin(b.Wh), fcBias: mk(flat1D(b.Bh)).b,
			wD: lin(b.Wout), projBias: mk(flat1D(b.Bout)).b,
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.FinalNorm.Gamma))
	d.bFinal = mk(flat1D(m.FinalNorm.Beta))
	d.out = lin(m.Out) // untied embed_out, no bias
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// newQuantDecoder uploads a quantized Llama: RMSNorm gains + KV caches as f32 device buffers, every
// projection as a RESIDENT quantized weight consumed by the record-mode QMatMulResident (§T415) —
// the 4-8× smaller weights of quantization combined with the batched-decode speedup.
func newQuantDecoder(m *nlp.QuantLlama, ops backendOps) (*Decoder, error) {
	if ops.uploadQWeight == nil {
		return nil, fmt.Errorf("llamagpu(%s): backend has no resident quantized upload", ops.name)
	}
	cfg := m.Config
	d, derr := newDecoderCommon(cfg, m.TokEmb, ops)
	if derr != nil {
		return nil, derr
	}
	var err error
	mk := d.mkBuf(&err)
	qlin := func(q *nn.QuantLinear) linear { // resident quantized [Out,In] weight
		if err != nil {
			return quantLinear{}
		}
		w, e := ops.uploadQWeight(q.Weight, uint32(q.QT), q.Out, q.In)
		if e != nil {
			err = e
			return quantLinear{}
		}
		d.qweights = append(d.qweights, w)
		return quantLinear{w: w}
	}
	// fused QKV for quantized blocks (§T613): ggml resident weights are [Out,In] ROW-major, so
	// fusing = appending the raw quantized bytes (out rows q‖k‖v) — valid only when all three
	// projections share one quant type; mixed-type blocks keep the unfused three-matmul chain.
	qfused := func(q1, q2, q3 *nn.QuantLinear) linear {
		if err != nil || q1.QT != q2.QT || q2.QT != q3.QT {
			return nil
		}
		wb := make([]byte, 0, len(q1.Weight)+len(q2.Weight)+len(q3.Weight))
		wb = append(append(append(wb, q1.Weight...), q2.Weight...), q3.Weight...)
		w, e := ops.uploadQWeight(wb, uint32(q1.QT), q1.Out+q2.Out+q3.Out, q1.In)
		if e != nil {
			err = e
			return nil
		}
		d.qweights = append(d.qweights, w)
		return quantLinear{w: w}
	}
	for _, b := range m.Blocks {
		gb := block{
			wqkv: qfused(b.Wq, b.Wk, b.Wv), wo: qlin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
			wG: qlin(b.FFN.Gate), wU: qlin(b.FFN.Up), wD: qlin(b.FFN.Down),
			kC: mk(make([]float32, d.maxLen*d.kvDim)).b, vC: mk(make([]float32, d.maxLen*d.kvDim)).b,
		}
		if gb.wqkv == nil { // mixed quant types: keep the unfused projections
			gb.wq, gb.wk, gb.wv = qlin(b.Wq), qlin(b.Wk), qlin(b.Wv)
		}
		d.blocks = append(d.blocks, gb)
	}
	d.gFinal = mk(flat1D(m.Norm.Gamma))
	d.out = qlin(m.Out)
	d.allocScratch(mk)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// encodeStep records the whole single-token layer stack + vocab head for absolute position
// pos into a fresh command buffer WITHOUT committing it. The recorded commands depend only
// on pos — not on the token, whose embedding is read from d.dx at execution time — which is
// what makes the §T614 encode-overlap legal: while the GPU executes step pos, the host
// pre-encodes step pos+1, and the next Step only uploads dx and commits.
func (d *Decoder) encodeStep(pos int) (recorder, error) {
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, KVH, dk, kvDim := d.qDim, d.h, d.kvH, d.dk, d.kvDim // D = attention Q/O width (h·dk); = d.d unless head_dim decoupled
	if d.rwkv {                                               // RWKV: an extra LayerNorm (ln0) transforms the embedding once before block 0
		if e := r.LayerNorm(d.dx.b, d.rwPreLNg.b, d.rwPreLNb.b, d.dx.b, 1, d.d, d.eps); e != nil {
			r.Free()
			return nil, e
		}
	}
	for _, b := range d.blocks {
		if d.rwkv { // RWKV-4 block: WKV time-mix + gated squared-ReLU channel-mix, no attention/KV cache
			if e := d.recordRWKVBlock(r, b); e != nil {
				r.Free()
				return nil, e
			}
			continue
		}
		if d.mamba { // Mamba selective-scan block: no attention/FFN split — the mixer is the whole layer
			if e := d.recordMambaBlock(r, b); e != nil {
				r.Free()
				return nil, e
			}
			continue
		}
		if d.mamba2 { // Mamba-2 SSD block (scalar-decay, grouped B/C, gated RMSNorm)
			if e := d.recordMamba2Block(r, b); e != nil {
				r.Free()
				return nil, e
			}
			continue
		}
		if d.mla { // DeepSeek-V2 Multi-head Latent Attention (rectangular, low-rank latent KV)
			if e := firstErr(d.recordMLAAttention(r, b, pos, 1), d.recordFFNSublayer(r, b, 1)); e != nil {
				r.Free()
				return nil, e
			}
			continue
		}
		// attention projections: fused single QKV matmul when available (§T613 — the q/k
		// bands rotate IN PLACE inside the combined row, k/v append to the cache straight
		// from their bands, attention reads q at band offset 0), else the unfused three.
		// Recording order is execution order: norm FIRST, then the projection branch.
		e := d.recordAttnNorm(r, b, 1)
		qBuf := d.q.b
		if e == nil && b.wqkv != nil {
			qBuf = d.qkv.b
			e = firstErr(
				d.recordQKVProj(r, b, 1),
				d.recordQKNorm(r, b, 1, D+2*kvDim),                // per-head QK-norm (Qwen3); no-op otherwise
				d.ropeQK(r, d.qkv.b, d.dinv.b, 1, D+2*kvDim, pos), // q+k bands, ONE dispatch (§T613); full ∨ partial rotary
				r.Blit(d.qkv.b, D, b.kC, pos*kvDim, kvDim),
				r.Blit(d.qkv.b, D+kvDim, b.vC, pos*kvDim, kvDim),
			)
		} else if e == nil {
			e = firstErr(
				b.wq.record(r, d.xn.b, d.q.b, 1),
				b.wk.record(r, d.xn.b, d.k.b, 1),
				b.wv.record(r, d.xn.b, d.v_.b, 1),
				r.RoPE(d.q.b, d.dinv.b, d.q.b, 1, D, H, dk, d.half, pos, d.posDiv),
				r.RoPE(d.k.b, d.dinv.b, d.k.b, 1, kvDim, KVH, dk, d.half, pos, d.posDiv),
				r.Blit(d.k.b, 0, b.kC, pos*kvDim, kvDim),
				r.Blit(d.v_.b, 0, b.vC, pos*kvDim, kvDim),
			)
		}
		e = firstErr(
			e,
			d.recordMHA(r, qBuf, b.kC, b.vC, d.attn.b, 1, pos+1),
			d.recordOProj(r, b, 1), // dx += attn·Wo (+ optional o-bias)
			d.recordFFNSublayer(r, b, 1),
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		d.norm(r, d.dx.b, d.gFinal.b, d.bFinalBeta(), d.xn.b, 1),
		d.recordLogits(r, 1),
	); e != nil {
		r.Free()
		return nil, e
	}
	return r, nil
}

// dropPending frees a pre-encoded next-step recorder that can no longer be used (the
// position moved unexpectedly, a prefill rewrote the cache, or the decoder is released).
func (d *Decoder) dropPending() {
	if d.pending != nil {
		d.pending.Free()
		d.pending = nil
	}
	d.pendingPos = -1
}

// Step advances the decoder by one token at absolute position pos (== cache length before the call),
// recording the whole layer stack + vocab head into one command buffer, and returns the [vocab]
// logits. pos must be < the model's Ctx. Sequential calls run the §T614 encode-overlap: while the
// GPU executes this step, the next position's command buffer is pre-encoded on the host.
func (d *Decoder) Step(token, pos int) ([]float32, error) {
	if pos < 0 || pos >= d.maxLen {
		return nil, fmt.Errorf("llamagpu(%s): pos %d out of [0,%d)", d.ops.name, pos, d.maxLen)
	}
	if token < 0 || token >= d.v {
		return nil, fmt.Errorf("llamagpu(%s): token %d out of vocab %d", d.ops.name, token, d.v)
	}
	// Mamba/Mamba-2 carry per-block conv/SSM state across Step calls with no KV cache; a fresh
	// sequence (pos==0) must start from zeroed state.
	if pos == 0 {
		if d.mamba {
			if err := d.resetMambaState(); err != nil {
				return nil, err
			}
		}
		if d.mamba2 {
			if err := d.resetMamba2State(); err != nil {
				return nil, err
			}
		}
		if d.rwkv {
			if err := d.resetRWKVState(); err != nil {
				return nil, err
			}
		}
	}
	// the token's embedding must be resident BEFORE commit — the recorded chain reads d.dx.
	if err := d.dx.b.UploadF32(d.gatherEmbed(token)); err != nil {
		return nil, err
	}
	var r recorder
	if d.pending != nil && d.pendingPos == pos {
		r, d.pending, d.pendingPos = d.pending, nil, -1
	} else {
		d.dropPending()
		var err error
		if r, err = d.encodeStep(pos); err != nil {
			return nil, err
		}
	}
	if err := r.Commit(); err != nil {
		r.Free()
		return nil, err
	}
	// encode-overlap (§T614): pre-encode pos+1 while the GPU executes pos. Failure is not
	// fatal — the next Step simply encodes fresh. Gated on asyncEncode: the vulkan bridge's
	// single global recording context cannot hold a second open recorder. Disabled for Mamba:
	// its blocks mutate resident conv/SSM state, so a pre-encoded pos+1 must not be recorded
	// until pos's state writes are committed.
	if d.ops.asyncEncode && !d.mamba && !d.mamba2 && !d.rwkv && pos+1 < d.maxLen {
		if nr, err := d.encodeStep(pos + 1); err == nil {
			d.pending, d.pendingPos = nr, pos+1
		}
	}
	if err := r.Wait(); err != nil {
		r.Free()
		return nil, err
	}
	r.Free()
	out := make([]float32, d.v)
	if err := d.logits.b.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// StepN advances the decoder by k tokens at absolute positions pos..pos+k-1 in ONE recorded step
// (§T418): the whole layer stack runs over [k,·] rows with causal attention against the growing
// cache (row i attends up to pos+i), and the new k KV rows are appended. Returns the [k,vocab]
// logits (row i = logits after tokens[..i]). This is the prompt-PREFILL fast path — one dispatch
// round-trip for the whole prompt instead of one per token — and the target-verification step
// speculative decoding needs. pos+k must be ≤ the model's Ctx.
func (d *Decoder) StepN(tokens []int, pos int) ([]float32, error) {
	k := len(tokens)
	if k == 0 {
		return nil, fmt.Errorf("llamagpu(%s): StepN needs ≥1 token", d.ops.name)
	}
	if pos < 0 || pos+k > d.maxLen {
		return nil, fmt.Errorf("llamagpu(%s): StepN [%d,%d) out of [0,%d)", d.ops.name, pos, pos+k, d.maxLen)
	}
	if d.mamba || d.mamba2 || d.rwkv {
		// Mamba/Mamba-2/RWKV's recurrence is inherently sequential (each token's state update reads the
		// previous token's), so there is no batched prefill — process the prompt token by token through
		// Step, which carries the recurrent state and resets it at pos==0. Returns [k, vocab].
		out := make([]float32, k*d.v)
		for i, tok := range tokens {
			row, err := d.Step(tok, pos+i)
			if err != nil {
				return nil, err
			}
			copy(out[i*d.v:(i+1)*d.v], row)
		}
		return out, nil
	}
	// a batched step rewrites the cache and scratch — any single-step encode pre-staged for
	// the old position is now invalid (§T614).
	d.dropPending()
	host := make([]float32, k*d.d)
	for i, tok := range tokens {
		if tok < 0 || tok >= d.v {
			return nil, fmt.Errorf("llamagpu(%s): token %d out of vocab %d", d.ops.name, tok, d.v)
		}
		copy(host[i*d.d:(i+1)*d.d], d.gatherEmbed(tok))
	}
	if err := d.dx.b.UploadF32(host); err != nil {
		return nil, err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, KVH, dk, kvDim := d.qDim, d.h, d.kvH, d.dk, d.kvDim // D = attention Q/O width (h·dk); = d.d unless head_dim decoupled
	stride := D + 2*kvDim
	for _, b := range d.blocks {
		if d.mla { // DeepSeek-V2 MLA prefill: k new tokens at position pos
			if e := firstErr(d.recordMLAAttention(r, b, pos, k), d.recordFFNSublayer(r, b, k)); e != nil {
				r.Free()
				return nil, e
			}
			continue
		}
		// fused QKV (§T613): one [k,stride] matmul; q/k bands rotate in place (RoPEAt's
		// width parameter acts as the row stride), then Copy2D extracts q contiguously
		// and deposits the k/v bands directly as cache rows. Recording order = execution
		// order: norm first, then the projection branch.
		e := d.recordAttnNorm(r, b, k)
		if e == nil && b.wqkv != nil {
			e = firstErr(
				d.recordQKVProj(r, b, k),
				d.recordQKNorm(r, b, k, stride),                // per-head QK-norm (Qwen3); no-op otherwise
				d.ropeQK(r, d.qkv.b, d.dinv.b, k, stride, pos), // q+k bands, ONE dispatch (§T613); full ∨ partial rotary
				r.Copy2D(d.qkv.b, 0, stride, d.q.b, 0, D, k, D),
				r.Copy2D(d.qkv.b, D, stride, b.kC, pos*kvDim, kvDim, k, kvDim),
				r.Copy2D(d.qkv.b, D+kvDim, stride, b.vC, pos*kvDim, kvDim, k, kvDim),
			)
		} else if e == nil {
			e = firstErr(
				b.wq.record(r, d.xn.b, d.q.b, k),
				b.wk.record(r, d.xn.b, d.k.b, k),
				b.wv.record(r, d.xn.b, d.v_.b, k),
				r.RoPE(d.q.b, d.dinv.b, d.q.b, k, D, H, dk, d.half, pos, d.posDiv),
				r.RoPE(d.k.b, d.dinv.b, d.k.b, k, kvDim, KVH, dk, d.half, pos, d.posDiv),
				r.Blit(d.k.b, 0, b.kC, pos*kvDim, k*kvDim),
				r.Blit(d.v_.b, 0, b.vC, pos*kvDim, k*kvDim),
			)
		}
		e = firstErr(
			e,
			// sq=k vs sk=pos+k: the kernel's causal offset (sk-sq = pos) makes row i attend
			// through absolute position pos+i — exactly the prefill/verify semantics.
			d.recordMHA(r, d.q.b, b.kC, b.vC, d.attn.b, k, pos+k),
			d.recordOProj(r, b, k), // dx += attn·Wo (+ optional o-bias)
			d.recordFFNSublayer(r, b, k),
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		d.norm(r, d.dx.b, d.gFinal.b, d.bFinalBeta(), d.xn.b, k),
		d.recordLogits(r, k),
		r.Finish(),
	); e != nil {
		r.Free()
		return nil, e
	}
	r.Free()
	out := make([]float32, k*d.v)
	if err := d.logits.b.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Generate runs the batched decode as a text-generation loop: it prefills the prompt, then samples
// up to maxNew tokens (bounded by the model's Ctx), feeding each back. Returns prompt+generated
// token ids. With a greedy sampler it produces the same ids as nlp.Llama.Generate.
func (d *Decoder) Generate(prompt []int, maxNew int, s nlp.TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): Generate needs a non-empty prompt", d.ops.name)
	}
	out := append([]int(nil), prompt...)
	// prefill the whole prompt in ONE recorded step (§T418) — one dispatch round-trip
	// instead of one per prompt token.
	all, err := d.StepN(prompt, 0)
	if err != nil {
		return nil, err
	}
	pos := len(prompt)
	logits := all[(len(prompt)-1)*d.v:] // last row = logits after the full prompt
	buf := make([]float64, d.v)
	for range maxNew {
		if pos >= d.maxLen {
			break
		}
		for i, x := range logits {
			buf[i] = float64(x)
		}
		next := s.SampleWithHistory(buf, out)
		out = append(out, next)
		l, err := d.Step(next, pos)
		if err != nil {
			return nil, err
		}
		logits = l
		pos++
	}
	return out, nil
}

// Vocab returns the model's vocabulary size.
func (d *Decoder) Vocab() int { return d.v }

// Ctx returns the model's maximum context length (the KV-cache capacity).
func (d *Decoder) Ctx() int { return d.maxLen }

// Release frees all device buffers and resident quantized weights.
func (d *Decoder) Release() {
	d.dropPending()
	for _, b := range d.all {
		if b != nil {
			b.Release()
		}
	}
	d.all = nil
	for _, w := range d.qweights {
		if w != nil {
			_ = w.Close()
		}
	}
	d.qweights = nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func firstErr(errs ...error) error {
	for _, e := range errs {
		if e != nil {
			return e
		}
	}
	return nil
}

func flat2D(t *tensor.Tensor) []float32 {
	r, c := t.Shape()[0], t.Shape()[1]
	o := make([]float32, r*c)
	for i := range r {
		for j := range c {
			o[i*c+j] = float32(t.AtF64(i, j))
		}
	}
	return o
}

func flat1D(t *tensor.Tensor) []float32 {
	n := t.Shape()[0]
	o := make([]float32, n)
	for i := range n {
		o[i] = float32(t.AtF64(i))
	}
	return o
}

func embedRow(table *tensor.Tensor, row, cols int) []float32 {
	o := make([]float32, cols)
	for j := range cols {
		o[j] = float32(table.AtF64(row, j))
	}
	return o
}

// gatherEmbed reads a token's embedding row and applies the Granite EmbeddingMult (embMult == 1
// for every non-Granite model, so this is embedRow verbatim then).
func (d *Decoder) gatherEmbed(token int) []float32 {
	e := embedRow(d.table, token, d.d)
	if d.embMult != 1 {
		for i := range e {
			e[i] *= d.embMult
		}
	}
	return e
}
