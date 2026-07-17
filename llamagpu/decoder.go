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
	unarySiLU    = 6
	unaryGELU    = 9
	unaryReLU2   = 10 // squared ReLU (Nemotron); cuda-only
	unarySigmoid = 11 // plain sigmoid (Qwen2-MoE shared-expert gate); cuda-only
	binaryAdd    = 0
	binaryMul    = 2
	binarySwiGLU = 6 // fused silu(a)·b — one dispatch instead of SiLU+Mul (§T613)
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
	MoEGate(logits, weights buffer, rows, e, k int) error
	RowAxpy(dst, src, arow buffer, rows, cols int) error
	// MHARect is multi-head attention with a rectangular head shape: query/key share head width dqk,
	// value has a different head width dv (o is heads·dv wide). The DeepSeek-V2 MLA shape. cuda-only.
	MHARect(q, k, v, o buffer, sq, sk, heads, kvHeads, dqk, dv, causal, window int, scale float32) error
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
	qkvBias, oBias, fcBias, projBias buffer
	qN, kN                           buffer   // per-head query/key RMSNorm gains [dk] (Qwen3); nil ⇒ no QK-norm
	gAttn2, gFFN2                    buffer   // sandwich-norm OUTPUT gains (Gemma2 post_attention/post_feedforward); nil otherwise
	moeRouter                        linear   // sparse-MoE gate: dim → nExperts logits (Mixtral); nil unless d.moe
	moeExperts                       []moeFFN // the per-expert SwiGLU FFNs (Mixtral); len == d.nExperts
	moeShared                        moeFFN   // Qwen2-MoE shared expert (SwiGLU on every token); zero-value when absent
	moeSharedGate                    linear   // Qwen2-MoE shared-expert gate: dim → 1 logit, sigmoid-scaled; nil when absent
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
	eps, posDiv, scale                            float32
	embMult                                       float32 // gathered embeddings ×= embMult (IBM Granite EmbeddingMult); 1 for everything else

	blocks                                                []block
	out                                                   linear
	gFinal, bFinal, outBias, dinv                         *bufSlot
	dx, xn, xn2, q, k, v_, attn, ao, gate, up, mo, logits *bufSlot
	moeGate, moeW, moeCol                                 *bufSlot       // sparse-MoE scratch: router logits [·,E], routing weights [·,E], one weight column [·]
	qkv                                                   *bufSlot       // fused QKV output rows [·, d+2·kvDim] (§T613)
	pending                                               recorder       // pre-encoded next-step command buffer (§T614)
	pendingPos                                            int            // position pending encodes; -1 = none
	table                                                 *tensor.Tensor // token embedding, host-side gather source
	invHost                                               []float32      // RoPE inverse freqs (uploaded by allocScratch)
	all                                                   []buffer
	qweights                                              []qweight // resident quantized weights (quant decoder)
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
	if d.moe {
		// sparse MoE (Mixtral): pre-norm the input, then route → per-expert SwiGLU → weighted combine.
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
	e = firstErr(e, r.MoEGate(d.moeGate.b, d.moeW.b, rows, d.nExperts, d.topK))
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
	if b.moeSharedGate != nil {
		// Qwen2-MoE shared expert: dx += sigmoid(in·gate) · SwiGLU_shared(in), run on every token.
		s := b.moeShared
		e = firstErr(e,
			b.moeSharedGate.record(r, in, d.moeCol.b, rows), // gate logit [rows,1] → moeCol
			r.Unary(d.moeCol.b, d.moeCol.b, unarySigmoid),   // sigmoid gate [rows]
			s.wG.record(r, in, d.gate.b, rows),
			s.wU.record(r, in, d.up.b, rows),
			r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU), // shared SwiGLU → gate
			s.wD.record(r, d.gate.b, d.mo.b, rows),             // shared output → mo
			r.RowAxpy(d.dx.b, d.mo.b, d.moeCol.b, rows, d.d),   // dx += gate · shared_out
		)
	}
	return e
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
	for _, b := range d.blocks {
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
	// single global recording context cannot hold a second open recorder.
	if d.ops.asyncEncode && pos+1 < d.maxLen {
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
