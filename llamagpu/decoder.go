// Package llamagpu decodes nlp.Llama models on the GPU with batched command buffers (ADR-0019):
// each per-token step records the whole layer stack into ONE command buffer over device-resident
// weights and KV cache, instead of paying a dispatch round-trip per op. Measured on a real model
// (D=512, GQA 8:2, 6 layers): 24× faster than nlp.Llama.DecodeStep on Metal and 21× on Vulkan,
// with token-for-token identical greedy output (§T404/§T409).
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
	Blit(src buffer, srcOff int, dst buffer, dstOff, n int) error
	// Copy2D moves a strided rows×rowFloats sub-matrix (the fused-QKV band extraction, §T613):
	// row r copies from src[srcOff+r·srcStride:] to dst[dstOff+r·dstStride:].
	Copy2D(src buffer, srcOff, srcStride int, dst buffer, dstOff, dstStride, rows, rowFloats int) error
	MHA(q, k, v, o buffer, sq, sk, dm, heads, kvHeads, dk, causal, window int, scale float32) error
	Unary(x, o buffer, op int) error
	Binary(a, b, o buffer, op int) error
	QMatMulResident(x buffer, w qweight, o buffer, m int) error
	Finish() error
	Free()
}

// backendOps is the per-backend constructor vtable the adapters fill in.
type backendOps struct {
	name          string
	newBuffer     func([]float32) (buffer, error)
	newRecorder   func() (recorder, error)
	uploadQWeight func(weight []byte, qt uint32, n, k int) (qweight, error) // resident quantized upload
}

type block struct {
	wq, wk, wv, wo, wG, wU, wD linear
	// wqkv is the fused QKV projection (§T613): one [in, D+2·kvDim] weight whose output row
	// is [q | k | v]. Non-nil = Step/StepN run the fused chain (one matmul instead of three);
	// nil = fall back to wq/wk/wv (quantized blocks whose projections mix quant types).
	wqkv                linear
	gAttn, gFFN, kC, vC buffer
}

// Decoder holds a Llama's weights + KV cache as device-resident buffers and runs one batched decode
// step per token. Not safe for concurrent use — one Decoder per goroutine. Release when done.
type Decoder struct {
	ops                                           backendOps
	d, h, kvH, dk, kvDim, half, hidden, v, maxLen int
	eps, posDiv, scale                            float32

	blocks                                                []block
	out                                                   linear
	gFinal, dinv                                          *bufSlot
	dx, xn, xn2, q, k, v_, attn, ao, gate, up, mo, logits *bufSlot
	qkv                                                   *bufSlot       // fused QKV output rows [·, d+2·kvDim] (§T613)
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
	}
	if d.kvH <= 0 {
		d.kvH = d.h
	}
	if d.h <= 0 || d.d%d.h != 0 {
		return nil, fmt.Errorf("llamagpu(%s): dim %d not divisible by heads %d", ops.name, d.d, d.h)
	}
	d.dk = d.d / d.h
	d.kvDim = d.kvH * d.dk
	d.half = d.dk / 2
	d.scale = float32(1.0 / math.Sqrt(float64(d.dk)))

	invF64, posDiv64 := backend.RoPEFreqs(d.dk, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: d.h})
	d.invHost = make([]float32, d.half)
	for i := range d.invHost {
		d.invHost[i] = float32(invF64[i])
	}
	d.posDiv = float32(posDiv64)
	return d, nil
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
	d.q = mk(make([]float32, c*d.d))
	d.k = mk(make([]float32, c*d.kvDim))
	d.v_ = mk(make([]float32, c*d.kvDim))
	d.qkv = mk(make([]float32, c*(d.d+2*d.kvDim)))
	d.attn = mk(make([]float32, c*d.d))
	d.ao = mk(make([]float32, c*d.d))
	d.gate = mk(make([]float32, c*d.hidden))
	d.up = mk(make([]float32, c*d.hidden))
	d.mo = mk(make([]float32, c*d.d))
	d.logits = mk(make([]float32, c*d.v))
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
	for _, b := range m.Blocks {
		gb := block{
			wqkv: fused(b.Wq, b.Wk, b.Wv), wo: lin(b.Wo),
			gAttn: mk(flat1D(b.AttnNorm.Gamma)).b, gFFN: mk(flat1D(b.FFNNorm.Gamma)).b,
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

// Step advances the decoder by one token at absolute position pos (== cache length before the call),
// recording the whole layer stack + vocab head into one command buffer, and returns the [vocab]
// logits. pos must be < the model's Ctx.
func (d *Decoder) Step(token, pos int) ([]float32, error) {
	if pos < 0 || pos >= d.maxLen {
		return nil, fmt.Errorf("llamagpu(%s): pos %d out of [0,%d)", d.ops.name, pos, d.maxLen)
	}
	if token < 0 || token >= d.v {
		return nil, fmt.Errorf("llamagpu(%s): token %d out of vocab %d", d.ops.name, token, d.v)
	}
	if err := d.dx.b.UploadF32(embedRow(d.table, token, d.d)); err != nil {
		return nil, err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, KVH, dk, kvDim := d.d, d.h, d.kvH, d.dk, d.kvDim
	for _, b := range d.blocks {
		// attention projections: fused single QKV matmul when available (§T613 — the q/k
		// bands rotate IN PLACE inside the combined row, k/v append to the cache straight
		// from their bands, attention reads q at band offset 0), else the unfused three.
		// Recording order is execution order: norm FIRST, then the projection branch.
		e := r.RMSNorm(d.dx.b, b.gAttn, d.xn.b, 1, D, d.eps)
		qBuf := d.q.b
		if e == nil && b.wqkv != nil {
			qBuf = d.qkv.b
			e = firstErr(
				b.wqkv.record(r, d.xn.b, d.qkv.b, 1),
				r.RoPEAt(d.qkv.b, d.dinv.b, d.qkv.b, 0, 1, D, H, dk, d.half, pos, d.posDiv),
				r.RoPEAt(d.qkv.b, d.dinv.b, d.qkv.b, D, 1, kvDim, KVH, dk, d.half, pos, d.posDiv),
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
			r.MHA(qBuf, b.kC, b.vC, d.attn.b, 1, pos+1, D, H, KVH, dk, 1, 0, d.scale),
			b.wo.recordAdd(r, d.attn.b, d.ao.b, d.dx.b, 1), // dx += attn·Wo (fused epilogue, §T613)
			r.RMSNorm(d.dx.b, b.gFFN, d.xn2.b, 1, D, d.eps),
			b.wG.record(r, d.xn2.b, d.gate.b, 1),
			b.wU.record(r, d.xn2.b, d.up.b, 1),
			r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU),
			b.wD.recordAdd(r, d.gate.b, d.mo.b, d.dx.b, 1), // dx += swiglu·Wdown (fused epilogue)
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		r.RMSNorm(d.dx.b, d.gFinal.b, d.xn.b, 1, D, d.eps),
		d.out.record(r, d.xn.b, d.logits.b, 1),
		r.Finish(),
	); e != nil {
		r.Free()
		return nil, e
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
	host := make([]float32, k*d.d)
	for i, tok := range tokens {
		if tok < 0 || tok >= d.v {
			return nil, fmt.Errorf("llamagpu(%s): token %d out of vocab %d", d.ops.name, tok, d.v)
		}
		copy(host[i*d.d:(i+1)*d.d], embedRow(d.table, tok, d.d))
	}
	if err := d.dx.b.UploadF32(host); err != nil {
		return nil, err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, KVH, dk, kvDim := d.d, d.h, d.kvH, d.dk, d.kvDim
	stride := D + 2*kvDim
	for _, b := range d.blocks {
		// fused QKV (§T613): one [k,stride] matmul; q/k bands rotate in place (RoPEAt's
		// width parameter acts as the row stride), then Copy2D extracts q contiguously
		// and deposits the k/v bands directly as cache rows. Recording order = execution
		// order: norm first, then the projection branch.
		e := r.RMSNorm(d.dx.b, b.gAttn, d.xn.b, k, D, d.eps)
		if e == nil && b.wqkv != nil {
			e = firstErr(
				b.wqkv.record(r, d.xn.b, d.qkv.b, k),
				r.RoPEAt(d.qkv.b, d.dinv.b, d.qkv.b, 0, k, stride, H, dk, d.half, pos, d.posDiv),
				r.RoPEAt(d.qkv.b, d.dinv.b, d.qkv.b, D, k, stride, KVH, dk, d.half, pos, d.posDiv),
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
			r.MHA(d.q.b, b.kC, b.vC, d.attn.b, k, pos+k, D, H, KVH, dk, 1, 0, d.scale),
			b.wo.recordAdd(r, d.attn.b, d.ao.b, d.dx.b, k), // dx += attn·Wo (fused epilogue, §T613)
			r.RMSNorm(d.dx.b, b.gFFN, d.xn2.b, k, D, d.eps),
			b.wG.record(r, d.xn2.b, d.gate.b, k),
			b.wU.record(r, d.xn2.b, d.up.b, k),
			r.Binary(d.gate.b, d.up.b, d.gate.b, binarySwiGLU),
			b.wD.recordAdd(r, d.gate.b, d.mo.b, d.dx.b, k), // dx += swiglu·Wdown (fused epilogue)
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		r.RMSNorm(d.dx.b, d.gFinal.b, d.xn.b, k, D, d.eps),
		d.out.record(r, d.xn.b, d.logits.b, k),
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
