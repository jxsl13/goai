package llamagpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// gptBlock is one pre-LN GPT transformer block on the device: LayerNorm gains/biases, the four
// attention projections, the biased GELU MLP, and the block's KV cache.
type gptBlock struct {
	g1, b1, g2, b2 buffer // LN1/LN2 gamma+beta [d]
	wq, wk, wv, wo linear // attention projections [d,d], no bias
	w1, w2         linear // FFN up [d,4d] / down [4d,d]
	fb1, fb2       buffer // FFN biases [4d] / [d]
	kC, vC         buffer // KV caches [maxLen*d]
}

// GPTDecoder is the batched-decode engine for nlp.GPT models (§T422) — the GPT-2-style sibling of
// [Decoder]: LayerNorm instead of RMSNorm, learned positional embeddings instead of RoPE, a biased
// GELU MLP instead of SwiGLU. Each Step records the whole layer stack into one command buffer.
// Not safe for concurrent use; Release when done.
type GPTDecoder struct {
	ops                 backendOps
	d, h, dk, v, maxLen int
	eps, scale          float32

	blocks            []gptBlock
	lnfG, lnfB        *bufSlot
	head              linear
	dx, xn, xn2       *bufSlot
	q, k, v_          *bufSlot
	attn, ao, hid, mo *bufSlot
	logits            *bufSlot
	tokEmb, posEmb    *tensor.Tensor // host gather sources
	all               []buffer
}

// newGPTDecoder uploads m's weights via ops and prepares per-block KV caches up to Ctx.
func newGPTDecoder(m *nlp.GPT, ops backendOps) (*GPTDecoder, error) {
	cfg := m.Config
	d := &GPTDecoder{
		ops: ops,
		d:   cfg.Dim, h: cfg.Heads, v: cfg.Vocab, maxLen: cfg.Ctx,
		eps: float32(cfg.Eps), tokEmb: m.TokEmb, posEmb: m.PosEmb,
	}
	if d.h <= 0 || d.d%d.h != 0 {
		return nil, fmt.Errorf("llamagpu(%s): gpt dim %d not divisible by heads %d", ops.name, d.d, d.h)
	}
	d.dk = d.d / d.h
	d.scale = float32(1.0 / math.Sqrt(float64(d.dk)))

	var err error
	mk := func(data []float32) *bufSlot {
		if err != nil {
			return &bufSlot{}
		}
		b, e := ops.newBuffer(data)
		if e != nil {
			err = e
			return &bufSlot{}
		}
		d.all = append(d.all, b)
		return &bufSlot{b: b}
	}
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)).b, k: in, n: out}
	}
	for _, b := range m.Blocks {
		gb := gptBlock{
			g1: mk(flat1D(b.LN1.Gamma)).b, b1: mk(flat1D(b.LN1.Beta)).b,
			g2: mk(flat1D(b.LN2.Gamma)).b, b2: mk(flat1D(b.LN2.Beta)).b,
			wq: lin(b.Attn.Wq), wk: lin(b.Attn.Wk), wv: lin(b.Attn.Wv), wo: lin(b.Attn.Wo),
			w1: lin(b.W1), w2: lin(b.W2),
			fb1: mk(flat1D(b.B1)).b, fb2: mk(flat1D(b.B2)).b,
			kC: mk(make([]float32, d.maxLen*d.d)).b, vC: mk(make([]float32, d.maxLen*d.d)).b,
		}
		d.blocks = append(d.blocks, gb)
	}
	d.lnfG = mk(flat1D(m.LNf.Gamma))
	d.lnfB = mk(flat1D(m.LNf.Beta))
	d.head = lin(m.Head)
	ffn := 4 * d.d
	if len(m.Blocks) > 0 {
		ffn = m.Blocks[0].W1.Shape()[1]
	}
	c := d.maxLen // scratch sized for multi-token StepN (§T423)
	d.dx = mk(make([]float32, c*d.d))
	d.xn = mk(make([]float32, c*d.d))
	d.xn2 = mk(make([]float32, c*d.d))
	d.q = mk(make([]float32, c*d.d))
	d.k = mk(make([]float32, c*d.d))
	d.v_ = mk(make([]float32, c*d.d))
	d.attn = mk(make([]float32, c*d.d))
	d.ao = mk(make([]float32, c*d.d))
	d.hid = mk(make([]float32, c*ffn))
	d.mo = mk(make([]float32, c*d.d))
	d.logits = mk(make([]float32, c*d.v))
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// Step advances the decoder by one token at absolute position pos (== cache length before the
// call), recording the whole layer stack + head into one command buffer, and returns the [vocab]
// logits. The token embedding and learned positional embedding are summed host-side.
func (d *GPTDecoder) Step(token, pos int) ([]float32, error) {
	if pos < 0 || pos >= d.maxLen {
		return nil, fmt.Errorf("llamagpu(%s): gpt pos %d out of [0,%d)", d.ops.name, pos, d.maxLen)
	}
	if token < 0 || token >= d.v {
		return nil, fmt.Errorf("llamagpu(%s): gpt token %d out of vocab %d", d.ops.name, token, d.v)
	}
	x := embedRow(d.tokEmb, token, d.d)
	pe := embedRow(d.posEmb, pos, d.d)
	for i := range x {
		x[i] += pe[i]
	}
	if err := d.dx.b.UploadF32(x); err != nil {
		return nil, err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, dk := d.d, d.h, d.dk
	for _, b := range d.blocks {
		e := firstErr(
			r.LayerNorm(d.dx.b, b.g1, b.b1, d.xn.b, 1, D, d.eps),
			b.wq.record(r, d.xn.b, d.q.b, 1),
			b.wk.record(r, d.xn.b, d.k.b, 1),
			b.wv.record(r, d.xn.b, d.v_.b, 1),
			r.Blit(d.k.b, 0, b.kC, pos*D, D),
			r.Blit(d.v_.b, 0, b.vC, pos*D, D),
			r.MHA(d.q.b, b.kC, b.vC, d.attn.b, 1, pos+1, D, H, H, dk, 1, 0, d.scale),
			b.wo.record(r, d.attn.b, d.ao.b, 1),
			r.Binary(d.dx.b, d.ao.b, d.dx.b, binaryAdd),
			r.LayerNorm(d.dx.b, b.g2, b.b2, d.xn2.b, 1, D, d.eps),
			b.w1.record(r, d.xn2.b, d.hid.b, 1),
			r.AddBias(d.hid.b, b.fb1, d.hid.b, 1, d.ffnDim(b)),
			r.Unary(d.hid.b, d.hid.b, unaryGELU),
			b.w2.record(r, d.hid.b, d.mo.b, 1),
			r.AddBias(d.mo.b, b.fb2, d.mo.b, 1, D),
			r.Binary(d.dx.b, d.mo.b, d.dx.b, binaryAdd),
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		r.LayerNorm(d.dx.b, d.lnfG.b, d.lnfB.b, d.xn.b, 1, D, d.eps),
		d.head.record(r, d.xn.b, d.logits.b, 1),
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
// (§T423, the GPT sibling of Decoder.StepN): the layer stack runs over [k,·] rows with causal
// attention against the growing cache, biases broadcast via the AddBias record op, and the k KV
// rows are appended. Returns the [k,vocab] logits.
func (d *GPTDecoder) StepN(tokens []int, pos int) ([]float32, error) {
	k := len(tokens)
	if k == 0 {
		return nil, fmt.Errorf("llamagpu(%s): gpt StepN needs ≥1 token", d.ops.name)
	}
	if pos < 0 || pos+k > d.maxLen {
		return nil, fmt.Errorf("llamagpu(%s): gpt StepN [%d,%d) out of [0,%d)", d.ops.name, pos, pos+k, d.maxLen)
	}
	host := make([]float32, k*d.d)
	for i, tok := range tokens {
		if tok < 0 || tok >= d.v {
			return nil, fmt.Errorf("llamagpu(%s): gpt token %d out of vocab %d", d.ops.name, tok, d.v)
		}
		row := host[i*d.d : (i+1)*d.d]
		copy(row, embedRow(d.tokEmb, tok, d.d))
		pe := embedRow(d.posEmb, pos+i, d.d)
		for j := range row {
			row[j] += pe[j]
		}
	}
	if err := d.dx.b.UploadF32(host); err != nil {
		return nil, err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	D, H, dk := d.d, d.h, d.dk
	for _, b := range d.blocks {
		e := firstErr(
			r.LayerNorm(d.dx.b, b.g1, b.b1, d.xn.b, k, D, d.eps),
			b.wq.record(r, d.xn.b, d.q.b, k),
			b.wk.record(r, d.xn.b, d.k.b, k),
			b.wv.record(r, d.xn.b, d.v_.b, k),
			r.Blit(d.k.b, 0, b.kC, pos*D, k*D),
			r.Blit(d.v_.b, 0, b.vC, pos*D, k*D),
			r.MHA(d.q.b, b.kC, b.vC, d.attn.b, k, pos+k, D, H, H, dk, 1, 0, d.scale),
			b.wo.record(r, d.attn.b, d.ao.b, k),
			r.Binary(d.dx.b, d.ao.b, d.dx.b, binaryAdd),
			r.LayerNorm(d.dx.b, b.g2, b.b2, d.xn2.b, k, D, d.eps),
			b.w1.record(r, d.xn2.b, d.hid.b, k),
			r.AddBias(d.hid.b, b.fb1, d.hid.b, k, d.ffnDim(b)),
			r.Unary(d.hid.b, d.hid.b, unaryGELU),
			b.w2.record(r, d.hid.b, d.mo.b, k),
			r.AddBias(d.mo.b, b.fb2, d.mo.b, k, D),
			r.Binary(d.dx.b, d.mo.b, d.dx.b, binaryAdd),
		)
		if e != nil {
			r.Free()
			return nil, e
		}
	}
	if e := firstErr(
		r.LayerNorm(d.dx.b, d.lnfG.b, d.lnfB.b, d.xn.b, k, D, d.eps),
		d.head.record(r, d.xn.b, d.logits.b, k),
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

// ffnDim returns the block's FFN width from its up-projection.
func (d *GPTDecoder) ffnDim(b gptBlock) int {
	if l, ok := b.w1.(f32Linear); ok {
		return l.n
	}
	return 4 * d.d
}

// Generate prefills the whole prompt in ONE recorded StepN (§T423), then samples up to maxNew
// tokens (bounded by Ctx), returning prompt+generated ids.
func (d *GPTDecoder) Generate(prompt []int, maxNew int, s nlp.TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): gpt Generate needs a non-empty prompt", d.ops.name)
	}
	out := append([]int(nil), prompt...)
	all, err := d.StepN(prompt, 0)
	if err != nil {
		return nil, err
	}
	pos := len(prompt)
	logits := all[(len(prompt)-1)*d.v:]
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
func (d *GPTDecoder) Vocab() int { return d.v }

// Ctx returns the model's maximum context length (the KV-cache capacity).
func (d *GPTDecoder) Ctx() int { return d.maxLen }

// Release frees all device buffers.
func (d *GPTDecoder) Release() {
	for _, b := range d.all {
		if b != nil {
			b.Release()
		}
	}
	d.all = nil
}
