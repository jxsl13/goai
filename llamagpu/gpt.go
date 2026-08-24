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
	g1, b1, g2, b2 buffer    // LN1/LN2 gamma+beta [d]
	wq, wk, wv, wo linear    // attention projections [d,d], no bias
	wqkv           f32Linear // fused attention projection [d,3d]
	w1, w2         linear    // FFN up [d,4d] / down [4d,d]
	fb1, fb2       buffer    // FFN biases [4d] / [d]
	kC, vC         buffer    // KV caches [maxLen*d]
	ffn            int       // FFN width, fixed when the block is uploaded
}

// GPTDecoder is the batched-decode engine for nlp.GPT models (§T422) — the GPT-2-style sibling of
// [Decoder]: LayerNorm instead of RMSNorm, learned positional embeddings instead of RoPE, a biased
// GELU MLP instead of SwiGLU. Each Step records the whole layer stack into one command buffer.
// Not safe for concurrent use; Release when done.
type GPTDecoder struct {
	ops                 backendOps
	d, h, dk, v, maxLen int
	ffn                 int
	eps, scale          float32

	blocks         []gptBlock
	lnfG, lnfB     *bufSlot
	head           linear
	dx, xn, xn2    *bufSlot
	q, k, v_, qkv  *bufSlot
	attn, hid      *bufSlot
	logits         *bufSlot
	logitsRows     int
	fullLogits     growBuffer
	scratchRows    int
	fullScratch    gptScratch
	tokEmb, posEmb *tensor.Tensor // host gather sources
	embedHost      []float32      // reusable single-token token+position embedding row [d]
	embedBatchHost []float32      // reusable batched embedding rows, retained at the exact high-water size
	// eagerBoundaryControl retains the historical per-call host allocations for same-binary benchmarks.
	eagerBoundaryControl bool
	// eagerGenerateResultControl retains the historical per-token logits allocations for same-binary benchmarks.
	eagerGenerateResultControl bool
	all                        []buffer
}

type f32QKVBandsRecorder interface {
	F32QKVBands(x, w, q, k, v buffer, rows, dim int) error
}

type biasGELURecorder interface {
	BiasGELU(x, b, o buffer, rows, n int) error
}

// gptScratch is one generation of transient activation storage. The decoder owns one resident
// generation for Step and, after the first multi-row call, at most one reusable high-water
// generation for StepN/StepNLast.
type gptScratch struct {
	rows                       int
	dx, xn, xn2, q, k, v_, qkv buffer
	attn, hid                  buffer
}

func (s *gptScratch) release() {
	for _, b := range []buffer{s.dx, s.xn, s.xn2, s.q, s.k, s.v_, s.qkv, s.attn, s.hid} {
		if b != nil {
			b.Release()
		}
	}
	*s = gptScratch{}
}

func (d *GPTDecoder) residentScratch() gptScratch {
	var qkv buffer
	if d.qkv != nil {
		qkv = d.qkv.b
	}
	return gptScratch{
		rows: d.scratchRows,
		dx:   d.dx.b, xn: d.xn.b, xn2: d.xn2.b,
		q: d.q.b, k: d.k.b, v_: d.v_.b,
		attn: d.attn.b, hid: d.hid.b,
		qkv: qkv,
	}
}

func (d *GPTDecoder) newScratch(rows int) (gptScratch, error) {
	s := gptScratch{rows: rows}
	var err error
	alloc := func(n int) buffer {
		if err != nil {
			return nil
		}
		var b buffer
		b, err = d.ops.newBuffer(make([]float32, n))
		return b
	}
	s.dx = alloc(rows * d.d)
	s.xn = alloc(rows * d.d)
	s.xn2 = alloc(rows * d.d)
	s.q = alloc(rows * d.d)
	s.k = alloc(rows * d.d)
	s.v_ = alloc(rows * d.d)
	if d.ops.fusedF32QKV {
		s.qkv = alloc(min(rows, 63) * 3 * d.d)
	}
	s.attn = alloc(rows * d.d)
	s.hid = alloc(rows * d.ffn)
	if err != nil {
		s.release()
		return gptScratch{}, err
	}
	return s, nil
}

func (d *GPTDecoder) scratchForRows(rows int) (gptScratch, error) {
	if rows <= d.scratchRows {
		return d.residentScratch(), nil
	}
	if d.fullScratch.rows >= rows {
		return d.fullScratch, nil
	}
	d.fullScratch.release()
	s, err := d.newScratch(rows)
	if err != nil {
		return gptScratch{}, err
	}
	d.fullScratch = s
	return s, nil
}

func (d *GPTDecoder) scratchElements(s gptScratch) int {
	n := s.rows * (7*d.d + d.ffn)
	if s.qkv != nil {
		n += min(s.rows, 63) * 3 * d.d
	}
	return n
}

func (d *GPTDecoder) recordAttentionResidual(r recorder, b gptBlock, s gptScratch, rows int) error {
	return b.wo.recordAdd(r, s.attn, nil, s.dx, rows, d.d)
}

func (d *GPTDecoder) recordFFNResidual(r recorder, b gptBlock, s gptScratch, rows int) error {
	return firstErr(
		b.w2.recordAdd(r, s.hid, nil, s.dx, rows, d.d),
		r.AddBias(s.dx, b.fb2, s.dx, rows, d.d),
	)
}

func (d *GPTDecoder) recordBiasGELU(r recorder, b gptBlock, s gptScratch, rows int) error {
	if d.ops.fusedBiasGELU {
		if fused, ok := r.(biasGELURecorder); ok {
			return fused.BiasGELU(s.hid, b.fb1, s.hid, rows, b.ffn)
		}
	}
	return firstErr(
		r.AddBias(s.hid, b.fb1, s.hid, rows, b.ffn),
		r.Unary(s.hid, s.hid, unaryGELU),
	)
}

// newGPTDecoder uploads m's weights via ops and prepares per-block KV caches up to Ctx.
func newGPTDecoder(m *nlp.GPT, ops backendOps) (*GPTDecoder, error) {
	cfg := m.Config
	d := &GPTDecoder{
		ops: ops,
		d:   cfg.Dim, h: cfg.Heads, v: cfg.Vocab, maxLen: cfg.Ctx,
		eps: float32(cfg.Eps), tokEmb: m.TokEmb, posEmb: m.PosEmb,
		embedHost: make([]float32, cfg.Dim),
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
	fusedQKV := func(wq, wk, wv *tensor.Tensor) f32Linear {
		in := wq.Shape()[0]
		nq, nk, nv := wq.Shape()[1], wk.Shape()[1], wv.Shape()[1]
		fq, fk, fv := flat2D(wq), flat2D(wk), flat2D(wv)
		stride := nq + nk + nv
		weight := make([]float32, in*stride)
		for row := range in {
			dst := weight[row*stride : (row+1)*stride]
			copy(dst[:nq], fq[row*nq:(row+1)*nq])
			copy(dst[nq:nq+nk], fk[row*nk:(row+1)*nk])
			copy(dst[nq+nk:], fv[row*nv:(row+1)*nv])
		}
		return f32Linear{w: mk(weight).b, k: in, n: stride}
	}
	for _, b := range m.Blocks {
		gb := gptBlock{
			g1: mk(flat1D(b.LN1.Gamma)).b, b1: mk(flat1D(b.LN1.Beta)).b,
			g2: mk(flat1D(b.LN2.Gamma)).b, b2: mk(flat1D(b.LN2.Beta)).b,
			wo: lin(b.Attn.Wo), w1: lin(b.W1), w2: lin(b.W2),
			fb1: mk(flat1D(b.B1)).b, fb2: mk(flat1D(b.B2)).b,
			kC: mk(make([]float32, d.maxLen*d.d)).b, vC: mk(make([]float32, d.maxLen*d.d)).b,
			ffn: b.W1.Shape()[1],
		}
		if ops.fusedF32QKV {
			gb.wqkv = fusedQKV(b.Attn.Wq, b.Attn.Wk, b.Attn.Wv)
		} else {
			gb.wq, gb.wk, gb.wv = lin(b.Attn.Wq), lin(b.Attn.Wk), lin(b.Attn.Wv)
		}
		d.blocks = append(d.blocks, gb)
	}
	d.lnfG = mk(flat1D(m.LNf.Gamma))
	d.lnfB = mk(flat1D(m.LNf.Beta))
	d.head = lin(m.Head)
	d.ffn = 4 * d.d
	if len(m.Blocks) > 0 {
		d.ffn = m.Blocks[0].W1.Shape()[1]
	}
	d.scratchRows = 1
	if ops.eagerFullGPTScratch {
		d.scratchRows = d.maxLen
	}
	d.dx = mk(make([]float32, d.scratchRows*d.d))
	d.xn = mk(make([]float32, d.scratchRows*d.d))
	d.xn2 = mk(make([]float32, d.scratchRows*d.d))
	d.q = mk(make([]float32, d.scratchRows*d.d))
	d.k = mk(make([]float32, d.scratchRows*d.d))
	d.v_ = mk(make([]float32, d.scratchRows*d.d))
	if ops.fusedF32QKV {
		// Large prefills address bands of the single fused weight directly. Only rows below the
		// strided-band crossover need grouped output, so every workspace bounds it to 63 rows.
		d.qkv = mk(make([]float32, min(d.scratchRows, 63)*3*d.d))
	}
	d.attn = mk(make([]float32, d.scratchRows*d.d))
	d.hid = mk(make([]float32, d.scratchRows*d.ffn))
	d.logitsRows = 1
	if ops.eagerFullLogits {
		d.logitsRows = d.maxLen
	}
	d.logits = mk(make([]float32, d.logitsRows*d.v))
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

func addEmbedRowInto(dst []float32, table *tensor.Tensor, row, cols int) {
	if table.IsContiguous() {
		off := table.Offset() + row*cols
		switch table.Dtype() {
		case tensor.F32:
			for j, v := range table.Storage().F32()[off : off+cols] {
				dst[j] += v
			}
			return
		case tensor.F64:
			for j, v := range table.Storage().F64()[off : off+cols] {
				dst[j] += float32(v)
			}
			return
		}
	}
	for j := range cols {
		dst[j] += float32(table.AtF64(row, j))
	}
}

func (d *GPTDecoder) gatherEmbedInto(dst []float32, token, pos int) {
	embedRowInto(dst, d.tokEmb, token, d.d)
	addEmbedRowInto(dst, d.posEmb, pos, d.d)
}

func (d *GPTDecoder) batchEmbedHost(rows int) []float32 {
	need := rows * d.d
	if len(d.embedBatchHost) < need {
		d.embedBatchHost = make([]float32, need)
	}
	return d.embedBatchHost[:need]
}

// stepInto records one token at absolute position pos and leaves its logits resident.
func (d *GPTDecoder) stepInto(token, pos int) error {
	if pos < 0 || pos >= d.maxLen {
		return fmt.Errorf("llamagpu(%s): gpt pos %d out of [0,%d)", d.ops.name, pos, d.maxLen)
	}
	if token < 0 || token >= d.v {
		return fmt.Errorf("llamagpu(%s): gpt token %d out of vocab %d", d.ops.name, token, d.v)
	}
	x := d.embedHost
	if d.eagerBoundaryControl {
		x = embedRow(d.tokEmb, token, d.d)
		pe := embedRow(d.posEmb, pos, d.d)
		for i := range x {
			x[i] += pe[i]
		}
	} else {
		d.gatherEmbedInto(x, token, pos)
	}
	s := d.residentScratch()
	if err := s.dx.UploadF32(x); err != nil {
		return err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return err
	}
	D, H, dk := d.d, d.h, d.dk
	for _, b := range d.blocks {
		e := r.LayerNorm(s.dx, b.g1, b.b1, s.xn, 1, D, d.eps)
		qBuf := s.q
		if d.ops.fusedF32QKV {
			qBuf = s.qkv
			e = firstErr(e,
				b.wqkv.record(r, s.xn, s.qkv, 1),
				r.Blit(s.qkv, D, b.kC, pos*D, D),
				r.Blit(s.qkv, 2*D, b.vC, pos*D, D),
			)
		} else {
			e = firstErr(e,
				b.wq.record(r, s.xn, s.q, 1),
				b.wk.record(r, s.xn, s.k, 1),
				b.wv.record(r, s.xn, s.v_, 1),
				r.Blit(s.k, 0, b.kC, pos*D, D),
				r.Blit(s.v_, 0, b.vC, pos*D, D),
			)
		}
		e = firstErr(e,
			r.MHA(qBuf, b.kC, b.vC, s.attn, 1, pos+1, D, H, H, dk, 1, 0, d.scale),
			d.recordAttentionResidual(r, b, s, 1),
			r.LayerNorm(s.dx, b.g2, b.b2, s.xn2, 1, D, d.eps),
			b.w1.record(r, s.xn2, s.hid, 1),
			d.recordBiasGELU(r, b, s, 1),
			d.recordFFNResidual(r, b, s, 1),
		)
		if e != nil {
			r.Free()
			return e
		}
	}
	if e := firstErr(
		r.LayerNorm(s.dx, d.lnfG.b, d.lnfB.b, s.xn, 1, D, d.eps),
		d.head.record(r, s.xn, d.logits.b, 1),
		r.Finish(),
	); e != nil {
		r.Free()
		return e
	}
	r.Free()
	return nil
}

// StepInto advances the decoder by one token and writes exactly Vocab logits into caller-owned out.
// Its length is validated before any cache mutation.
func (d *GPTDecoder) StepInto(token, pos int, out []float32) error {
	if len(out) != d.v {
		return fmt.Errorf("llamagpu(%s): gpt StepInto output length %d != vocab %d", d.ops.name, len(out), d.v)
	}
	if err := d.stepInto(token, pos); err != nil {
		return err
	}
	return d.logits.b.DownloadF32(out)
}

// Step advances the decoder by one token at absolute position pos and returns newly allocated logits.
func (d *GPTDecoder) Step(token, pos int) ([]float32, error) {
	if err := d.stepInto(token, pos); err != nil {
		return nil, err
	}
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
	return d.gptStepN(tokens, pos, false)
}

// StepNInto is StepN with exactly len(tokens)*Vocab caller-owned output elements.
func (d *GPTDecoder) StepNInto(tokens []int, pos int, out []float32) error {
	want := len(tokens) * d.v
	if len(out) != want {
		return fmt.Errorf("llamagpu(%s): gpt StepNInto output length %d != tokens*vocab %d", d.ops.name, len(out), want)
	}
	return d.gptStepNInto(tokens, pos, false, out)
}

// StepNLast is StepN projecting only the FINAL prompt row's logits (LM head over 1 row, download of
// vocab not k·vocab); the transformer body still runs all k rows so the KV cache and the returned
// [vocab] row are bit-identical to StepN's last row. For generation/KV-seeding prefills that read at
// most the last row — mirrors Decoder.StepNLast.
func (d *GPTDecoder) StepNLast(tokens []int, pos int) ([]float32, error) {
	return d.gptStepN(tokens, pos, true)
}

// StepNLastInto is StepNLast with exactly Vocab caller-owned output elements.
func (d *GPTDecoder) StepNLastInto(tokens []int, pos int, out []float32) error {
	if len(out) != d.v {
		return fmt.Errorf("llamagpu(%s): gpt StepNLastInto output length %d != vocab %d", d.ops.name, len(out), d.v)
	}
	return d.gptStepNInto(tokens, pos, true, out)
}

func (d *GPTDecoder) gptStepN(tokens []int, pos int, lastOnly bool) ([]float32, error) {
	if err := d.validateStepN(tokens, pos); err != nil {
		return nil, err
	}
	rows := len(tokens)
	if lastOnly {
		rows = 1
	}
	out := make([]float32, rows*d.v)
	if err := d.gptStepNInto(tokens, pos, lastOnly, out); err != nil {
		return nil, err
	}
	return out, nil
}

func (d *GPTDecoder) validateStepN(tokens []int, pos int) error {
	k := len(tokens)
	if k == 0 {
		return fmt.Errorf("llamagpu(%s): gpt StepN needs ≥1 token", d.ops.name)
	}
	if pos < 0 || pos+k > d.maxLen {
		return fmt.Errorf("llamagpu(%s): gpt StepN [%d,%d) out of [0,%d)", d.ops.name, pos, pos+k, d.maxLen)
	}
	for _, tok := range tokens {
		if tok < 0 || tok >= d.v {
			return fmt.Errorf("llamagpu(%s): gpt token %d out of vocab %d", d.ops.name, tok, d.v)
		}
	}
	return nil
}

func (d *GPTDecoder) gptStepNInto(tokens []int, pos int, lastOnly bool, out []float32) error {
	k := len(tokens)
	if err := d.validateStepN(tokens, pos); err != nil {
		return err
	}
	logits := d.logits.b
	if !lastOnly {
		var err error
		logits, err = logitsForRows(d.ops, d.logits.b, d.logitsRows, &d.fullLogits, k, d.v)
		if err != nil {
			return err
		}
	}
	s, err := d.scratchForRows(k)
	if err != nil {
		return err
	}
	host := d.batchEmbedHost(k)
	if d.eagerBoundaryControl {
		host = make([]float32, k*d.d)
		for i, tok := range tokens {
			row := host[i*d.d : (i+1)*d.d]
			copy(row, embedRow(d.tokEmb, tok, d.d))
			pe := embedRow(d.posEmb, pos+i, d.d)
			for j := range row {
				row[j] += pe[j]
			}
		}
	} else {
		for i, tok := range tokens {
			d.gatherEmbedInto(host[i*d.d:(i+1)*d.d], tok, pos+i)
		}
	}
	if err := s.dx.UploadF32(host); err != nil {
		return err
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return err
	}
	D, H, dk := d.d, d.h, d.dk
	for _, b := range d.blocks {
		e := r.LayerNorm(s.dx, b.g1, b.b1, s.xn, k, D, d.eps)
		if d.ops.fusedF32QKV {
			if bands, ok := r.(f32QKVBandsRecorder); ok && k >= 64 {
				e = firstErr(e, bands.F32QKVBands(s.xn, b.wqkv.w, s.q, s.k, s.v_, k, D))
			} else {
				e = firstErr(e,
					b.wqkv.record(r, s.xn, s.qkv, k),
					r.Copy2D(s.qkv, 0, 3*D, s.q, 0, D, k, D),
					r.Copy2D(s.qkv, D, 3*D, s.k, 0, D, k, D),
					r.Copy2D(s.qkv, 2*D, 3*D, s.v_, 0, D, k, D),
				)
			}
		} else {
			e = firstErr(e,
				b.wq.record(r, s.xn, s.q, k),
				b.wk.record(r, s.xn, s.k, k),
				b.wv.record(r, s.xn, s.v_, k),
			)
		}
		e = firstErr(e,
			r.Blit(s.k, 0, b.kC, pos*D, k*D),
			r.Blit(s.v_, 0, b.vC, pos*D, k*D),
			r.MHA(s.q, b.kC, b.vC, s.attn, k, pos+k, D, H, H, dk, 1, 0, d.scale),
			d.recordAttentionResidual(r, b, s, k),
			r.LayerNorm(s.dx, b.g2, b.b2, s.xn2, k, D, d.eps),
			b.w1.record(r, s.xn2, s.hid, k),
			d.recordBiasGELU(r, b, s, k),
			d.recordFFNResidual(r, b, s, k),
		)
		if e != nil {
			r.Free()
			return e
		}
	}
	rows := k
	if lastOnly {
		// Only the final row's logits are wanted: move its residual to row 0 so the final LN + LM head
		// project ONE row and the download shrinks from k·vocab to vocab (bit-identical — row 0 then
		// holds exactly what row k-1 would have produced; a no-op when k==1).
		if e := r.Blit(s.dx, (k-1)*D, s.dx, 0, D); e != nil {
			r.Free()
			return e
		}
		rows = 1
	}
	if e := firstErr(
		r.LayerNorm(s.dx, d.lnfG.b, d.lnfB.b, s.xn, rows, D, d.eps),
		d.head.record(r, s.xn, logits, rows),
		r.Finish(),
	); e != nil {
		r.Free()
		return e
	}
	r.Free()
	if err := logits.DownloadF32(out); err != nil {
		return err
	}
	return nil
}

// Generate prefills the whole prompt in ONE recorded StepN (§T423), then samples up to maxNew
// tokens (bounded by Ctx), returning prompt+generated ids.
func (d *GPTDecoder) Generate(prompt []int, maxNew int, s nlp.TokenSampler) ([]int, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): gpt Generate needs a non-empty prompt", d.ops.name)
	}
	out := append([]int(nil), prompt...)
	// generation samples only the post-prompt row, so project just the last prompt row's logits
	// (StepNLast) instead of the full [len(prompt), vocab] LM head + download.
	var logits []float32
	var err error
	if d.eagerGenerateResultControl {
		logits, err = d.StepNLast(prompt, 0)
	} else {
		logits = make([]float32, d.v)
		err = d.StepNLastInto(prompt, 0, logits)
	}
	if err != nil {
		return nil, err
	}
	pos := len(prompt)
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
		if d.eagerGenerateResultControl {
			logits, err = d.Step(next, pos)
		} else {
			err = d.StepInto(next, pos, logits)
		}
		if err != nil {
			return nil, err
		}
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
	d.fullScratch.release()
	d.fullLogits.release()
	d.embedHost = nil
	d.embedBatchHost = nil
	for _, b := range d.all {
		if b != nil {
			b.Release()
		}
	}
	d.all = nil
}
