package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GPUT5Decoder is the device-resident T5 seq2seq (encoder-decoder) DECODER — the last architecture in
// the nlp catalogue to reach the GPU, and the first encoder-decoder model. Each block has THREE
// sublayers (all PRE-LN, RMSNorm, no bias): causal self-attention with the T5 relative-position bias
// (via MHABias over a growing self-KV cache), cross-attention over the fixed encoder output (plain
// MHA, no bias/relpos), and a gated-GELU/ReLU FFN. Attention is UNSCALED (T5 folds 1/√d into init).
// Pair it with a GPUT5 encoder: run the encoder, feed its [eseq, dim] hidden states to Decode.
type GPUT5Decoder struct {
	ops                             backendOps
	dim, heads, headDim, ffn, vocab int
	attWidth                        int
	eps                             float32
	shared                          *tensor.Tensor     // host token embedding (tied)
	relBias                         *nn.T5RelativeBias // causal relative-position bias (host-computed)
	finalNorm                       buffer
	lmHead                          linear // [dim, vocab]
	blocks                          []t5DecBlock
	all                             []buffer
}

type t5DecBlock struct {
	selfNorm           buffer
	sWq, sWk, sWv, sWo linear
	crossNorm          buffer
	cWq, cWk, cWv, cWo linear
	ffnNorm            buffer
	wi0, wOut          linear
	wi1                linear
	gated              bool
}

// newT5Decoder uploads an nlp.T5Decoder's weights via ops into device buffers. cuda: NewT5DecoderCUDA.
func newT5Decoder(m *nlp.T5Decoder, ops backendOps) (*GPUT5Decoder, error) {
	cfg := m.Config
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): T5Decoder has no blocks", ops.name)
	}
	if m.LMHead == nil {
		return nil, fmt.Errorf("llamagpu(%s): T5Decoder has no LM head", ops.name)
	}
	// Derive dims from the weight shapes — T5DecoderFromHF only sets Heads/HeadDim/Eps in Config, not
	// Dim/FFN/Vocab. Shared is [vocab, dim]; SWq is [dim, heads·headDim]; Wi0 is [dim, ffn].
	b0 := m.Blocks[0]
	d := &GPUT5Decoder{
		ops: ops, dim: m.Shared.Shape()[1], heads: cfg.Heads, headDim: cfg.HeadDim,
		ffn: b0.Wi0.Shape()[1], vocab: m.Shared.Shape()[0], attWidth: b0.SWq.Shape()[1],
		eps:    float32(cfg.Eps),
		shared: m.Shared, relBias: m.RelBias,
	}
	var err error
	mk := func(data []float32) buffer {
		if err != nil {
			return nil
		}
		b, e2 := ops.newBuffer(data)
		if e2 != nil {
			err = e2
			return nil
		}
		d.all = append(d.all, b)
		return b
	}
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)), k: in, n: out}
	}
	for _, b := range m.Blocks {
		gb := t5DecBlock{
			selfNorm: mk(flat1D(b.SelfNorm.Gamma)),
			sWq:      lin(b.SWq), sWk: lin(b.SWk), sWv: lin(b.SWv), sWo: lin(b.SWo),
			crossNorm: mk(flat1D(b.CrossNorm.Gamma)),
			cWq:       lin(b.CWq), cWk: lin(b.CWk), cWv: lin(b.CWv), cWo: lin(b.CWo),
			ffnNorm: mk(flat1D(b.FFNNorm.Gamma)),
			wi0:     lin(b.Wi0), wOut: lin(b.WOut),
		}
		if b.Wi1 != nil {
			gb.wi1, gb.gated = lin(b.Wi1), true
		}
		d.blocks = append(d.blocks, gb)
	}
	d.finalNorm = mk(flat1D(m.FinalNorm.Gamma))
	d.lmHead = lin(m.LMHead)
	if err != nil {
		d.Release()
		return nil, err
	}
	return d, nil
}

// selfBiasRow builds the [heads·(pos+1)] causal relative-position bias for a query at position pos
// attending to keys 0..pos — the same table nn.T5Decoder.relBiasRow uses, laid out as the scores
// buffer for sq=1 (element (h,k) at h·(pos+1)+k).
func (d *GPUT5Decoder) selfBiasRow(pos int) ([]float32, error) {
	ctx := backend.NewContext().WithBackend(backend.Reference())
	full, err := d.relBias.Bias(ctx, pos+1, pos+1) // [pos+1, pos+1, heads]
	if err != nil {
		return nil, err
	}
	fs := full.Contiguous().Storage().F64()
	kk := pos + 1
	out := make([]float32, d.heads*kk)
	for h := 0; h < d.heads; h++ {
		for k := 0; k < kk; k++ {
			out[h*kk+k] = float32(fs[(pos*kk+k)*d.heads+h]) // full[pos][k][h]
		}
	}
	return out, nil
}

// Decode runs the decoder over decTokens attending to the encoder output encOut ([eseq, dim] flattened
// row-major), returning the per-step logits [len(decTokens), vocab] flattened row-major. It mirrors
// nlp.T5Decoder.DecodeStep across the sequence: per token a causal self-attention over the growing KV
// cache (with the T5 relpos bias), a cross-attention over the once-projected encoder K/V, and a FFN.
func (d *GPUT5Decoder) Decode(encOut []float32, eseq int, decTokens []int) ([]float32, error) {
	dlen := len(decTokens)
	if dlen == 0 {
		return nil, fmt.Errorf("llamagpu(%s): T5 decode needs ≥1 token", d.ops.name)
	}
	if len(encOut) != eseq*d.dim {
		return nil, fmt.Errorf("llamagpu(%s): encOut %d != eseq·dim %d", d.ops.name, len(encOut), eseq*d.dim)
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	defer r.Free()

	var se error
	var scratch []buffer
	sb := func(n int) buffer {
		if se != nil {
			return nil
		}
		b, e2 := d.ops.newBuffer(make([]float32, n))
		if e2 != nil {
			se = e2
			return nil
		}
		scratch = append(scratch, b)
		return b
	}
	// resident encoder output + per-block cross K/V (projected once) and self K/V cache (grown to dlen).
	enc := sb(eseq * d.dim)
	nb := len(d.blocks)
	crossK := make([]buffer, nb)
	crossV := make([]buffer, nb)
	selfK := make([]buffer, nb)
	selfV := make([]buffer, nb)
	for l := 0; l < nb; l++ {
		crossK[l] = sb(eseq * d.attWidth)
		crossV[l] = sb(eseq * d.attWidth)
		selfK[l] = sb(dlen * d.attWidth)
		selfV[l] = sb(dlen * d.attWidth)
	}
	x := sb(d.dim)
	hn := sb(d.dim)
	q := sb(d.attWidth)
	kt := sb(d.attWidth)
	vt := sb(d.attWidth)
	attn := sb(d.attWidth)
	ao := sb(d.dim)
	g := sb(d.ffn)
	u := sb(d.ffn)
	ff := sb(d.dim)
	dbias := sb(d.heads * dlen) // max bias row [heads·(pos+1)] ≤ heads·dlen
	logit := sb(d.vocab)
	freeScratch := func() {
		for _, b := range scratch {
			if b != nil {
				b.Release()
			}
		}
	}
	if se != nil {
		freeScratch()
		return nil, se
	}
	defer freeScratch()

	if err := enc.UploadF32(encOut); err != nil {
		return nil, err
	}
	// Project the fixed encoder K/V once per block.
	for l, b := range d.blocks {
		if err := firstErr(b.cWk.record(r, enc, crossK[l], eseq), b.cWv.record(r, enc, crossV[l], eseq)); err != nil {
			return nil, err
		}
	}

	out := make([]float32, dlen*d.vocab)
	for i, tok := range decTokens {
		if tok < 0 || tok >= d.vocab {
			return nil, fmt.Errorf("llamagpu(%s): token %d outside vocab %d", d.ops.name, tok, d.vocab)
		}
		emb := make([]float32, d.dim)
		for j := 0; j < d.dim; j++ {
			emb[j] = float32(d.shared.AtF64(tok, j))
		}
		biasRow, err := d.selfBiasRow(i)
		if err != nil {
			return nil, err
		}
		if err := firstErr(x.UploadF32(emb), dbias.UploadF32(biasRow)); err != nil {
			return nil, err
		}
		for l, b := range d.blocks {
			// causal self-attention over the growing KV cache (write row i, attend 0..i).
			err = firstErr(
				r.RMSNorm(x, b.selfNorm, hn, 1, d.dim, d.eps),
				b.sWq.record(r, hn, q, 1), b.sWk.record(r, hn, kt, 1), b.sWv.record(r, hn, vt, 1),
				r.Blit(kt, 0, selfK[l], i*d.attWidth, d.attWidth),
				r.Blit(vt, 0, selfV[l], i*d.attWidth, d.attWidth),
				r.MHABias(q, selfK[l], selfV[l], attn, dbias, 1, i+1, d.attWidth, d.heads, d.heads, d.headDim, 0, 0, 1),
				b.sWo.record(r, attn, ao, 1),
				r.Binary(x, ao, x, binaryAdd),
				// cross-attention over the fixed encoder K/V (no bias, no mask).
				r.RMSNorm(x, b.crossNorm, hn, 1, d.dim, d.eps),
				b.cWq.record(r, hn, q, 1),
				r.MHA(q, crossK[l], crossV[l], attn, 1, eseq, d.attWidth, d.heads, d.heads, d.headDim, 0, 0, 1),
				b.cWo.record(r, attn, ao, 1),
				r.Binary(x, ao, x, binaryAdd),
				// PRE-LN FFN.
				r.RMSNorm(x, b.ffnNorm, hn, 1, d.dim, d.eps),
			)
			if b.gated {
				err = firstErr(err,
					b.wi0.record(r, hn, g, 1), r.Unary(g, g, unaryGELU),
					b.wi1.record(r, hn, u, 1),
					r.Binary(g, u, g, binaryMul),
					b.wOut.record(r, g, ff, 1),
				)
			} else {
				err = firstErr(err,
					b.wi0.record(r, hn, g, 1), r.Unary(g, g, unaryReLU),
					b.wOut.record(r, g, ff, 1),
				)
			}
			err = firstErr(err, r.Binary(x, ff, x, binaryAdd))
			if err != nil {
				return nil, err
			}
		}
		// final norm + LM head, then read this step's logits back.
		if err := firstErr(
			r.RMSNorm(x, d.finalNorm, hn, 1, d.dim, d.eps),
			d.lmHead.record(r, hn, logit, 1),
			r.Commit(), r.Wait(),
		); err != nil {
			return nil, err
		}
		if err := logit.DownloadF32(out[i*d.vocab : (i+1)*d.vocab]); err != nil {
			return nil, err
		}
	}
	return out, nil
}

// Release frees every device buffer held by the decoder.
func (d *GPUT5Decoder) Release() {
	for _, b := range d.all {
		if b != nil {
			b.Release()
		}
	}
	d.all = nil
}
