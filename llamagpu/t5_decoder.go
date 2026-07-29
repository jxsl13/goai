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
	qweights                        []qweight // resident Q8 projection weights (NewT5DecoderQ8CUDA); closed in Release
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
	// lin: resident Q8_0 when ops.quantizeF32 is set (NewT5DecoderQ8CUDA), else f32. The self/cross
	// attention, FFN and lm_head projections go Q8; RMSNorm and the relpos bias stay f32. Autoregressive
	// decode is M=1 (Q8 GEMV, a weight-bandwidth win); the cross K/V projections run once over the encoder
	// output (M=eseq>1, so the int8 tensor-core MMQ path).
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		if ops.quantizeF32 == nil {
			return f32Linear{w: mk(flat2D(w)), k: in, n: out}
		}
		t := tensor.New(tensor.F32, tensor.Shape{in, out})
		copy(t.Storage().F32(), flat2D(w))
		qw, qe := ops.quantizeF32(t)
		if qe != nil {
			if err == nil {
				err = qe
			}
			return f32Linear{k: in, n: out}
		}
		d.qweights = append(d.qweights, qw)
		return quantLinear{w: qw}
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
	kk := pos + 1
	// Build only the single query row: O(pos) instead of materializing the full O(pos²) bias
	// table and discarding all but row pos — bit-identical to Bias(pos+1,pos+1)[pos] (§perf).
	// Row layout is [kk, heads]: element (k,h) at k*heads+h.
	row, err := d.relBias.BiasRow(ctx, pos, kk)
	if err != nil {
		return nil, err
	}
	rs := row.Contiguous().Storage().F64()
	out := make([]float32, d.heads*kk)
	for h := 0; h < d.heads; h++ {
		for k := 0; k < kk; k++ {
			out[h*kk+k] = float32(rs[k*d.heads+h]) // row[k][h]
		}
	}
	return out, nil
}

// t5DecSession is one incremental-decoding run: a recorder, the resident encoder output + once-
// projected cross K/V, a growing self-KV cache (sized to maxSteps) and the per-step scratch. Both
// Decode (fixed tokens) and Generate (argmax feedback) drive it via step(token, pos).
type t5DecSession struct {
	d                                                  *GPUT5Decoder
	r                                                  recorder
	eseq                                               int
	scratch                                            []buffer
	crossK, crossV                                     []buffer
	selfK, selfV                                       []buffer
	x, hn, q, kt, vt, attn, ao, g, u, ff, dbias, logit buffer
}

// newSession allocates a decode session for up to maxSteps tokens and projects the fixed encoder K/V.
func (d *GPUT5Decoder) newSession(encOut []float32, eseq, maxSteps int) (*t5DecSession, error) {
	if len(encOut) != eseq*d.dim {
		return nil, fmt.Errorf("llamagpu(%s): encOut %d != eseq·dim %d", d.ops.name, len(encOut), eseq*d.dim)
	}
	r, err := d.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	s := &t5DecSession{d: d, r: r, eseq: eseq}
	var se error
	sb := func(n int) buffer {
		if se != nil {
			return nil
		}
		b, e2 := d.ops.newBuffer(make([]float32, n))
		if e2 != nil {
			se = e2
			return nil
		}
		s.scratch = append(s.scratch, b)
		return b
	}
	enc := sb(eseq * d.dim)
	nb := len(d.blocks)
	s.crossK, s.crossV = make([]buffer, nb), make([]buffer, nb)
	s.selfK, s.selfV = make([]buffer, nb), make([]buffer, nb)
	for l := 0; l < nb; l++ {
		s.crossK[l], s.crossV[l] = sb(eseq*d.attWidth), sb(eseq*d.attWidth)
		s.selfK[l], s.selfV[l] = sb(maxSteps*d.attWidth), sb(maxSteps*d.attWidth)
	}
	s.x, s.hn = sb(d.dim), sb(d.dim)
	s.q, s.kt, s.vt, s.attn = sb(d.attWidth), sb(d.attWidth), sb(d.attWidth), sb(d.attWidth)
	s.ao, s.ff = sb(d.dim), sb(d.dim)
	s.g, s.u = sb(d.ffn), sb(d.ffn)
	s.dbias = sb(d.heads * maxSteps)
	s.logit = sb(d.vocab)
	if se != nil {
		s.free()
		return nil, se
	}
	if err := enc.UploadF32(encOut); err != nil {
		s.free()
		return nil, err
	}
	for l, b := range d.blocks {
		if err := firstErr(b.cWk.record(r, enc, s.crossK[l], eseq), b.cWv.record(r, enc, s.crossV[l], eseq)); err != nil {
			s.free()
			return nil, err
		}
	}
	return s, nil
}

func (s *t5DecSession) free() {
	for _, b := range s.scratch {
		if b != nil {
			b.Release()
		}
	}
	if s.r != nil {
		s.r.Free()
	}
}

// step decodes one token at absolute position pos, updating the KV cache, and returns that step's
// logits [vocab]. It mirrors nlp.T5Decoder.DecodeStep: causal self-attn (relpos bias, growing cache)
// → cross-attn (fixed encoder K/V) → gated/ReLU FFN, all PRE-LN, then the final norm + LM head.
func (s *t5DecSession) step(token, pos int) ([]float32, error) {
	d, r := s.d, s.r
	if token < 0 || token >= d.vocab {
		return nil, fmt.Errorf("llamagpu(%s): token %d outside vocab %d", d.ops.name, token, d.vocab)
	}
	emb := make([]float32, d.dim)
	for j := 0; j < d.dim; j++ {
		emb[j] = float32(d.shared.AtF64(token, j))
	}
	biasRow, err := d.selfBiasRow(pos)
	if err != nil {
		return nil, err
	}
	if err := firstErr(s.x.UploadF32(emb), s.dbias.UploadF32(biasRow)); err != nil {
		return nil, err
	}
	for l, b := range d.blocks {
		err = firstErr(
			// causal self-attention over the growing KV cache (write row pos, attend 0..pos).
			r.RMSNorm(s.x, b.selfNorm, s.hn, 1, d.dim, d.eps),
			b.sWq.record(r, s.hn, s.q, 1), b.sWk.record(r, s.hn, s.kt, 1), b.sWv.record(r, s.hn, s.vt, 1),
			r.Blit(s.kt, 0, s.selfK[l], pos*d.attWidth, d.attWidth),
			r.Blit(s.vt, 0, s.selfV[l], pos*d.attWidth, d.attWidth),
			r.MHABias(s.q, s.selfK[l], s.selfV[l], s.attn, s.dbias, 1, pos+1, d.attWidth, d.heads, d.heads, d.headDim, 0, 0, 1),
			b.sWo.record(r, s.attn, s.ao, 1),
			r.Binary(s.x, s.ao, s.x, binaryAdd),
			// cross-attention over the fixed encoder K/V (no bias, no mask).
			r.RMSNorm(s.x, b.crossNorm, s.hn, 1, d.dim, d.eps),
			b.cWq.record(r, s.hn, s.q, 1),
			r.MHA(s.q, s.crossK[l], s.crossV[l], s.attn, 1, s.eseq, d.attWidth, d.heads, d.heads, d.headDim, 0, 0, 1),
			b.cWo.record(r, s.attn, s.ao, 1),
			r.Binary(s.x, s.ao, s.x, binaryAdd),
			// PRE-LN FFN.
			r.RMSNorm(s.x, b.ffnNorm, s.hn, 1, d.dim, d.eps),
		)
		if b.gated {
			err = firstErr(err,
				b.wi0.record(r, s.hn, s.g, 1), r.Unary(s.g, s.g, unaryGELU),
				b.wi1.record(r, s.hn, s.u, 1),
				r.Binary(s.g, s.u, s.g, binaryMul),
				b.wOut.record(r, s.g, s.ff, 1),
			)
		} else {
			err = firstErr(err,
				b.wi0.record(r, s.hn, s.g, 1), r.Unary(s.g, s.g, unaryReLU),
				b.wOut.record(r, s.g, s.ff, 1),
			)
		}
		err = firstErr(err, r.Binary(s.x, s.ff, s.x, binaryAdd))
		if err != nil {
			return nil, err
		}
	}
	out := make([]float32, d.vocab)
	if err := firstErr(
		r.RMSNorm(s.x, d.finalNorm, s.hn, 1, d.dim, d.eps),
		d.lmHead.record(r, s.hn, s.logit, 1),
		r.Commit(), r.Wait(),
	); err != nil {
		return nil, err
	}
	if err := s.logit.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Decode runs the decoder over decTokens attending to the encoder output encOut ([eseq, dim] flattened
// row-major), returning the per-step logits [len(decTokens), vocab] flattened row-major.
func (d *GPUT5Decoder) Decode(encOut []float32, eseq int, decTokens []int) ([]float32, error) {
	dlen := len(decTokens)
	if dlen == 0 {
		return nil, fmt.Errorf("llamagpu(%s): T5 decode needs ≥1 token", d.ops.name)
	}
	s, err := d.newSession(encOut, eseq, dlen)
	if err != nil {
		return nil, err
	}
	defer s.free()
	out := make([]float32, dlen*d.vocab)
	for i, tok := range decTokens {
		logits, err := s.step(tok, i)
		if err != nil {
			return nil, err
		}
		copy(out[i*d.vocab:(i+1)*d.vocab], logits)
	}
	return out, nil
}

// Generate greedily decodes up to maxNew tokens starting from startToken, attending to encOut, and
// returns the generated tokens (excluding startToken, stopping at eosToken). Greedy argmax — the
// seq2seq translation/summarization use case; sampled generation is a follow-up.
func (d *GPUT5Decoder) Generate(encOut []float32, eseq, maxNew, startToken, eosToken int) ([]int, error) {
	if maxNew <= 0 {
		return nil, fmt.Errorf("llamagpu(%s): maxNew must be > 0", d.ops.name)
	}
	s, err := d.newSession(encOut, eseq, maxNew)
	if err != nil {
		return nil, err
	}
	defer s.free()
	gen := []int{startToken}
	for pos := 0; pos < maxNew; pos++ {
		logits, err := s.step(gen[pos], pos)
		if err != nil {
			return nil, err
		}
		next, best := 0, logits[0]
		for v := 1; v < d.vocab; v++ {
			if logits[v] > best {
				next, best = v, logits[v]
			}
		}
		if next == eosToken {
			break
		}
		gen = append(gen, next)
	}
	return gen[1:], nil
}

// Release frees every device buffer held by the decoder.
func (d *GPUT5Decoder) Release() {
	for _, b := range d.all {
		if b != nil {
			b.Release()
		}
	}
	d.all = nil
	for _, w := range d.qweights {
		if w != nil {
			w.Close()
		}
	}
	d.qweights = nil
}
