package llamagpu

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GPUT5 is a device-resident T5 bidirectional encoder — the encoder half of the text-to-text
// transformer, and the second non-decoder GPU model (after GPUBert). T5 departs from BERT in ways
// that reuse existing ops plus the one T5-specific primitive (MHABias): PRE-LN residuals (norm the
// input, add the sublayer output to the raw residual), RMSNorm (T5LayerNorm — no mean, no bias), NO
// 1/√d attention scaling, NO absolute/rotary position (a learned per-head RELATIVE-position bias is
// added to the scores instead), a gated-GELU (v1.1) or ReLU (v1.0) FFN, and NO bias in any
// projection. It returns the [seq, dim] hidden states (T5EncoderModel's last_hidden_state).
type GPUT5 struct {
	ops                      backendOps
	dim, heads, headDim, ffn int
	attWidth                 int // heads·headDim (may differ from dim — T5's d_kv is independent)
	eps                      float32
	vocab                    int
	shared                   *tensor.Tensor     // host token embedding (gathered per Forward; tied)
	relBias                  *nn.T5RelativeBias // shared relative-position bias source (host-computed)
	finalNorm                buffer
	blocks                   []t5Block
	all                      []buffer
}

type t5Block struct {
	attnNorm       buffer
	wq, wk, wv, wo linear
	ffnNorm        buffer
	wi0, wOut      linear
	wi1            linear // gated-GELU up projection (v1.1); nil for v1.0 ReLU
	gated          bool
}

// newT5Encoder uploads an nlp.T5's weights via ops into device buffers. cuda constructor: NewT5CUDA.
func newT5Encoder(m *nlp.T5, ops backendOps) (*GPUT5, error) {
	cfg := m.Config
	if len(m.Blocks) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): T5 has no blocks", ops.name)
	}
	e := &GPUT5{
		ops: ops, dim: cfg.Dim, heads: cfg.Heads, headDim: cfg.HeadDim, ffn: cfg.FFN,
		attWidth: cfg.Heads * cfg.HeadDim, eps: float32(cfg.Eps), vocab: cfg.Vocab,
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
		e.all = append(e.all, b)
		return b
	}
	lin := func(w *tensor.Tensor) linear {
		in, out := w.Shape()[0], w.Shape()[1]
		return f32Linear{w: mk(flat2D(w)), k: in, n: out}
	}
	for _, b := range m.Blocks {
		gb := t5Block{
			attnNorm: mk(flat1D(b.AttnNorm.Gamma)),
			wq:       lin(b.Wq), wk: lin(b.Wk), wv: lin(b.Wv), wo: lin(b.Wo),
			ffnNorm: mk(flat1D(b.FFNNorm.Gamma)),
			wi0:     lin(b.Wi0), wOut: lin(b.WOut),
		}
		if b.Wi1 != nil {
			gb.wi1, gb.gated = lin(b.Wi1), true
		}
		e.blocks = append(e.blocks, gb)
	}
	e.finalNorm = mk(flat1D(m.FinalNorm.Gamma))
	if err != nil {
		e.Release()
		return nil, err
	}
	return e, nil
}

// relPosBias computes the shared [heads·seq·seq] relative-position bias on the host (via the same
// nn.T5RelativeBias the reference uses), laid out identically to the internal scores buffer
// [heads·seq, seq] — element (h,i,j) at h·seq·seq + i·seq + j.
func (e *GPUT5) relPosBias(seq int) ([]float32, error) {
	ctx := backend.NewContext().WithBackend(backend.Reference())
	bias, err := e.relBias.Bias(ctx, seq, seq) // [seq, seq, heads]
	if err != nil {
		return nil, err
	}
	if bias, err = bias.Permute(2, 0, 1); err != nil { // [heads, seq, seq]
		return nil, err
	}
	bias = bias.Contiguous()
	out := make([]float32, e.heads*seq*seq)
	for h := 0; h < e.heads; h++ {
		for i := 0; i < seq; i++ {
			for j := 0; j < seq; j++ {
				out[h*seq*seq+i*seq+j] = float32(bias.AtF64(h, i, j))
			}
		}
	}
	return out, nil
}

// Forward runs the encoder over tokens and returns the last hidden state [seq, dim] flattened
// row-major. One recorded command buffer: per block a PRE-LN self-attention sublayer (RMSNorm →
// q/k/v → MHABias with the shared relpos bias, unscaled → o·Wo → residual add onto the raw stream)
// and a PRE-LN gated-GELU/ReLU FFN sublayer, then a final RMSNorm.
func (e *GPUT5) Forward(tokens []int) ([]float32, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("llamagpu(%s): T5 empty token sequence", e.ops.name)
	}
	emb := make([]float32, seq*e.dim)
	for i, tok := range tokens {
		if tok < 0 || tok >= e.vocab {
			return nil, fmt.Errorf("llamagpu(%s): token %d outside vocab %d", e.ops.name, tok, e.vocab)
		}
		for j := 0; j < e.dim; j++ {
			emb[i*e.dim+j] = float32(e.shared.AtF64(tok, j))
		}
	}
	biasHost, err := e.relPosBias(seq)
	if err != nil {
		return nil, err
	}

	r, err := e.ops.newRecorder()
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
		b, e2 := e.ops.newBuffer(make([]float32, n))
		if e2 != nil {
			se = e2
			return nil
		}
		scratch = append(scratch, b)
		return b
	}
	x := sb(seq * e.dim)
	hn := sb(seq * e.dim)
	q := sb(seq * e.attWidth)
	k := sb(seq * e.attWidth)
	v := sb(seq * e.attWidth)
	attn := sb(seq * e.attWidth)
	ao := sb(seq * e.dim)
	g := sb(seq * e.ffn)
	u := sb(seq * e.ffn)
	ff := sb(seq * e.dim)
	dbias := sb(e.heads * seq * seq)
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

	if err := firstErr(x.UploadF32(emb), dbias.UploadF32(biasHost)); err != nil {
		return nil, err
	}
	for _, b := range e.blocks {
		// PRE-LN self-attention: hn = RMSNorm(x); q/k/v = hn·W; MHABias (unscaled, +relpos); ao =
		// attn·Wo; x += ao (residual onto the raw stream).
		err = firstErr(err,
			r.RMSNorm(x, b.attnNorm, hn, seq, e.dim, e.eps),
			b.wq.record(r, hn, q, seq), b.wk.record(r, hn, k, seq), b.wv.record(r, hn, v, seq),
			r.MHABias(q, k, v, attn, dbias, seq, seq, e.attWidth, e.heads, e.heads, e.headDim, 0, 0, 1),
			b.wo.record(r, attn, ao, seq),
			r.Binary(x, ao, x, binaryAdd),
			// PRE-LN FFN: hn = RMSNorm(x); gated-GELU (v1.1) or ReLU (v1.0); x += ff·WOut.
			r.RMSNorm(x, b.ffnNorm, hn, seq, e.dim, e.eps),
		)
		if b.gated {
			err = firstErr(err,
				b.wi0.record(r, hn, g, seq), r.Unary(g, g, unaryGELU),
				b.wi1.record(r, hn, u, seq),
				r.Binary(g, u, g, binaryMul), // gelu(hn·Wi0) ⊙ (hn·Wi1)
				b.wOut.record(r, g, ff, seq),
			)
		} else {
			err = firstErr(err,
				b.wi0.record(r, hn, g, seq), r.Unary(g, g, unaryReLU),
				b.wOut.record(r, g, ff, seq),
			)
		}
		err = firstErr(err, r.Binary(x, ff, x, binaryAdd))
		if err != nil {
			return nil, err
		}
	}
	err = firstErr(err, r.RMSNorm(x, e.finalNorm, x, seq, e.dim, e.eps), r.Commit(), r.Wait())
	if err != nil {
		return nil, err
	}
	out := make([]float32, seq*e.dim)
	if err := x.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Release frees every device buffer held by the encoder.
func (e *GPUT5) Release() {
	for _, b := range e.all {
		if b != nil {
			b.Release()
		}
	}
	e.all = nil
}
