package llamagpu

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// GPUBert is a device-resident BERT bidirectional encoder — the first non-decoder GPU model in
// llamagpu. Unlike the autoregressive Decoder (Step/StepN produce next-token logits over a growing
// KV cache), an encoder runs ONE bidirectional forward over the whole sequence and returns the
// contextualized hidden states [seq, dim]. It is post-LN (LayerNorm AFTER each residual add), attends
// with NO causal mask, and takes its position signal from a learned absolute position embedding —
// so it reuses the existing recorder ops (MHA with causal=0, LayerNorm, GELU MLP, AddBias) with no
// new kernel. Weights are uploaded once; Forward records the encoder into one command buffer.
type GPUBert struct {
	ops                    backendOps
	dim, heads, dk, inter  int
	maxPos, posOffset      int
	vocab, typeVocab       int
	eps, scale             float32
	tokEmb, posEmb, segEmb *tensor.Tensor // host-side embedding tables (gathered per Forward)
	embLNg, embLNb         buffer
	blocks                 []bertBlock
	all                    []buffer
	qweights               []qweight // resident Q8 projection weights (NewBertQ8CUDA); closed in Release
}

type bertBlock struct {
	wq, wk, wv, wo   linear
	bq, bk, bv, bo   buffer
	attnLNg, attnLNb buffer
	w1, w2           linear
	b1, b2           buffer
	ffnLNg, ffnLNb   buffer
}

// newBertEncoder uploads an nlp.Bert's weights via ops into device buffers. Backend-agnostic; the
// cuda constructor is NewBertCUDA.
func newBertEncoder(m *nlp.Bert, ops backendOps) (*GPUBert, error) {
	cfg := m.Config
	if len(m.Layers) == 0 {
		return nil, fmt.Errorf("llamagpu(%s): BERT has no layers", ops.name)
	}
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("llamagpu(%s): dim %d not divisible by heads %d", ops.name, cfg.Dim, cfg.Heads)
	}
	e := &GPUBert{
		ops: ops, dim: cfg.Dim, heads: cfg.Heads, dk: cfg.Dim / cfg.Heads,
		maxPos: cfg.MaxPos, posOffset: cfg.PosOffset, vocab: cfg.Vocab, typeVocab: cfg.TypeVocab,
		eps:    float32(cfg.Eps),
		tokEmb: m.TokEmb, posEmb: m.PosEmb, segEmb: m.SegEmb,
	}
	e.scale = float32(1.0 / math.Sqrt(float64(e.dk)))
	e.inter = m.Layers[0].W1.Shape()[1] // FFN intermediate width

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
	// lin builds a projection: resident Q8_0 when ops.quantizeF32 is set (NewBertQ8CUDA), else f32.
	// BERT encodes the whole sequence at once (M=seq>1), so a Q8 projection runs on the int8 tensor-core
	// MMQ GEMM (QMatMulResident's m>1 path) — a compute+memory win. Biases/LayerNorm stay f32.
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
		e.qweights = append(e.qweights, qw)
		return quantLinear{w: qw}
	}
	e.embLNg = mk(flat1D(m.EmbLN.Gamma))
	e.embLNb = mk(flat1D(m.EmbLN.Beta))
	for _, l := range m.Layers {
		gb := bertBlock{
			wq: lin(l.Attn.Wq), wk: lin(l.Attn.Wk), wv: lin(l.Attn.Wv), wo: lin(l.Attn.Wo),
			bq: mk(flat1D(l.Attn.Bias["q"])), bk: mk(flat1D(l.Attn.Bias["k"])),
			bv: mk(flat1D(l.Attn.Bias["v"])), bo: mk(flat1D(l.Attn.Bias["o"])),
			attnLNg: mk(flat1D(l.AttnLN.Gamma)), attnLNb: mk(flat1D(l.AttnLN.Beta)),
			w1: lin(l.W1), b1: mk(flat1D(l.B1)), w2: lin(l.W2), b2: mk(flat1D(l.B2)),
			ffnLNg: mk(flat1D(l.FFNLN.Gamma)), ffnLNb: mk(flat1D(l.FFNLN.Beta)),
		}
		e.blocks = append(e.blocks, gb)
	}
	if err != nil {
		e.Release()
		return nil, err
	}
	return e, nil
}

// summedEmbedding gathers word + position (+ optional segment) embeddings and sums them on the host —
// the encoder's input before EmbLN. Matches nlp.Bert.Embed. segments may be nil (all-zero).
func (e *GPUBert) summedEmbedding(tokens, segments []int) ([]float32, error) {
	seq := len(tokens)
	out := make([]float32, seq*e.dim)
	for i, tok := range tokens {
		if tok < 0 || tok >= e.vocab {
			return nil, fmt.Errorf("llamagpu(%s): token %d outside vocab %d", e.ops.name, tok, e.vocab)
		}
		pos := e.posOffset + i
		seg := 0
		if segments != nil {
			seg = segments[i]
		}
		// Bound the segment (token-type) id the same way the token is bounded one line above:
		// unchecked, it indexes segEmb below and panics inside the tensor, where the CPU reference
		// nlp.Bert.Embed returns a clean "ref: embed index %d out of range" for the identical input
		// (§V29 — a malformed input is an error, not a panic). Only models that carry a segment
		// table have a bound: DistilBERT has segEmb nil and typeVocab 0, and never reads seg.
		if e.segEmb != nil && (seg < 0 || seg >= e.typeVocab) {
			return nil, fmt.Errorf("llamagpu(%s): segment id %d outside token-type vocab %d", e.ops.name, seg, e.typeVocab)
		}
		row := out[i*e.dim : (i+1)*e.dim]
		//perfscan:ignore PS1005 embed gather O(s·d) « transformer forward, sub-1pct
		for j := 0; j < e.dim; j++ {
			v := e.tokEmb.AtF64(tok, j) + e.posEmb.AtF64(pos, j)
			if e.segEmb != nil {
				v += e.segEmb.AtF64(seg, j)
			}
			row[j] = float32(v)
		}
	}
	return out, nil
}

// Forward runs the encoder over tokens (with optional segment ids) and returns the last hidden state
// [seq, dim] flattened row-major — the transformers last_hidden_state. One recorded command buffer:
// summed-embedding LayerNorm, then per layer a post-LN bidirectional-attention sublayer and a post-LN
// GELU-MLP sublayer.
func (e *GPUBert) Forward(tokens, segments []int) ([]float32, error) {
	seq := len(tokens)
	if seq == 0 || e.posOffset+seq > e.maxPos {
		return nil, fmt.Errorf("llamagpu(%s): seq %d (+offset %d) outside (0,%d]", e.ops.name, seq, e.posOffset, e.maxPos)
	}
	if segments != nil && len(segments) != seq {
		return nil, fmt.Errorf("llamagpu(%s): segments %d != tokens %d", e.ops.name, len(segments), seq)
	}
	emb, err := e.summedEmbedding(tokens, segments)
	if err != nil {
		return nil, err
	}

	r, err := e.ops.newRecorder()
	if err != nil {
		return nil, err
	}
	defer r.Free()

	// Per-Forward scratch (seq-sized; freed after). x holds the running hidden state.
	var scratchErr error
	var scratch []buffer
	sb := func(n int) buffer {
		if scratchErr != nil {
			return nil
		}
		b, e2 := e.ops.newBuffer(make([]float32, n))
		if e2 != nil {
			scratchErr = e2
			return nil
		}
		scratch = append(scratch, b)
		return b
	}
	x := sb(seq * e.dim)
	q := sb(seq * e.dim)
	k := sb(seq * e.dim)
	v := sb(seq * e.dim)
	attn := sb(seq * e.dim)
	ao := sb(seq * e.dim)
	h := sb(seq * e.inter)
	mo := sb(seq * e.dim)
	t := sb(seq * e.dim)
	freeScratch := func() {
		for _, b := range scratch {
			if b != nil {
				b.Release()
			}
		}
	}
	if scratchErr != nil {
		freeScratch()
		return nil, scratchErr
	}
	defer freeScratch()

	if err := x.UploadF32(emb); err != nil {
		return nil, err
	}
	// EmbLN over the summed embedding, then the encoder layers.
	err = r.LayerNorm(x, e.embLNg, e.embLNb, x, seq, e.dim, e.eps)
	for _, b := range e.blocks {
		err = firstErr(err,
			// bidirectional self-attention: q/k/v = x·W + bias, MHA (causal=0), o = attn·Wo + bo.
			b.wq.record(r, x, q, seq), r.AddBias(q, b.bq, q, seq, e.dim),
			b.wk.record(r, x, k, seq), r.AddBias(k, b.bk, k, seq, e.dim),
			b.wv.record(r, x, v, seq), r.AddBias(v, b.bv, v, seq, e.dim),
			r.MHA(q, k, v, attn, seq, seq, e.dim, e.heads, e.heads, e.dk, 0, 0, e.scale),
			b.wo.record(r, attn, ao, seq), r.AddBias(ao, b.bo, ao, seq, e.dim),
			// post-LN attention residual: x = LN(x + ao).
			r.Binary(x, ao, t, binaryAdd),
			r.LayerNorm(t, b.attnLNg, b.attnLNb, x, seq, e.dim, e.eps),
			// GELU MLP: h = gelu(x·W1 + b1); mo = h·W2 + b2.
			b.w1.record(r, x, h, seq), r.AddBias(h, b.b1, h, seq, e.inter),
			r.Unary(h, h, unaryGELU),
			b.w2.record(r, h, mo, seq), r.AddBias(mo, b.b2, mo, seq, e.dim),
			// post-LN FFN residual: x = LN(x + mo).
			r.Binary(x, mo, t, binaryAdd),
			r.LayerNorm(t, b.ffnLNg, b.ffnLNb, x, seq, e.dim, e.eps),
		)
		if err != nil {
			return nil, err
		}
	}
	if err := firstErr(err, r.Commit(), r.Wait()); err != nil {
		return nil, err
	}
	out := make([]float32, seq*e.dim)
	if err := x.DownloadF32(out); err != nil {
		return nil, err
	}
	return out, nil
}

// Release frees every device buffer held by the encoder.
func (e *GPUBert) Release() {
	for _, b := range e.all {
		if b != nil {
			b.Release()
		}
	}
	e.all = nil
	for _, w := range e.qweights {
		if w != nil {
			w.Close()
		}
	}
	e.qweights = nil
}
