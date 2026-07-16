package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// T5Decoder is the decoder half of the T5 seq2seq transformer. Each block has
// three sublayers (pre-LN residual, no bias): causal self-attention with a
// relative-position bias, cross-attention over the encoder output (no bias, no
// mask), and a gated/ReLU FFN. Pair it with a [T5] encoder (both loaded from the
// same checkpoint) for full sequence-to-sequence; [T5Decoder.Decode] returns the
// decoder's final hidden states, which a tied [T5Decoder.Shared] head turns into
// logits.
type T5Decoder struct {
	Config    T5Config
	Shared    *tensor.Tensor     // [vocab, dim] tied token embedding
	RelBias   *nn.T5RelativeBias // causal (unidirectional) relative-position bias
	Blocks    []*T5DecoderBlock
	FinalNorm *nn.RMSNorm
	LMHead    *tensor.Tensor // [dim, vocab] output head (tied models: sharedᵀ·d_model^-0.5)
}

// Logits maps decoder hidden states [dec, dim] (from [T5Decoder.Decode]) to
// vocabulary logits [dec, vocab] via the LM head — the last row is the next-token
// distribution for greedy/beam decoding.
func (d *T5Decoder) Logits(ctx *backend.Context, hidden *tensor.Tensor) (*tensor.Tensor, error) {
	if d.LMHead == nil {
		return nil, fmt.Errorf("nlp: T5Decoder has no LM head")
	}
	return exec1(ctx, backend.OpMatMul, nil, hidden, d.LMHead)
}

// T5DecoderBlock is one decoder block: self-attn, cross-attn, FFN.
type T5DecoderBlock struct {
	SelfNorm           *nn.RMSNorm
	SWq, SWk, SWv, SWo *tensor.Tensor // causal self-attention
	CrossNorm          *nn.RMSNorm
	CWq, CWk, CWv, CWo *tensor.Tensor // cross-attention over encoder output
	FFNNorm            *nn.RMSNorm
	Wi0, WOut          *tensor.Tensor
	Wi1                *tensor.Tensor // gated FFN up (nil → ReLU)
}

// Decode runs the decoder over decoder tokens attending to encoderOut [enc, dim]
// (from [T5.Forward]), returning the decoder hidden states [dec, dim].
func (d *T5Decoder) Decode(ctx *backend.Context, encoderOut *tensor.Tensor, tokens []int) (*tensor.Tensor, error) {
	dseq := len(tokens)
	if dseq == 0 {
		return nil, fmt.Errorf("nlp: T5 decoder empty tokens")
	}
	if encoderOut == nil || encoderOut.Ndim() != 2 {
		return nil, fmt.Errorf("nlp: T5 decoder needs encoderOut [enc, dim]")
	}
	eseq := encoderOut.Shape()[0]
	idx := tensor.New(d.Shared.Dtype(), tensor.Shape{dseq})
	for i, t := range tokens {
		if t < 0 || t >= d.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, d.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, d.Shared, idx)
	if err != nil {
		return nil, err
	}
	// Self-attention mask [heads, dseq, dseq] = causal relative bias, with −Inf on
	// future keys (j > i) so the decoder is autoregressive.
	rb, err := d.RelBias.Bias(ctx, dseq, dseq)
	if err != nil {
		return nil, err
	}
	if rb, err = rb.Permute(2, 0, 1); err != nil {
		return nil, err
	}
	rb = rb.Contiguous()
	sm := rb.Storage().F64()
	for h := 0; h < d.Config.Heads; h++ {
		for i := 0; i < dseq; i++ {
			for j := i + 1; j < dseq; j++ {
				sm[(h*dseq+i)*dseq+j] = math.Inf(-1)
			}
		}
	}
	// Cross-attention mask [dseq, eseq] = all zero (no bias, attend to everything).
	crossMask := tensor.New(tensor.F64, tensor.Shape{dseq, eseq})
	attrs := backend.AttnAttrs{Heads: d.Config.Heads, Scale: math.Sqrt(float64(d.Config.HeadDim))}
	for _, b := range d.Blocks {
		// causal self-attention
		h, err := b.SelfNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWv)
		if err != nil {
			return nil, err
		}
		attn, err := exec1(ctx, backend.OpMHAMasked, attrs, q, k, v, rb)
		if err != nil {
			return nil, err
		}
		if attn, err = exec1(ctx, backend.OpMatMul, nil, attn, b.SWo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, attn); err != nil {
			return nil, err
		}
		// cross-attention over the encoder output
		if h, err = b.CrossNorm.Forward(ctx, x); err != nil {
			return nil, err
		}
		cq, err := exec1(ctx, backend.OpMatMul, nil, h, b.CWq)
		if err != nil {
			return nil, err
		}
		ck, err := exec1(ctx, backend.OpMatMul, nil, encoderOut, b.CWk)
		if err != nil {
			return nil, err
		}
		cv, err := exec1(ctx, backend.OpMatMul, nil, encoderOut, b.CWv)
		if err != nil {
			return nil, err
		}
		cattn, err := exec1(ctx, backend.OpMHAMasked, attrs, cq, ck, cv, crossMask)
		if err != nil {
			return nil, err
		}
		if cattn, err = exec1(ctx, backend.OpMatMul, nil, cattn, b.CWo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, cattn); err != nil {
			return nil, err
		}
		// FFN
		if h, err = b.FFNNorm.Forward(ctx, x); err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.Wi1 != nil {
			g, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wi0)
			if err != nil {
				return nil, err
			}
			if g, err = exec1(ctx, backend.OpGELU, nil, g); err != nil {
				return nil, err
			}
			u, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wi1)
			if err != nil {
				return nil, err
			}
			if ff, err = exec1(ctx, backend.OpMul, nil, g, u); err != nil {
				return nil, err
			}
		} else {
			if ff, err = exec1(ctx, backend.OpMatMul, nil, h, b.Wi0); err != nil {
				return nil, err
			}
			if ff, err = exec1(ctx, backend.OpReLU, nil, ff); err != nil {
				return nil, err
			}
		}
		if ff, err = exec1(ctx, backend.OpMatMul, nil, ff, b.WOut); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	return d.FinalNorm.Forward(ctx, x)
}

// T5DecoderFromHF builds a [T5Decoder] from a Hugging Face T5Model / T5ForConditionalGeneration
// checkpoint (the decoder.* tensors). cfg supplies Heads/HeadDim/Eps as for [T5FromHF]. Pair
// it with T5FromHF(ts, cfg) on the same map for full seq2seq.
func T5DecoderFromHF(ts map[string]*tensor.Tensor, cfg T5Config) (*T5Decoder, error) {
	get := func(name string) (*tensor.Tensor, error) {
		if t, ok := ts[name]; ok {
			return t, nil
		}
		return nil, fmt.Errorf("nlp: HF T5 decoder missing %s", name)
	}
	shared, err := get("shared.weight")
	if err != nil {
		return nil, err
	}
	if cfg.Heads <= 0 || cfg.HeadDim <= 0 {
		return nil, fmt.Errorf("nlp: T5DecoderFromHF needs cfg.Heads and cfg.HeadDim")
	}
	if cfg.NumBuckets == 0 {
		cfg.NumBuckets = 32
	}
	if cfg.MaxDistance == 0 {
		cfg.MaxDistance = 128
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-6
	}
	cfg.Vocab, cfg.Dim = shared.Shape()[0], shared.Shape()[1]
	rms := func(name string) (*nn.RMSNorm, error) {
		g, err := get(name)
		if err != nil {
			return nil, err
		}
		return &nn.RMSNorm{Gamma: cloneF64(g), Eps: cfg.Eps}, nil
	}
	w := func(name string) (*tensor.Tensor, error) {
		t, err := get(name)
		if err != nil {
			return nil, err
		}
		return transpose2D(t), nil
	}
	// Decoder relative bias is causal (unidirectional bucketing).
	rbw, err := get("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
	if err != nil {
		return nil, err
	}
	relBias, err := nn.NewT5RelativeBias(cfg.NumBuckets, cfg.Heads, cfg.MaxDistance, false, tensor.F64)
	if err != nil {
		return nil, err
	}
	relBias.Table = cloneF64(rbw)
	m := &T5Decoder{Config: cfg, Shared: cloneF64(shared), RelBias: relBias}

	layers := 0
	for {
		if _, ok := ts[fmt.Sprintf("decoder.block.%d.layer.0.SelfAttention.q.weight", layers)]; !ok {
			break
		}
		layers++
	}
	if layers == 0 {
		return nil, fmt.Errorf("nlp: HF T5 has no decoder.block.*")
	}
	cfg.Layers = layers
	_, gated := ts["decoder.block.0.layer.2.DenseReluDense.wi_0.weight"]
	cfg.Gated = gated
	m.Config = cfg
	for l := range layers {
		p := fmt.Sprintf("decoder.block.%d.", l)
		blk := &T5DecoderBlock{}
		if blk.SWq, err = w(p + "layer.0.SelfAttention.q.weight"); err != nil {
			return nil, err
		}
		if blk.SWk, err = w(p + "layer.0.SelfAttention.k.weight"); err != nil {
			return nil, err
		}
		if blk.SWv, err = w(p + "layer.0.SelfAttention.v.weight"); err != nil {
			return nil, err
		}
		if blk.SWo, err = w(p + "layer.0.SelfAttention.o.weight"); err != nil {
			return nil, err
		}
		if blk.SelfNorm, err = rms(p + "layer.0.layer_norm.weight"); err != nil {
			return nil, err
		}
		if blk.CWq, err = w(p + "layer.1.EncDecAttention.q.weight"); err != nil {
			return nil, err
		}
		if blk.CWk, err = w(p + "layer.1.EncDecAttention.k.weight"); err != nil {
			return nil, err
		}
		if blk.CWv, err = w(p + "layer.1.EncDecAttention.v.weight"); err != nil {
			return nil, err
		}
		if blk.CWo, err = w(p + "layer.1.EncDecAttention.o.weight"); err != nil {
			return nil, err
		}
		if blk.CrossNorm, err = rms(p + "layer.1.layer_norm.weight"); err != nil {
			return nil, err
		}
		if blk.FFNNorm, err = rms(p + "layer.2.layer_norm.weight"); err != nil {
			return nil, err
		}
		if gated {
			if blk.Wi0, err = w(p + "layer.2.DenseReluDense.wi_0.weight"); err != nil {
				return nil, err
			}
			if blk.Wi1, err = w(p + "layer.2.DenseReluDense.wi_1.weight"); err != nil {
				return nil, err
			}
		} else {
			if blk.Wi0, err = w(p + "layer.2.DenseReluDense.wi.weight"); err != nil {
				return nil, err
			}
		}
		if blk.WOut, err = w(p + "layer.2.DenseReluDense.wo.weight"); err != nil {
			return nil, err
		}
		m.Blocks = append(m.Blocks, blk)
	}
	if m.FinalNorm, err = rms("decoder.final_layer_norm.weight"); err != nil {
		return nil, err
	}
	// LM head. Tied models (t5-small/base/large, flan-T5) save lm_head.weight equal
	// to shared and apply T5's d_model^-0.5 rescale of the decoder output; folding
	// that scale into the transposed head makes Logits a plain matmul. Untied heads
	// (rare) carry their own scale, so cfg.UntiedLMHead skips the rescale.
	head := shared
	if lh, ok := ts["lm_head.weight"]; ok {
		head = lh
	}
	m.LMHead = transpose2D(head) // [dim, vocab]
	if !cfg.UntiedLMHead {
		scale := 1.0 / math.Sqrt(float64(cfg.Dim))
		hs := m.LMHead.Storage().F64()
		for i := range hs {
			hs[i] *= scale
		}
	}
	return m, nil
}

// Generate performs autoregressive seq2seq decoding: starting from startToken
// (T5's decoder start id is the pad token 0), it repeatedly decodes the sequence
// so far against encoderOut, samples the next token from the final-position
// logits with s, and stops at eosToken (T5: 1) or after maxNew tokens. It returns the generated
// tokens (excluding startToken) and uses the KV-cache ([DecodeStep]) so each step
// is O(sequence) rather than re-decoding the whole prefix; use [Greedy] for argmax.
func (d *T5Decoder) Generate(ctx *backend.Context, encoderOut *tensor.Tensor, maxNew, startToken, eosToken int, s TokenSampler) ([]int, error) {
	gen := []int{startToken}
	vocab := d.Config.Vocab
	cache := d.NewCache()
	for pos := 0; pos < maxNew; pos++ {
		hidden, err := d.DecodeStep(ctx, cache, encoderOut, gen[pos], pos)
		if err != nil {
			return nil, err
		}
		logits, err := d.Logits(ctx, hidden)
		if err != nil {
			return nil, err
		}
		row := make([]float64, vocab)
		for vtok := 0; vtok < vocab; vtok++ {
			row[vtok] = logits.AtF64(0, vtok)
		}
		next := s.SampleWithHistory(row, gen)
		if next == eosToken {
			break
		}
		gen = append(gen, next)
	}
	return gen[1:], nil
}

// T5DecoderCache holds the incremental-decoding state for [T5Decoder.DecodeStep]:
// the growing self-attention key/value per block, and the fixed cross-attention
// key/value (the encoder is projected once). Create one per generated sequence.
type T5DecoderCache struct {
	selfK, selfV   []*tensor.Tensor // per block, grown one row per step
	crossK, crossV []*tensor.Tensor // per block, computed once from the encoder
}

// NewT5DecoderCache allocates a cache sized for the decoder's blocks.
func (d *T5Decoder) NewCache() *T5DecoderCache {
	n := len(d.Blocks)
	return &T5DecoderCache{
		selfK: make([]*tensor.Tensor, n), selfV: make([]*tensor.Tensor, n),
		crossK: make([]*tensor.Tensor, n), crossV: make([]*tensor.Tensor, n),
	}
}

// relBiasRow builds the [heads, 1, pos+1] additive bias for a single query at
// absolute position pos attending to keys 0..pos (all past, so no causal mask).
func (d *T5Decoder) relBiasRow(ctx *backend.Context, pos int) (*tensor.Tensor, error) {
	full, err := d.RelBias.Bias(ctx, pos+1, pos+1) // [pos+1, pos+1, heads]
	if err != nil {
		return nil, err
	}
	fs := full.Contiguous().Storage().F64()
	heads, kk := d.Config.Heads, pos+1
	out := tensor.New(tensor.F64, tensor.Shape{heads, 1, kk})
	os := out.Storage().F64()
	for h := 0; h < heads; h++ {
		for k := 0; k < kk; k++ {
			os[h*kk+k] = fs[(pos*kk+k)*heads+h] // full[pos][k][h]
		}
	}
	return out, nil
}

// DecodeStep decodes a single token at absolute position pos against encoderOut,
// using and updating cache, returning that step's hidden state [1, dim]. It is
// the O(n)-per-step incremental form of [T5Decoder.Decode].
func (d *T5Decoder) DecodeStep(ctx *backend.Context, cache *T5DecoderCache, encoderOut *tensor.Tensor, token, pos int) (*tensor.Tensor, error) {
	if token < 0 || token >= d.Config.Vocab {
		return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, d.Config.Vocab)
	}
	idx := tensor.New(d.Shared.Dtype(), tensor.Shape{1})
	idx.SetF64(float64(token), 0)
	x, err := exec1(ctx, backend.OpEmbed, nil, d.Shared, idx)
	if err != nil {
		return nil, err
	}
	selfMask, err := d.relBiasRow(ctx, pos)
	if err != nil {
		return nil, err
	}
	eseq := encoderOut.Shape()[0]
	crossMask := tensor.New(tensor.F64, tensor.Shape{1, eseq})
	attrs := backend.AttnAttrs{Heads: d.Config.Heads, Scale: math.Sqrt(float64(d.Config.HeadDim))}
	for l, b := range d.Blocks {
		// causal self-attention with the growing K/V cache
		h, err := b.SelfNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWq)
		if err != nil {
			return nil, err
		}
		kt, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWk)
		if err != nil {
			return nil, err
		}
		vt, err := exec1(ctx, backend.OpMatMul, nil, h, b.SWv)
		if err != nil {
			return nil, err
		}
		if cache.selfK[l] == nil {
			cache.selfK[l], cache.selfV[l] = kt, vt
		} else {
			cache.selfK[l] = concatRows(cache.selfK[l], kt)
			cache.selfV[l] = concatRows(cache.selfV[l], vt)
		}
		attn, err := exec1(ctx, backend.OpMHAMasked, attrs, q, cache.selfK[l], cache.selfV[l], selfMask)
		if err != nil {
			return nil, err
		}
		if attn, err = exec1(ctx, backend.OpMatMul, nil, attn, b.SWo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, attn); err != nil {
			return nil, err
		}
		// cross-attention; project the (fixed) encoder K/V once
		if h, err = b.CrossNorm.Forward(ctx, x); err != nil {
			return nil, err
		}
		if cache.crossK[l] == nil {
			if cache.crossK[l], err = exec1(ctx, backend.OpMatMul, nil, encoderOut, b.CWk); err != nil {
				return nil, err
			}
			if cache.crossV[l], err = exec1(ctx, backend.OpMatMul, nil, encoderOut, b.CWv); err != nil {
				return nil, err
			}
		}
		cq, err := exec1(ctx, backend.OpMatMul, nil, h, b.CWq)
		if err != nil {
			return nil, err
		}
		cattn, err := exec1(ctx, backend.OpMHAMasked, attrs, cq, cache.crossK[l], cache.crossV[l], crossMask)
		if err != nil {
			return nil, err
		}
		if cattn, err = exec1(ctx, backend.OpMatMul, nil, cattn, b.CWo); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, cattn); err != nil {
			return nil, err
		}
		// FFN
		if h, err = b.FFNNorm.Forward(ctx, x); err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.Wi1 != nil {
			g, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wi0)
			if err != nil {
				return nil, err
			}
			if g, err = exec1(ctx, backend.OpGELU, nil, g); err != nil {
				return nil, err
			}
			u, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wi1)
			if err != nil {
				return nil, err
			}
			if ff, err = exec1(ctx, backend.OpMul, nil, g, u); err != nil {
				return nil, err
			}
		} else {
			if ff, err = exec1(ctx, backend.OpMatMul, nil, h, b.Wi0); err != nil {
				return nil, err
			}
			if ff, err = exec1(ctx, backend.OpReLU, nil, ff); err != nil {
				return nil, err
			}
		}
		if ff, err = exec1(ctx, backend.OpMatMul, nil, ff, b.WOut); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	return d.FinalNorm.Forward(ctx, x)
}
