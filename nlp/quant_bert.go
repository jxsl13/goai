package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/format/gguf"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// QuantBert is a [Bert] encoder whose six projection matrices per layer stay in
// ggml block-quantized byte storage instead of full-precision tensors. It is the
// memory-efficient inference form for a BERT, RoBERTa, or DistilBERT loaded from
// Hugging Face: load with [BertFromHF], [RobertaFromHF], or [DistilBertFromHF],
// then call [QuantizeBert].
//
// The forward preserves BERT's exact structure: f32 token/position/segment
// embeddings and LayerNorms; separately quantized q/k/v/o projections around
// bidirectional attention; and a quantized two-layer GELU FFN. Keeping the small
// embeddings, biases, and norm parameters in f32 avoids spending accuracy where
// little memory is saved. The large 2-D projection weights are [nn.QuantLinear]
// byte blocks and are never retained as float matrices, so callers may release
// the source [Bert] after conversion. Inference-only: quantized weights bypass
// autograd.
type QuantBert struct {
	Config BertConfig // encoder geometry copied from the source BERT

	tokEmb *tensor.Tensor // [vocab,dim], f32 lookup table
	posEmb *tensor.Tensor // [maxPos,dim], f32 lookup table
	segEmb *tensor.Tensor // [typeVocab,dim], f32; nil for DistilBERT
	embLN  *nn.LayerNorm  // embedding LayerNorm, f32 gamma and beta
	layers []*quantBertLayer
}

// quantBertLayer mirrors BertLayer without retaining any full-precision 2-D
// projection. Biases and LayerNorm parameters remain f32.
type quantBertLayer struct {
	wq, wk, wv, wo *nn.QuantLinear
	bq, bk, bv, bo *tensor.Tensor
	attnLN         *nn.LayerNorm
	w1, w2         *nn.QuantLinear
	b1, b2         *tensor.Tensor
	ffnLN          *nn.LayerNorm
}

// QuantizeBert converts a float [Bert] into persistent block-quantized inference
// storage. qt controls the accuracy/memory trade-off; Q8_0 is the near-lossless
// INT8 choice used for loaded Hugging Face BERT inference, while the other
// [gguf.QuantType] values provide lower-bit variants. Every projection's input
// dimension must be a multiple of qt's block size (32 for Q8_0/Q4_0 and 256 for
// the k-quants); standard BERT widths such as 768 satisfy Q8_0 directly.
//
// Conversion is offline: it temporarily reads the source float weights to emit
// the byte blocks, but QuantBert itself retains only those bytes for the six large
// projections per layer. Small lookup tables, one-dimensional biases, and norm
// parameters are copied to f32.
func QuantizeBert(m *Bert, qt gguf.QuantType) (*QuantBert, error) {
	if m == nil {
		return nil, fmt.Errorf("nlp: QuantizeBert nil model")
	}
	mkQ := func(w *tensor.Tensor) (*nn.QuantLinear, error) {
		if w == nil || w.Ndim() != 2 {
			return nil, fmt.Errorf("expected rank-2 projection, got %v", shapeOf(w))
		}
		in, out := w.Shape()[0], w.Shape()[1] // GoAI [in,out]
		weight, err := gguf.Quantize(transpose2D(w), qt)
		if err != nil {
			return nil, err
		}
		return &nn.QuantLinear{Weight: weight, QT: qt, In: in, Out: out}, nil
	}

	q := &QuantBert{
		Config: m.Config,
		tokEmb: f32Clone(m.TokEmb),
		posEmb: f32Clone(m.PosEmb),
		segEmb: f32CloneIf(m.SegEmb),
		embLN:  f32LayerNorm(m.EmbLN),
	}
	for i, l := range m.Layers {
		if l == nil || l.Attn == nil {
			return nil, fmt.Errorf("nlp: QuantizeBert layer %d missing attention", i)
		}
		qb := &quantBertLayer{
			bq:     f32CloneIf(l.Attn.Bias["q"]),
			bk:     f32CloneIf(l.Attn.Bias["k"]),
			bv:     f32CloneIf(l.Attn.Bias["v"]),
			bo:     f32CloneIf(l.Attn.Bias["o"]),
			attnLN: f32LayerNorm(l.AttnLN),
			b1:     f32CloneIf(l.B1),
			b2:     f32CloneIf(l.B2),
			ffnLN:  f32LayerNorm(l.FFNLN),
		}
		var err error
		for _, item := range []struct {
			name string
			src  *tensor.Tensor
			dst  **nn.QuantLinear
		}{
			{"Wq", l.Attn.Wq, &qb.wq}, {"Wk", l.Attn.Wk, &qb.wk},
			{"Wv", l.Attn.Wv, &qb.wv}, {"Wo", l.Attn.Wo, &qb.wo},
			{"W1", l.W1, &qb.w1}, {"W2", l.W2, &qb.w2},
		} {
			if *item.dst, err = mkQ(item.src); err != nil {
				return nil, fmt.Errorf("nlp: QuantizeBert layer %d %s: %w", i, item.name, err)
			}
		}
		q.layers = append(q.layers, qb)
	}
	return q, nil
}

// Forward runs the quantized bidirectional encoder and returns contextual hidden
// states [seq,dim]. segments follows [Bert.Forward]: nil means all-zero segment ids,
// and DistilBERT ignores it because its segment table is absent.
func (b *QuantBert) Forward(ctx *backend.Context, tokens, segments []int) (*tensor.Tensor, error) {
	x, err := b.embed(ctx, tokens, segments)
	if err != nil {
		return nil, err
	}
	attn := backend.AttnAttrs{Heads: b.Config.Heads, Causal: false}
	for _, l := range b.layers {
		q, err := quantProjBias(ctx, x, l.wq, l.bq)
		if err != nil {
			return nil, err
		}
		k, err := quantProjBias(ctx, x, l.wk, l.bk)
		if err != nil {
			return nil, err
		}
		v, err := quantProjBias(ctx, x, l.wv, l.bv)
		if err != nil {
			return nil, err
		}
		h, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		if h, err = quantProjBias(ctx, h, l.wo, l.bo); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
		if x, err = l.attnLN.Forward(ctx, x); err != nil {
			return nil, err
		}

		if h, err = quantProjBias(ctx, x, l.w1, l.b1); err != nil {
			return nil, err
		}
		if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = quantProjBias(ctx, h, l.w2, l.b2); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
		if x, err = l.ffnLN.Forward(ctx, x); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Close releases any device-resident copies of the quantized projection weights.
// It is idempotent; the host-side byte blocks remain available for CPU fallback.
func (b *QuantBert) Close() error {
	var first error
	for _, l := range b.layers {
		for _, w := range []*nn.QuantLinear{l.wq, l.wk, l.wv, l.wo, l.w1, l.w2} {
			if w != nil {
				if err := w.Close(); err != nil && first == nil {
					first = err
				}
			}
		}
	}
	return first
}

func (b *QuantBert) embed(ctx *backend.Context, tokens, segments []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || b.Config.PosOffset+seq > b.Config.MaxPos {
		return nil, fmt.Errorf("nlp: bert sequence length %d (+offset %d) outside (0,%d]", seq, b.Config.PosOffset, b.Config.MaxPos)
	}
	if segments != nil && len(segments) != seq {
		return nil, fmt.Errorf("nlp: bert segments length %d != tokens %d", len(segments), seq)
	}
	tokIdx := tensor.New(b.tokEmb.Dtype(), tensor.Shape{seq})
	posIdx := tensor.New(b.posEmb.Dtype(), tensor.Shape{seq})
	for i, token := range tokens {
		if token < 0 || token >= b.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", token, b.Config.Vocab)
		}
		tokIdx.SetF64(float64(token), i)
		posIdx.SetF64(float64(b.Config.PosOffset+i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, b.tokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, b.posEmb, posIdx)
	if err != nil {
		return nil, err
	}
	x, err := exec2(ctx, backend.OpAdd, nil, et, ep)
	if err != nil {
		return nil, err
	}
	if b.segEmb != nil {
		segIdx := tensor.New(b.segEmb.Dtype(), tensor.Shape{seq})
		if segments != nil {
			for i, segment := range segments {
				segIdx.SetF64(float64(segment), i)
			}
		}
		es, err := exec1(ctx, backend.OpEmbed, nil, b.segEmb, segIdx)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, es); err != nil {
			return nil, err
		}
	}
	return b.embLN.Forward(ctx, x)
}

func (b *QuantBert) projectionBytes() int {
	var n int
	for _, l := range b.layers {
		for _, w := range []*nn.QuantLinear{l.wq, l.wk, l.wv, l.wo, l.w1, l.w2} {
			n += len(w.Weight)
		}
	}
	return n
}

func shapeOf(t *tensor.Tensor) tensor.Shape {
	if t == nil {
		return nil
	}
	return t.Shape()
}
