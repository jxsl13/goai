package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Bert is a bidirectional transformer encoder (BERT, §R: Devlin et al. 2019),
// the post-LN counterpart of the decoder [GPT]. It maps a token sequence to
// contextualized hidden states [seq, dim] — the representation consumed by
// [MeanPool] / [CosineRerank] for embeddings and by masked-LM heads (see
// [MLMMask]). Unlike GPT it attends bidirectionally (no causal mask), adds a
// learned segment (token-type) embedding, applies a LayerNorm to the summed
// embeddings, and places LayerNorm AFTER each residual add (post-LN):
//
//	x = LN_emb(tokEmb[tokens] + posEmb[:seq] + segEmb[segments])
//	per layer: x = LN_a(x + SelfAttn(x));  x = LN_o(x + FFN(x)) with GELU
//
// Load a Hugging Face checkpoint with [BertFromHF].
type Bert struct {
	Config BertConfig     // encoder geometry: width, heads, layers (see BertConfig)
	TokEmb *tensor.Tensor // [vocab, dim]
	PosEmb *tensor.Tensor // [maxPos, dim]
	SegEmb *tensor.Tensor // [typeVocab, dim]
	EmbLN  *nn.LayerNorm  // LayerNorm over the summed embeddings
	Layers []*BertLayer   // the post-LN encoder layers
}

// BertConfig fixes the encoder geometry.
type BertConfig struct {
	Vocab     int     // vocabulary size
	MaxPos    int     // maximum position embeddings
	TypeVocab int     // number of segment/token-type ids (BERT: 2)
	Dim       int     // hidden width
	Heads     int     // attention heads
	Layers    int     // encoder layers
	Eps       float64 // LayerNorm epsilon
	// PosOffset shifts position ids: 0 for BERT (positions 0,1,2,…), or
	// padding_idx+1 for RoBERTa (positions start at 2). See [RobertaFromHF].
	PosOffset int
}

// BertLayer is one post-LN encoder layer.
type BertLayer struct {
	Attn   *MHA           // bidirectional biased self-attention
	AttnLN *nn.LayerNorm  // LayerNorm after the attention residual
	W1, B1 *tensor.Tensor // intermediate (FFN up): [dim, inter], [inter]
	W2, B2 *tensor.Tensor // output (FFN down): [inter, dim], [dim]
	FFNLN  *nn.LayerNorm  // LayerNorm after the FFN residual
}

// Embed builds the summed, LayerNorm'd input embedding [seq, dim] for the given
// tokens and segment ids. segments may be nil (all-zero, the single-sentence case).
func (b *Bert) Embed(ctx *backend.Context, tokens, segments []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || b.Config.PosOffset+seq > b.Config.MaxPos {
		return nil, fmt.Errorf("nlp: bert sequence length %d (+offset %d) outside (0,%d]", seq, b.Config.PosOffset, b.Config.MaxPos)
	}
	if segments != nil && len(segments) != seq {
		return nil, fmt.Errorf("nlp: bert segments length %d != tokens %d", len(segments), seq)
	}
	tokIdx := tensor.New(b.TokEmb.Dtype(), tensor.Shape{seq})
	posIdx := tensor.New(b.PosEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= b.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, b.Config.Vocab)
		}
		tokIdx.SetF64(float64(t), i)
		posIdx.SetF64(float64(b.Config.PosOffset+i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, b.TokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, b.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	x, err := exec2(ctx, backend.OpAdd, nil, et, ep)
	if err != nil {
		return nil, err
	}
	// Segment (token-type) embedding is optional: BERT/RoBERTa have it, DistilBERT
	// does not (SegEmb == nil).
	if b.SegEmb != nil {
		segIdx := tensor.New(b.SegEmb.Dtype(), tensor.Shape{seq})
		if segments != nil {
			for i := range tokens {
				segIdx.SetF64(float64(segments[i]), i)
			}
		}
		es, err := exec1(ctx, backend.OpEmbed, nil, b.SegEmb, segIdx)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, es); err != nil {
			return nil, err
		}
	}
	return b.EmbLN.Forward(ctx, x)
}

// Forward runs the encoder and returns the final hidden states [seq, dim] — the
// last_hidden_state in transformers terms. segments may be nil.
func (b *Bert) Forward(ctx *backend.Context, tokens, segments []int) (*tensor.Tensor, error) {
	x, err := b.Embed(ctx, tokens, segments)
	if err != nil {
		return nil, err
	}
	for _, l := range b.Layers {
		// post-LN attention sublayer: x = LN(x + Attn(x))
		h, err := l.Attn.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
		if x, err = l.AttnLN.Forward(ctx, x); err != nil {
			return nil, err
		}
		// post-LN FFN sublayer: x = LN(x + W2·gelu(W1·x+b1)+b2)
		if h, err = exec1(ctx, backend.OpMatMul, nil, x, l.W1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, l.B1); err != nil {
			return nil, err
		}
		if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, l.W2); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, l.B2); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
		if x, err = l.FFNLN.Forward(ctx, x); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// Params returns every trainable tensor for optimizers.
func (b *Bert) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{b.TokEmb, b.PosEmb, b.EmbLN.Gamma, b.EmbLN.Beta}
	if b.SegEmb != nil {
		ps = append(ps, b.SegEmb)
	}
	for _, l := range b.Layers {
		ps = append(ps, l.Attn.Wq, l.Attn.Wk, l.Attn.Wv, l.Attn.Wo,
			l.Attn.Bias["q"], l.Attn.Bias["k"], l.Attn.Bias["v"], l.Attn.Bias["o"],
			l.AttnLN.Gamma, l.AttnLN.Beta, l.W1, l.B1, l.W2, l.B2, l.FFNLN.Gamma, l.FFNLN.Beta)
	}
	return ps
}
