package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// T5 is the encoder of the T5 text-to-text transformer (Raffel et al. 2020): a
// bidirectional stack with relative-position attention bias, RMSNorm (T5's
// "layer norm" is RMS with no bias/mean), no absolute position embedding, no
// bias in any projection, and — unlike standard attention — NO 1/sqrt(d) score
// scaling (folded into the weight init). This is the encoder half; it maps
// tokens to contextual states [seq, dim] (T5EncoderModel's last_hidden_state),
// which the sentence-T5 / retrieval and text-classification use cases consume.
// Load a Hugging Face checkpoint with [T5FromHF].
//
// Further reading: Raffel et al., "Exploring the Limits of Transfer Learning
// with a Unified Text-to-Text Transformer" (https://arxiv.org/abs/1910.10683),
// and the Hugging Face T5 documentation
// (https://huggingface.co/docs/transformers/model_doc/t5).
type T5 struct {
	Config    T5Config           // Config describes the loaded encoder geometry.
	Shared    *tensor.Tensor     // [vocab, dim] tied token embedding
	RelBias   *nn.T5RelativeBias // shared relative-position bias (block 0 in HF)
	Blocks    []*T5Block         // Blocks are the encoder layers in execution order.
	FinalNorm *nn.RMSNorm        // FinalNorm normalizes the encoder output.
}

// T5Config fixes the encoder geometry.
type T5Config struct {
	Vocab        int     // Vocab is the number of token embeddings; T5FromHF infers it.
	Dim          int     // d_model
	Heads        int     // num_heads
	HeadDim      int     // d_kv (heads*HeadDim need NOT equal Dim)
	Layers       int     // num_layers
	FFN          int     // d_ff
	NumBuckets   int     // relative_attention_num_buckets (default 32)
	MaxDistance  int     // relative_attention_max_distance (default 128)
	Eps          float64 // layer_norm_epsilon (default 1e-6)
	Gated        bool    // T5 v1.1 gated-GELU FFN vs v1.0 ReLU
	UntiedLMHead bool    // decoder LM head is not tied to the embedding (skips the d_model^-0.5 rescale)
}

// T5Block is one encoder block (pre-LN residual, no bias anywhere).
type T5Block struct {
	AttnNorm       *nn.RMSNorm    // AttnNorm normalizes inputs to self-attention.
	Wq, Wk, Wv, Wo *tensor.Tensor // [dim, heads*HeadDim] … [heads*HeadDim, dim]
	FFNNorm        *nn.RMSNorm    // FFNNorm normalizes inputs to the feed-forward network.
	Wi0, WOut      *tensor.Tensor // FFN in (gate for gated) and out
	Wi1            *tensor.Tensor // gated FFN up projection; nil for v1.0 ReLU
}

// Forward runs the encoder and returns the final hidden states [seq, dim].
func (m *T5) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: T5 empty token sequence")
	}
	idx := tensor.New(m.Shared.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.Shared, idx)
	if err != nil {
		return nil, err
	}
	// Relative bias comes back [seq, seq, heads]; the masked-attention kernel wants
	// a per-head [heads, seq, seq] mask, shared across all blocks (as in HF).
	bias, err := m.RelBias.Bias(ctx, seq, seq)
	if err != nil {
		return nil, err
	}
	if bias, err = bias.Permute(2, 0, 1); err != nil {
		return nil, err
	}
	bias = bias.Contiguous()
	// Scale = sqrt(HeadDim) cancels OpMHAMasked's built-in 1/sqrt(HeadDim), giving
	// T5's unscaled QKᵀ (T5 folds the scale into initialization).
	attrs := backend.AttnAttrs{Heads: m.Config.Heads, Scale: math.Sqrt(float64(m.Config.HeadDim))}
	for _, b := range m.Blocks {
		// self-attention sublayer
		h, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, h, b.Wv)
		if err != nil {
			return nil, err
		}
		attn, err := exec1(ctx, backend.OpMHAMasked, attrs, q, k, v, bias)
		if err != nil {
			return nil, err
		}
		if attn, err = exec1(ctx, backend.OpMatMul, nil, attn, b.Wo); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, attn); err != nil {
			return nil, err
		}
		// FFN sublayer
		if h, err = b.FFNNorm.Forward(ctx, x); err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if b.Wi1 != nil { // gated-GELU (v1.1): gelu(h·Wi0) ⊙ (h·Wi1)
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
			if ff, err = exec2(ctx, backend.OpMul, nil, g, u); err != nil {
				return nil, err
			}
		} else { // ReLU (v1.0)
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
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(ctx, x)
}

// Params returns the encoder's trainable tensors — for fine-tuning the loaded
// model (now that OpMHAMasked has a VJP, its relative-position attention trains).
func (m *T5) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Shared, m.RelBias.Table, m.FinalNorm.Gamma}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma, b.Wi0, b.WOut)
		if b.Wi1 != nil {
			ps = append(ps, b.Wi1)
		}
	}
	return ps
}
