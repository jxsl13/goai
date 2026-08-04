package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Mamba is a selective-state-space language model (Gu & Dao 2023,
// arXiv:2312.00752, §R77) — the FIRST non-transformer architecture in GoAI. It
// replaces the transformer's attention + FFN stack with a stack of Mamba mixers
// ([nn.MambaBlock]): each layer is a linear-time selective scan, so the whole
// model runs in O(L) instead of the O(L²) of self-attention, with no KV cache.
//
// The graph mirrors Hugging Face MambaForCausalLM exactly: token embedding →
// L × (pre-RMSNorm → MambaBlock → residual add) → final RMSNorm → tied LM head
// (logits = hidden · embeddingᵀ). There is NO attention and NO separate MLP —
// the mixer is the only sequence-mixing primitive. Build one from a checkpoint
// with [MambaFromHF].
type Mamba struct {
	Config MambaConfig    // checkpoint dimensions (see MambaConfig)
	Embed  *tensor.Tensor // token embedding [vocab, d_model]; tied LM head is its transpose
	Layers []MambaLayer   // the selective-scan residual blocks
	Norm   *nn.RMSNorm    // final RMSNorm (backbone.norm_f)
	Head   *tensor.Tensor // [d_model, vocab] = Embedᵀ (tied); logits = hidden · Head
}

// MambaLayer is one residual block: pre-norm then the selective-scan mixer.
type MambaLayer struct {
	Norm  *nn.RMSNorm    // pre-mixer RMSNorm (backbone.layers.N.norm)
	Mixer *nn.MambaBlock // the selective-scan sequence mixer (backbone.layers.N.mixer)
}

// MambaConfig carries the dimensions of a Mamba checkpoint. All fields except Eps
// are inferred from the tensors by [MambaFromHF]; Eps defaults to 1e-5 (HF's
// layer_norm_epsilon) when left zero.
type MambaConfig struct {
	DModel int     // hidden_size (model/embedding width)
	N      int     // state_size (SSM state dimension, d_state)
	DConv  int     // conv_kernel (depthwise causal-conv width)
	Expand int     // expand factor; d_inner = Expand·DModel
	DtRank int     // time_step_rank (Δ low-rank projection rank)
	Layers int     // num_hidden_layers
	Vocab  int     // vocab_size
	Eps    float64 // RMSNorm epsilon (layer_norm_epsilon)
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → per layer
// h = h + Mixer(RMSNorm(h)) → final RMSNorm → tied head (h · Embedᵀ).
func (m *Mamba) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: Mamba prompt is empty")
	}
	idx := tensor.New(m.Embed.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	h, err := exec1(ctx, backend.OpEmbed, nil, m.Embed, idx)
	if err != nil {
		return nil, err
	}
	for i := range m.Layers {
		n, err := m.Layers[i].Norm.Forward(ctx, h)
		if err != nil {
			return nil, err
		}
		mix, err := m.Layers[i].Mixer.Forward(ctx, n)
		if err != nil {
			return nil, err
		}
		//perfscan:ignore PS6017 alloc-only; OpAdd dispatch dominates variadic slice
		if h, err = exec1(ctx, backend.OpAdd, nil, h, mix); err != nil {
			return nil, err
		}
	}
	h, err = m.Norm.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Head)
}

// Params returns every trainable tensor: embedding, per-layer norm + mixer, and
// the final norm. The head is tied to Embed, so it is not returned separately.
func (m *Mamba) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Embed}
	for i := range m.Layers {
		ps = append(ps, m.Layers[i].Norm.Params()...)
		ps = append(ps, m.Layers[i].Mixer.Params()...)
	}
	ps = append(ps, m.Norm.Params()...)
	return ps
}
