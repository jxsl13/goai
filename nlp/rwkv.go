package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// RWKV is an RWKV-4 language model (Peng et al. 2023, "RWKV: Reinventing RNNs
// for the Transformer Era", EMNLP Findings, arXiv:2305.13048) — GoAI's third
// recurrent / linear-attention family after the Mamba SSMs and the Retention/
// GLA line. Every layer is an [nn.RWKVBlock]: WKV time-mixing (a per-channel,
// time-decayed weighted average — linear attention with no softmax and no KV
// cache) followed by a gated squared-ReLU channel-mixing MLP. Like Mamba it
// runs in O(L) with O(1)-per-token recurrent inference; unlike a transformer
// there is no self-attention and no positional encoding (the one-step token
// shift inside each block is RWKV's positional signal).
//
// The graph mirrors Hugging Face RwkvForCausalLM exactly: token embedding →
// an EXTRA LayerNorm applied only before block 0 (rwkv.blocks.0.pre_ln, "ln0")
// → L × [nn.RWKVBlock] (each with its own ln1/ln2 and two residuals) → a final
// LayerNorm (rwkv.ln_out) → an UNTIED linear head (head.weight). Build one from
// a checkpoint with [RWKVFromHF].
type RWKV struct {
	Config RWKVConfig
	Embed  *tensor.Tensor  // token embedding [vocab, dim]
	PreLN  *nn.LayerNorm   // block-0 pre-norm (rwkv.blocks.0.pre_ln / "ln0")
	Blocks []*nn.RWKVBlock // the L time-mix + channel-mix layers
	LNOut  *nn.LayerNorm   // final LayerNorm (rwkv.ln_out)
	Head   *tensor.Tensor  // [dim, vocab] LM head (UNTIED); logits = hidden · Head
}

// RWKVConfig carries the dimensions of an RWKV-4 checkpoint. Every field except
// Eps is inferred from the tensors by [RWKVFromHF]; Eps defaults to 1e-5 (HF's
// layer_norm_epsilon) when left zero.
type RWKVConfig struct {
	Vocab  int     // vocab_size
	Dim    int     // hidden_size (model width; also the WKV attention width)
	Hidden int     // intermediate_size (channel-mixing hidden width, ~4·Dim)
	Layers int     // num_hidden_layers
	Eps    float64 // LayerNorm epsilon (layer_norm_epsilon)
}

// Forward computes logits [seq, vocab] for the prompt tokens: embed → block-0
// pre-LayerNorm → per block time-mix + channel-mix (each with its two internal
// LayerNorms and residuals) → final LayerNorm → untied head.
func (m *RWKV) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 {
		return nil, fmt.Errorf("nlp: RWKV prompt is empty")
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
	// The pre_ln normalises the embedding before block 0 only (RWKV-4 quirk).
	if h, err = m.PreLN.Forward(ctx, h); err != nil {
		return nil, err
	}
	for _, b := range m.Blocks {
		if h, err = b.Forward(ctx, h); err != nil {
			return nil, err
		}
	}
	if h, err = m.LNOut.Forward(ctx, h); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, m.Head)
}

// Params returns every trainable tensor: embedding, block-0 pre-norm, each
// block's parameters, the final norm, and the untied head.
func (m *RWKV) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Embed}
	ps = append(ps, m.PreLN.Params()...)
	for _, b := range m.Blocks {
		ps = append(ps, b.Params()...)
	}
	ps = append(ps, m.LNOut.Params()...)
	ps = append(ps, m.Head)
	return ps
}
