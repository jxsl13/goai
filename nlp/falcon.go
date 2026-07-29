package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Falcon is TII's Falcon decoder-only transformer (transformers'
// FalconForCausalLM), targeting the ORIGINAL Falcon-7B architecture:
// new_decoder_architecture=False, parallel_attn=True, multi_query=True,
// alibi=False. It is a separate type — rather than a [Llama] or [Cohere]
// variant — because of its fused multi-query QKV layout. Every departure below is
// confirmed against transformers.models.falcon.modeling_falcon:
//
//   - SINGLE-NORM PARALLEL residual (like [Cohere]). One input_layernorm feeds BOTH
//     the attention and the MLP, and both sublayer outputs are summed onto the raw
//     residual (FalconDecoderLayer.forward, the non-new-architecture parallel_attn
//     branch where mlp_layernorm_out = attention_layernorm_out):
//
//     residual = x; xn = input_layernorm(x); x = residual + attention(xn) + mlp(xn)
//
//   - Full LayerNorm WITH bias (not RMSNorm, not weight-only). Falcon's norms are
//     standard nn.LayerNorm carrying both γ and β — the block's input_layernorm and
//     the model's ln_f. (config.bias=False refers only to the LINEAR layers, never to
//     the LayerNorms, which always have a bias.)
//
//   - MULTI-QUERY attention (MQA) with a FUSED query_key_value weight — the crux.
//     self_attention.query_key_value.weight is a single [num_heads·head_dim +
//     2·head_dim, dim] matrix (bias-free). All num_heads query heads share ONE key
//     head and ONE value head. transformers views the fused output as [num_heads+2,
//     head_dim] and returns slices [:-2] as the queries, [-2] as the single key and
//     [-1] as the single value (FalconAttention._split_heads, multi_query branch). So
//     in row terms: q = rows [0 : num_heads·hd], k = rows [num_heads·hd :
//     (num_heads+1)·hd] (ONE head), v = rows [(num_heads+1)·hd : (num_heads+2)·hd]
//     (ONE head). [FalconFromHF] slices those contiguous row bands into Wq / Wk / Wv;
//     GoAI's OpMHA runs the MQA with AttnAttrs.KVHeads=1. Attention is bias-free
//     (query_key_value and dense both bias=False).
//
//   - Standard split-half FULL RoPE (alibi=False). Falcon's rotate_half is split-half
//     (the same convention as GoAI's OpRoPE), so no interleave permutation is needed
//     (contrast [Cohere]). The full head_dim is rotated with base rope_theta.
//
//   - GELU MLP, no bias: dense_h_to_4h → exact-erf GELU → dense_4h_to_h, both Linear
//     layers bias-free (config.activation defaults to "gelu", matching GoAI's OpGELU).
//
// Attention uses the standard 1/√head_dim score scale, is causal, and the LM head is
// UNTIED (lm_head.weight; tie_word_embeddings=false). Load a Hugging Face
// FalconForCausalLM checkpoint with [FalconFromHF].
type Falcon struct {
	Config    FalconConfig   // the model hyperparameters
	TokEmb    *tensor.Tensor // [vocab, dim] token embedding (transformer.word_embeddings)
	Blocks    []*FalconBlock // the stacked single-norm parallel-residual blocks
	FinalNorm *nn.LayerNorm  // transformer.ln_f (LayerNorm with bias)
	Out       *tensor.Tensor // [dim, vocab] output projection (lm_head, untied)
}

// FalconConfig fixes the model geometry. Dim, Vocab, Layers and Hidden are inferred
// from the checkpoint by [FalconFromHF]; the remaining fields come from config.json.
// Falcon-7B multi-query attention has exactly ONE key/value head, so KVHeads is fixed
// to 1 by the loader.
type FalconConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length (max_position_embeddings)
	Dim      int     // embedding width (hidden_size)
	Heads    int     // number of query heads (num_attention_heads)
	KVHeads  int     // key/value heads; Falcon MQA fixes this to 1 (0 → 1)
	Layers   int     // number of transformer blocks
	Hidden   int     // MLP inner width (ffn_hidden_size, typically 4·Dim)
	Eps      float64 // LayerNorm epsilon (layer_norm_epsilon, ~1e-5)
	RopeBase float64 // RoPE frequency base θ (rope_theta); 0 → 10000
}

// FalconBlock is one single-norm parallel-residual Falcon transformer block. InputNorm
// is the ONE per-block LayerNorm read by BOTH sublayers; the attention and MLP outputs
// are summed onto the raw residual. There is no post-norm and no separate MLP norm. All
// linear projections are bias-free; the MQA split gives Wk/Wv a single head each.
type FalconBlock struct {
	InputNorm      *nn.LayerNorm  // input_layernorm (LayerNorm with bias), feeds attention AND MLP
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wk/Wv are single-head (MQA)
	Wh, Wout       *tensor.Tensor // mlp.dense_h_to_4h / mlp.dense_4h_to_h (no bias)
}

func (c FalconConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return 1
	}
	return c.KVHeads
}

func (c FalconConfig) ropeBase() float64 {
	if c.RopeBase == 0 {
		return 10000
	}
	return c.RopeBase
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Falcon) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = m.hidden(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// embed gathers the token embeddings for the prompt (Falcon has no learned positional
// embedding and no embedding scale — position enters through RoPE inside attention).
func (m *Falcon) embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Falcon prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	return exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
}

// hidden runs the single-norm parallel-residual block stack and the final LayerNorm on
// an embedding x [seq, dim], returning the pre-logits hidden states [seq, dim].
func (m *Falcon) hidden(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenCapture(ctx, x, nil)
}

// hiddenCapture is hidden with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// POST-RoPE single-head k and raw single-head v [seq, headDim] (Falcon is MQA:
// KVHeads=1) — exactly the rows [Falcon.DecodeStep] appends per token, which is
// what lets [Falcon.Prefill] seed a cache from ONE batched pass over the prompt.
// The tensors handed to capture are the same ones the ongoing forward consumes
// (callers must not mutate them; the cache copies rows into its own buffers). A
// nil capture is the plain forward: no extra ops are recorded, so tape/training
// semantics of hidden are byte-identical to before this hook existed.
func (m *Falcon) hiddenCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: cfg.kvHeads(), Causal: true}
	for l, b := range m.Blocks {
		// Single-norm parallel residual: ONE norm feeds both sublayers; their outputs are
		// summed onto the raw residual. xn = input_layernorm(x); x = x + attn(xn) + mlp(xn).
		xn, err := b.InputNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var tap func(k, v *tensor.Tensor)
		if capture != nil {
			tap = func(k, v *tensor.Tensor) { capture(l, k, v) }
		}
		a, err := m.attention(ctx, b, xn, attn, tap)
		if err != nil {
			return nil, err
		}
		f, err := m.mlp(ctx, b, xn)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, a); err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, f); err != nil {
			return nil, err
		}
	}
	return m.FinalNorm.Forward(ctx, x)
}

// attention computes Falcon's multi-query attention over the normalized input xn
// [seq, dim]: project the (already de-fused) q/k/v — k and v have ONE head each — apply
// standard split-half full RoPE, then causal MQA via OpMHA (KVHeads=1) and the bias-free
// output dense. Mirrors FalconAttention.forward (alibi=False branch, no QK-norm).
// capture, when non-nil, receives the POST-RoPE single-head k and the raw single-head
// v — the per-token cache rows — for [Falcon.Prefill].
func (m *Falcon) attention(ctx *backend.Context, b *FalconBlock, xn *tensor.Tensor, attn backend.AttnAttrs, capture func(k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	q, err := project(ctx, xn, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := project(ctx, xn, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := project(ctx, xn, b.Wv)
	if err != nil {
		return nil, err
	}
	if q, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(ctx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.ropeBase(), Heads: kv}, k); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(k, v)
	}
	a, err := exec3(ctx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	return project(ctx, a, b.Wo)
}

// mlp runs Falcon's bias-free GELU feed-forward on the normalized input xn [seq, dim]:
// dense_h_to_4h → exact-erf GELU → dense_4h_to_h. Mirrors FalconMLP.forward.
func (m *Falcon) mlp(ctx *backend.Context, b *FalconBlock, xn *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := project(ctx, xn, b.Wh)
	if err != nil {
		return nil, err
	}
	if h, err = exec1a(ctx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	return project(ctx, h, b.Wout)
}

// Params returns every trainable tensor for optimizers.
func (m *Falcon) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.Wq, b.Wk, b.Wv, b.Wo, b.Wh, b.Wout)
		ps = append(ps, b.InputNorm.Params()...)
	}
	ps = append(ps, m.FinalNorm.Params()...)
	ps = append(ps, m.Out)
	return ps
}
