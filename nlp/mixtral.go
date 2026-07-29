package nlp

import (
	"fmt"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Mixtral is the Mixtral sparse-Mixture-of-Experts decoder (Jiang et al. 2024,
// arXiv:2401.04088). Its attention is identical to [Llama] — RMSNorm, RoPE,
// grouped-query attention, no biases — but each block's feed-forward is a sparse
// MoE: a router picks the top-2 of N SwiGLU experts per token and mixes their
// outputs with the renormalized router weights (nn.SparseMoE). This is the
// modern MoE counterpart to the dense Llama FFN; load a Hugging Face checkpoint
// with [MixtralFromHF].
type Mixtral struct {
	Config MixtralConfig   // geometry: dims, heads, expert count (see MixtralConfig)
	TokEmb *tensor.Tensor  // [vocab, dim] token embedding
	Blocks []*MixtralBlock // the attention + sparse-MoE blocks
	Norm   *nn.RMSNorm     // final pre-logits RMSNorm (model.norm)
	Out    *tensor.Tensor  // [dim, vocab] output projection
}

// MixtralConfig fixes the model geometry. Dim, Vocab, Layers, Hidden, Experts are
// inferred from the checkpoint by MixtralFromHF; the rest come from config.json.
type MixtralConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // maximum context length in tokens
	Dim      int     // embedding width (hidden_size)
	Heads    int     // query heads (num_attention_heads)
	KVHeads  int     // GQA; 0 → Heads
	Layers   int     // number of decoder layers
	Hidden   int     // per-expert SwiGLU inner width
	Experts  int     // number of experts (num_local_experts)
	TopK     int     // experts per token (num_experts_per_tok); 0 → 2
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE θ; 0 → 10000
}

// MixtralBlock is one pre-norm Mixtral block: Llama-style attention + a sparse-MoE FFN.
// QNorm/KNorm are optional per-head query/key RMSNorms applied before RoPE — nil for
// plain Mixtral, non-nil for Qwen3-MoE (see [Qwen3MoeFromHF]).
type MixtralBlock struct {
	AttnNorm       *nn.RMSNorm    // RMSNorm before attention (input_layernorm)
	Wq, Wk, Wv, Wo *tensor.Tensor // bias-free attention projections (q/k/v/o)
	QNorm, KNorm   *nn.RMSNorm    // optional per-head QK-norm before RoPE (Qwen3-MoE); nil otherwise
	FFNNorm        *nn.RMSNorm    // RMSNorm before the MoE FFN (post_attention_layernorm)
	MoE            *nn.SparseMoE  // sparse top-k SwiGLU expert bank
}

func (c MixtralConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Mixtral) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	return m.forwardCapture(ctx, tokens, nil, false)
}

// forwardCapture is Forward with an optional per-layer KV tap. When capture is
// non-nil it is invoked once per block with the layer index and that block's
// post-QK-norm, POST-RoPE k and raw v [seq, kvWidth] — exactly the rows
// [Mixtral.DecodeStep] appends per token, which is what lets [Mixtral.Prefill]
// seed a cache from ONE batched pass over the prompt. The tensors handed to
// capture are the same ones the ongoing forward consumes (callers must not
// mutate them; the cache copies rows into its own buffers). A nil capture with
// sparseFFN=false is the plain forward: no extra ops are recorded, so
// tape/training semantics of Forward are byte-identical to before this hook
// existed.
//
// sparseFFN selects the MoE evaluation path: false is the dense b.MoE.Forward
// (training/tape; every expert evaluated), true is the same inference-only
// multi-row b.MoE.ForwardDecode DecodeStep uses. The two are mathematically
// identical (same renormalized top-k routing), but NOT bit-identical: the dense
// OpMoECombine kernel accumulates acc += (w/denom)·e in one fused expression,
// which Go may compile to an FMA (spec-sanctioned fusion, arm64 FMADD), while
// the sparse path rounds the product through a stored OpMul tensor before the
// OpAdd — a ~1-ulp divergence. Prefill therefore passes sparseFFN=true so its
// residual stream (and thus every later layer's cached k/v) matches DecodeStep
// exactly, bit for bit.
func (m *Mixtral) forwardCapture(ctx *backend.Context, tokens []int, capture func(layer int, k, v *tensor.Tensor), sparseFFN bool) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: Mixtral prompt length %d outside (0,%d]", seq, m.Config.Ctx)
	}
	idx := tensor.New(m.TokEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		if t < 0 || t >= m.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, m.Config.Vocab)
		}
		idx.SetF64(float64(t), i)
	}
	x, err := exec1(ctx, backend.OpEmbed, nil, m.TokEmb, idx)
	if err != nil {
		return nil, err
	}

	cfg := m.Config
	kv := cfg.kvHeads()
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	// Box these attrs into the Attrs INTERFACE once per token, above the layer loop: the values
	// are layer-independent, and as concrete structs handed to an interface parameter inside the
	// loop each was heap-boxed once per layer per token (escape analysis named every one).
	// exec1a/exec3 also pool their input slices, and only when ctx.Recorder == nil, so a taped
	// training context keeps the fresh-slice path.
	qRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads})
	kRoPE := backend.Attrs(backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv})
	for l, b := range m.Blocks {
		// attention sublayer (identical to Llama)
		xb, err := b.AttnNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		q, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wq)
		if err != nil {
			return nil, err
		}
		k, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wk)
		if err != nil {
			return nil, err
		}
		v, err := exec1(ctx, backend.OpMatMul, nil, xb, b.Wv)
		if err != nil {
			return nil, err
		}
		// Qwen3-MoE per-head QK-norm (before RoPE); nil for plain Mixtral (no-op).
		if q, err = applyQKNorm(ctx, q, b.QNorm, cfg.Heads); err != nil {
			return nil, err
		}
		if k, err = applyQKNorm(ctx, k, b.KNorm, kv); err != nil {
			return nil, err
		}
		if q, err = exec1a(ctx, backend.OpRoPE, qRoPE, q); err != nil {
			return nil, err
		}
		if k, err = exec1a(ctx, backend.OpRoPE, kRoPE, k); err != nil {
			return nil, err
		}
		if capture != nil {
			capture(l, k, v)
		}
		a, err := exec1(ctx, backend.OpMHA, attn, q, k, v)
		if err != nil {
			return nil, err
		}
		o, err := exec1(ctx, backend.OpMatMul, nil, a, b.Wo)
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, o); err != nil {
			return nil, err
		}
		// sparse-MoE FFN sublayer (dense for training, ForwardDecode for prefill —
		// see the sparseFFN contract in the method comment)
		xf, err := b.FFNNorm.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		var ff *tensor.Tensor
		if sparseFFN {
			ff, _, err = b.MoE.ForwardDecode(ctx, xf)
		} else {
			ff, _, err = b.MoE.Forward(ctx, xf)
		}
		if err != nil {
			return nil, err
		}
		if x, err = exec2(ctx, backend.OpAdd, nil, x, ff); err != nil {
			return nil, err
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, m.Out)
}

// Params returns every trainable tensor for optimizers (router + all experts included).
func (m *Mixtral) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma, b.MoE.Router.W)
		for _, n := range []*nn.RMSNorm{b.QNorm, b.KNorm} {
			if n != nil {
				ps = append(ps, n.Gamma)
			}
		}
		for _, e := range b.MoE.Experts {
			ps = append(ps, e.Wgate, e.Wup, e.Wdown)
		}
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
