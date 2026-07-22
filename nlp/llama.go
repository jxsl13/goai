package nlp

import (
	"fmt"
	"math"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// Llama is the LLaMA / Llama-2 decoder-only transformer (Touvron et al. 2023,
// arXiv:2302.13971 / 2307.09288, §R92) assembled from GoAI's verified primitives. It
// is the modern-LLM counterpart to the GPT-2-style [GPT]: pre-normalization with
// RMSNorm, rotary position embeddings (RoPE) on the queries and keys, grouped-query
// attention, a SwiGLU feed-forward, and NO biases or learned positional embeddings.
// Per block (pre-norm residual):
//
//	h = x + Wo·Attn( RoPE(x̄·Wq), RoPE(x̄·Wk), x̄·Wv )   with x̄ = RMSNorm(x)
//	x = h + SwiGLU( RMSNorm(h) )
//
// then a final RMSNorm and the untied output projection produce the logits.
type Llama struct {
	Config LlamaConfig    // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim] token embedding (no positional embedding)
	Blocks []*LlamaBlock  // the stacked transformer blocks
	Norm   *nn.RMSNorm    // final pre-logits RMSNorm
	Out    *tensor.Tensor // [dim, vocab] output projection (untied)
	// LayerBackends optionally assigns each transformer block's batched forward
	// execution (including Prefill/ForwardHidden/ForwardResiduals) to a backend.
	// Set it to [backend.PlanOffload]'s Layers result to spill blocks that exceed
	// device memory onto CPU/ref. nil keeps the historical single-backend path;
	// a non-empty slice must contain exactly one name per block.
	LayerBackends []backend.Name
}

// LlamaConfig fixes the model geometry.
type LlamaConfig struct {
	Vocab    int     // vocabulary size
	Ctx      int     // max context length
	Dim      int     // embedding width (Heads·headDim)
	Heads    int     // number of query heads
	KVHeads  int     // key/value heads (GQA); 0 → Heads (standard MHA)
	Layers   int     // number of transformer blocks
	Hidden   int     // SwiGLU inner width (Llama uses ≈ (2/3)·4·Dim rounded)
	Eps      float64 // RMSNorm epsilon
	RopeBase float64 // RoPE frequency base θ; 0 → 10000

	// Optional IBM Granite scalar multipliers (transformers GraniteForCausalLM).
	// Granite is structurally a plain Llama plus four learned-at-config scalars; each
	// field is 0 ("unset") for Llama/Qwen2/Qwen3/Mistral, which makes every hook below a
	// byte-identical no-op. A GraniteConfig sets non-zero values via [GraniteConfigFromHF].
	EmbeddingMult float64 // inputs_embeds *= EmbeddingMult after lookup (Gemma-style, but a config scalar); 0 or 1 → identity
	AttentionMult float64 // pre-softmax attention scale = AttentionMult (replaces the default 1/√headDim); 0 → default
	ResidualMult  float64 // each residual add is x = residual + sublayer·ResidualMult (both attn and FFN); 0 or 1 → identity
	LogitsScale   float64 // final logits are divided by LogitsScale; 0 or 1 → identity
}

// LlamaBlock is one pre-norm Llama transformer block.
type LlamaBlock struct {
	AttnNorm       *nn.RMSNorm    // RMSNorm before attention
	Wq, Wk, Wv, Wo *tensor.Tensor // attention projections (no bias); Wk/Wv are [dim, KVHeads·headDim]
	Bq, Bk, Bv     *tensor.Tensor // optional q/k/v projection biases (Qwen2/Qwen2.5 family); nil for Llama/Mistral
	QNorm, KNorm   *nn.RMSNorm    // optional per-head query/key RMSNorm applied before RoPE (Qwen3); nil otherwise
	FFNNorm        *nn.RMSNorm    // RMSNorm before the FFN
	FFN            *nn.SwiGLU     // SwiGLU feed-forward
}

func (c LlamaConfig) kvHeads() int {
	if c.KVHeads <= 0 {
		return c.Heads
	}
	return c.KVHeads
}

// NewLlama builds a Llama with randomly initialized weights (Xavier for the
// projections, ones for the RMSNorm gains), for training or as a target to load
// weights into. It errors on an inconsistent geometry.
func NewLlama(cfg LlamaConfig, seed uint64) (*Llama, error) {
	if cfg.Heads <= 0 || cfg.Dim%cfg.Heads != 0 {
		return nil, fmt.Errorf("nlp: Llama dim %d not divisible by heads %d", cfg.Dim, cfg.Heads)
	}
	kv := cfg.kvHeads()
	if cfg.Heads%kv != 0 {
		return nil, fmt.Errorf("nlp: Llama heads %d not divisible by kv_heads %d", cfg.Heads, kv)
	}
	dk := cfg.Dim / cfg.Heads
	if dk%2 != 0 {
		return nil, fmt.Errorf("nlp: Llama head dim %d must be even for RoPE", dk)
	}
	kvDim := kv * dk
	m := &Llama{Config: cfg}
	m.TokEmb = tensor.New(tensor.F64, tensor.Shape{cfg.Vocab, cfg.Dim})
	nn.XavierUniform(m.TokEmb, cfg.Vocab, cfg.Dim, seed)
	s := seed + 1
	ones := func(n int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{n})
		for i := range n {
			t.SetF64(1, i)
		}
		return t
	}
	proj := func(in, out int) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{in, out})
		nn.XavierUniform(t, in, out, s)
		s++
		return t
	}
	for range cfg.Layers {
		m.Blocks = append(m.Blocks, &LlamaBlock{
			AttnNorm: &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			Wq:       proj(cfg.Dim, cfg.Dim),
			Wk:       proj(cfg.Dim, kvDim),
			Wv:       proj(cfg.Dim, kvDim),
			Wo:       proj(cfg.Dim, cfg.Dim),
			FFNNorm:  &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps},
			FFN:      nn.NewSwiGLU(tensor.F64, cfg.Dim, cfg.Hidden, s),
		})
		s += 3
	}
	m.Norm = &nn.RMSNorm{Gamma: ones(cfg.Dim), Eps: cfg.Eps}
	m.Out = proj(cfg.Dim, cfg.Vocab)
	return m, nil
}

// Embed gathers the token embeddings for the prompt (Llama has no positional
// embedding — position enters through RoPE inside attention).
func (m *Llama) Embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > m.Config.Ctx {
		return nil, fmt.Errorf("nlp: prompt length %d outside (0,%d]", seq, m.Config.Ctx)
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

// Forward computes logits [seq, vocab] for the prompt tokens.
func (m *Llama) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return m.ForwardFromEmbed(ctx, x)
}

// ForwardExit computes early-exit logits after the zero-based transformer block
// exitLayer. It executes only blocks 0 through exitLayer, then applies the model's
// shared final RMSNorm, output projection and any configured Granite logit scaling.
// Official LayerSkip checkpoints are ordinary Llama-family models trained so this
// shared readout is useful at intermediate layers; [LlamaFromHF] loads them without
// auxiliary weights. Choosing the final block is bit-identical to [Llama.Forward].
//
// The bounded stack is the draft primitive for [LlamaSelfSpeculativeDecode]. It
// preserves every hook supported by the shared Llama block path: GQA, Qwen2 biases,
// Qwen3 QK-norm, and Granite multipliers.
func (m *Llama) ForwardExit(ctx *backend.Context, tokens []int, exitLayer int) (*tensor.Tensor, error) {
	if m == nil {
		return nil, fmt.Errorf("nlp: Llama.ForwardExit needs a model")
	}
	if exitLayer < 0 || exitLayer >= len(m.Blocks) {
		return nil, fmt.Errorf("nlp: Llama.ForwardExit exitLayer %d outside [0,%d)", exitLayer, len(m.Blocks))
	}
	if err := validateLayerBackends(m.LayerBackends, len(m.Blocks)); err != nil {
		return nil, err
	}
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	if x, err = scaleScalar(ctx, x, m.Config.EmbeddingMult); err != nil {
		return nil, err
	}
	for layer := 0; layer <= exitLayer; layer++ {
		if x, err = m.forwardBlock(ctx, x, layer, nil); err != nil {
			return nil, err
		}
	}
	return m.Unembed(ctx, x)
}

// ForwardHidden returns the final hidden states [seq, dim] — the residual stream after
// all blocks and the final RMSNorm, i.e. Forward WITHOUT the output projection (§T447,
// the Llama sibling of GPT.ForwardHidden). This is the representation auxiliary decoding
// heads attach to (Medusa): Forward(tokens) ≡ ForwardHidden(tokens)·Out.
func (m *Llama) ForwardHidden(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return m.hiddenFromEmbed(ctx, x)
}

// ForwardFromEmbed runs the transformer blocks, final RMSNorm and output projection on
// a precomputed embedding x [seq, dim], returning logits [seq, vocab]. Splitting the
// embedding step out lets a training loop inject NEFTune noise (nn.NEFTune) between the
// embedding and the blocks.
func (m *Llama) ForwardFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.hiddenFromEmbed(ctx, x)
	if err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, h, m.Out)
	if err != nil {
		return nil, err
	}
	// Granite: logits /= logits_scaling (no-op for Llama, LogitsScale 0).
	return divLogits(ctx, logits, m.Config.LogitsScale)
}

// Unembed maps residual-stream rows h [n, dim] to logits [n, vocab] through the
// model's final RMSNorm, output projection and (Granite) logits scaling — the
// [LensReadoutModel] seam (§T812): Unembed applied to the pre-final-norm
// residual reproduces Forward's logits exactly, and the J-lens borrows it to
// decode transported activations with the model's own head.
func (m *Llama) Unembed(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	x, err := m.Norm.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	logits, err := exec1(ctx, backend.OpMatMul, nil, x, m.Out)
	if err != nil {
		return nil, err
	}
	return divLogits(ctx, logits, m.Config.LogitsScale)
}

// scaleScalar multiplies x by the scalar s via a rank-0 OpMul broadcast when s is a set,
// non-identity Granite multiplier (s != 0 && s != 1); an unset (0) or identity (1) s
// returns x unchanged, so the plain-Llama path (all Granite scalars 0) is byte-identical.
// Used for Granite's embedding and residual multipliers.
func scaleScalar(ctx *backend.Context, x *tensor.Tensor, s float64) (*tensor.Tensor, error) {
	if s == 0 || s == 1 {
		return x, nil
	}
	// The scalar must match x's dtype: the f64 decoders run an f64 residual stream,
	// but QuantLlama runs f32 (its embedding table is f32), and OpMul rejects a dtype
	// mismatch. SetF64 casts into the storage dtype, so f64 paths are unchanged.
	sc := tensor.New(x.Dtype(), tensor.Shape{})
	sc.SetF64(s)
	return exec1(ctx, backend.OpMul, nil, x, sc)
}

// divLogits divides logits by the Granite logits_scaling (multiply by 1/logitsScale),
// a no-op when logitsScale is 0 (unset) or 1 (identity).
func divLogits(ctx *backend.Context, logits *tensor.Tensor, logitsScale float64) (*tensor.Tensor, error) {
	if logitsScale == 0 || logitsScale == 1 {
		return logits, nil
	}
	return scaleScalar(ctx, logits, 1/logitsScale)
}

// hiddenFromEmbed is the shared block stack + final RMSNorm.
func (m *Llama) hiddenFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.hiddenFromEmbedCapture(ctx, x, nil)
}

// hiddenFromEmbedCapture is hiddenFromEmbed with an optional per-layer KV tap.
// When capture is non-nil it is invoked once per block with the layer index and
// that block's post-bias, post-QK-norm, POST-RoPE k and v [seq, kvWidth] — exactly
// the rows a KV-cache decode appends per token (DecodeStep), which is what lets
// Prefill seed a cache from ONE batched pass over the prompt. The tensors handed
// to capture are the same ones the ongoing forward consumes (callers must not
// mutate them; the cache copies rows into its own buffers). A nil capture is the
// plain forward: no extra ops are recorded, so tape/training semantics of
// hiddenFromEmbed are byte-identical to before this hook existed.
func (m *Llama) hiddenFromEmbedCapture(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	return m.hiddenFromEmbedTaps(ctx, x, capture, nil)
}

// ForwardResiduals is Forward's residual-stream observer (§T810, the J-lens
// foundation §R250): it runs the full block stack and hands every residual-stream
// snapshot h_l [seq, dim] to capture — layer 0 is the post-embedding block input
// h_0, layer l ∈ 1..Layers the residual AFTER block l (PRE final norm), so a
// non-nil capture fires Layers+1 times in order. The returned tensor is the
// post-final-RMSNorm hidden states, identical to ForwardHidden. The tensors handed
// to capture are the LIVE ones the ongoing forward consumes: observers must not
// mutate them (a test-only injector may, on an eager backend, at its own risk).
// A nil capture records exactly the same ops as ForwardHidden — byte-identical,
// same discipline as the KV tap above.
func (m *Llama) ForwardResiduals(ctx *backend.Context, tokens []int, capture ResidualCapture) (*tensor.Tensor, error) {
	x, err := m.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return m.hiddenFromEmbedTaps(ctx, x, nil, capture)
}

// hiddenFromEmbedTaps is the single block-stack implementation behind
// hiddenFromEmbed (both taps nil), hiddenFromEmbedCapture (KV tap) and
// ForwardResiduals (residual tap). Both taps are pure observers: nil taps make
// this byte-identical to the historical hiddenFromEmbed (no extra ops recorded).
func (m *Llama) hiddenFromEmbedTaps(ctx *backend.Context, x *tensor.Tensor, capture func(layer int, k, v *tensor.Tensor), resid ResidualCapture) (*tensor.Tensor, error) {
	cfg := m.Config
	if err := validateLayerBackends(m.LayerBackends, len(m.Blocks)); err != nil {
		return nil, err
	}

	var err error
	// Granite: inputs_embeds *= embedding_multiplier (no-op for Llama, EmbeddingMult 0).
	if x, err = scaleScalar(ctx, x, cfg.EmbeddingMult); err != nil {
		return nil, err
	}
	if resid != nil {
		resid(0, x) // h_0: the post-embedding (post-Granite-scale) input to block 0
	}
	for l := range m.Blocks {
		if x, err = m.forwardBlock(ctx, x, l, capture); err != nil {
			return nil, err
		}
		if resid != nil {
			resid(l+1, x) // h_{l+1}: the residual stream after block l
		}
	}
	if x, err = m.Norm.Forward(ctx, x); err != nil {
		return nil, err
	}
	return x, nil
}

// forwardBlock applies one Llama-family block through its configured layer backend.
// The optional capture observes the post-RoPE key and value rows used by Prefill.
// Forward, ForwardExit, residual observation and prefill all share this exact path.
func (m *Llama) forwardBlock(ctx *backend.Context, x *tensor.Tensor, layer int, capture func(layer int, k, v *tensor.Tensor)) (*tensor.Tensor, error) {
	cfg := m.Config
	kv := cfg.kvHeads()
	b := m.Blocks[layer]
	layerCtx := contextForLayer(ctx, m.LayerBackends, layer)
	attn := backend.AttnAttrs{Heads: cfg.Heads, KVHeads: kv, Causal: true}
	// Granite: replace the default 1/√headDim pre-softmax scale with attention_multiplier.
	// OpMHA builds in a 1/√headDim, and AttnAttrs.Scale is an EXTRA factor on top of it
	// (as T5 uses it), so Scale = AttentionMult·√headDim leaves a net scale of AttentionMult.
	if cfg.AttentionMult != 0 {
		attn.Scale = cfg.AttentionMult * math.Sqrt(float64(cfg.Dim/cfg.Heads))
	}

	// attention sublayer: x += Wo·Attn(RoPE(Wq·x̄), RoPE(Wk·x̄), Wv·x̄), x̄=RMSNorm(x)
	xb, err := b.AttnNorm.Forward(layerCtx, x)
	if err != nil {
		return nil, err
	}
	q, err := project(layerCtx, xb, b.Wq)
	if err != nil {
		return nil, err
	}
	k, err := project(layerCtx, xb, b.Wk)
	if err != nil {
		return nil, err
	}
	v, err := project(layerCtx, xb, b.Wv)
	if err != nil {
		return nil, err
	}
	// Qwen2-family q/k/v projection biases (added before RoPE); nil for Llama/Mistral.
	if q, err = addBiasIf(layerCtx, q, b.Bq); err != nil {
		return nil, err
	}
	if k, err = addBiasIf(layerCtx, k, b.Bk); err != nil {
		return nil, err
	}
	if v, err = addBiasIf(layerCtx, v, b.Bv); err != nil {
		return nil, err
	}
	// Qwen3 per-head QK-norm (before RoPE); nil for Llama/Qwen2.
	if q, err = applyQKNorm(layerCtx, q, b.QNorm, cfg.Heads); err != nil {
		return nil, err
	}
	if k, err = applyQKNorm(layerCtx, k, b.KNorm, kv); err != nil {
		return nil, err
	}
	if q, err = exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: cfg.Heads}, q); err != nil {
		return nil, err
	}
	if k, err = exec1(layerCtx, backend.OpRoPE, backend.RoPEAttrs{Base: cfg.RopeBase, Heads: kv}, k); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(layer, k, v)
	}
	a, err := exec1(layerCtx, backend.OpMHA, attn, q, k, v)
	if err != nil {
		return nil, err
	}
	o, err := project(layerCtx, a, b.Wo)
	if err != nil {
		return nil, err
	}
	// Granite: x = x + o·residual_multiplier (no-op for Llama, ResidualMult 0).
	if o, err = scaleScalar(layerCtx, o, cfg.ResidualMult); err != nil {
		return nil, err
	}
	if x, err = exec1(layerCtx, backend.OpAdd, nil, x, o); err != nil {
		return nil, err
	}

	// FFN sublayer: x += SwiGLU(RMSNorm(x))
	xf, err := b.FFNNorm.Forward(layerCtx, x)
	if err != nil {
		return nil, err
	}
	ff, err := b.FFN.Forward(layerCtx, xf)
	if err != nil {
		return nil, err
	}
	// Granite: x = x + ff·residual_multiplier (same scalar as the attention residual).
	if ff, err = scaleScalar(layerCtx, ff, cfg.ResidualMult); err != nil {
		return nil, err
	}
	return exec1(layerCtx, backend.OpAdd, nil, x, ff)
}

// project computes x·W (a bias-free linear layer).
func project(ctx *backend.Context, x, w *tensor.Tensor) (*tensor.Tensor, error) {
	return exec2(ctx, backend.OpMatMul, nil, x, w)
}

// addBiasIf adds bias to x when bias is non-nil (Qwen2-family projections), else
// returns x unchanged (the bias-free Llama/Mistral path).
func addBiasIf(ctx *backend.Context, x, bias *tensor.Tensor) (*tensor.Tensor, error) {
	if bias == nil {
		return x, nil
	}
	return exec1(ctx, backend.OpAddBias, nil, x, bias)
}

// applyQKNorm applies a per-head RMSNorm to a projected q or k tensor x [seq, heads·headDim]
// when norm is non-nil (Qwen3 QK-norm; OLMo2 uses a separate full-width variant, see [OLMo2]), else returns x unchanged. The tensor is
// reshaped to [seq·heads, headDim] so RMSNorm (last-axis) normalizes each head independently,
// then reshaped back — matching HF's q_norm(q_proj(x).view(.., heads, headDim)). Both reshapes
// are differentiable (OpReshape has a VJP), so QK-norm models remain trainable.
func applyQKNorm(ctx *backend.Context, x *tensor.Tensor, norm *nn.RMSNorm, heads int) (*tensor.Tensor, error) {
	if norm == nil {
		return x, nil
	}
	seq := x.Shape()[0]
	hd := x.Shape()[1] / heads
	r, err := exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq * heads, hd}}, x)
	if err != nil {
		return nil, err
	}
	if r, err = norm.Forward(ctx, r); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpReshape, backend.ReshapeAttrs{Shape: tensor.Shape{seq, heads * hd}}, r)
}

// Params returns every trainable tensor for optimizers.
func (m *Llama) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.TokEmb}
	for _, b := range m.Blocks {
		ps = append(ps, b.AttnNorm.Gamma, b.Wq, b.Wk, b.Wv, b.Wo, b.FFNNorm.Gamma)
		for _, bb := range []*tensor.Tensor{b.Bq, b.Bk, b.Bv} {
			if bb != nil {
				ps = append(ps, bb)
			}
		}
		for _, n := range []*nn.RMSNorm{b.QNorm, b.KNorm} {
			if n != nil {
				ps = append(ps, n.Gamma)
			}
		}
		ps = append(ps, b.FFN.Params()...)
	}
	ps = append(ps, m.Norm.Gamma, m.Out)
	return ps
}
