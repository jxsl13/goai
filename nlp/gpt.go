package nlp

import (
	"fmt"
	"sync"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// GPT is a decoder-only transformer for inference (§T23), pre-LN architecture
// (GPT-2 style):
//
//	x = tokEmb[tokens] + posEmb[:seq]
//	per block: x += CausalMHA(LN1(x));  x += FFN(LN2(x)) with GELU
//	logits = LNf(x) · Head
type GPT struct {
	Config GPTConfig      // the model hyperparameters
	TokEmb *tensor.Tensor // [vocab, dim]
	PosEmb *tensor.Tensor // [ctx, dim]
	Blocks []*Block       // the stacked transformer blocks
	LNf    *nn.LayerNorm  // final pre-logits LayerNorm
	Head   *tensor.Tensor // [dim, vocab]
	// LayerBackends optionally assigns each transformer block's batched forward
	// execution (Forward/ForwardHidden/ForwardResiduals) to a backend.
	// Set it to [backend.PlanOffload]'s Layers result to spill blocks that exceed
	// device memory onto CPU/ref. nil keeps the historical single-backend path;
	// a non-empty slice must contain exactly one name per block.
	LayerBackends []backend.Name
}

// GPTConfig fixes the model geometry.
type GPTConfig struct {
	Vocab  int     // vocabulary size
	Ctx    int     // max context length
	Dim    int     // embedding width
	Heads  int     // number of attention heads
	Layers int     // number of transformer blocks
	Eps    float64 // LayerNorm epsilon
}

// Block is one pre-LN transformer block.
type Block struct {
	LN1, LN2 *nn.LayerNorm  // LN1 pre-attention, LN2 pre-MLP LayerNorm
	Attn     *MHA           // multi-head self-attention
	W1, B1   *tensor.Tensor // FFN up: [dim, 4dim], [4dim]
	W2, B2   *tensor.Tensor // FFN down: [4dim, dim], [dim]
}

// FromSafetensors assembles a GPT from a tensor map using the goai naming
// convention: tok_emb, pos_emb, blocks.{i}.{ln1,ln2}.{gamma,beta},
// blocks.{i}.attn.{wq,wk,wv,wo}, blocks.{i}.ffn.{w1,b1,w2,b2}, lnf.{gamma,beta},
// head. Weight layout is [in,out] (§B19: transpose torch [out,in] on import).
func FromSafetensors(cfg GPTConfig, ts map[string]*tensor.Tensor) (*GPT, error) {
	get := func(name string) (*tensor.Tensor, error) {
		t := ts[name]
		if t == nil {
			return nil, fmt.Errorf("nlp: missing tensor %q", name)
		}
		return t, nil
	}
	tok, err := get("tok_emb")
	if err != nil {
		return nil, err
	}
	pos, err := get("pos_emb")
	if err != nil {
		return nil, err
	}
	g := &GPT{Config: cfg, TokEmb: tok, PosEmb: pos}

	for l := range cfg.Layers {
		p := fmt.Sprintf("blocks.%d.", l)
		names := []string{
			p + "ln1.gamma", p + "ln1.beta",
			p + "attn.wq", p + "attn.wk", p + "attn.wv", p + "attn.wo",
			p + "ln2.gamma", p + "ln2.beta",
			p + "ffn.w1", p + "ffn.b1", p + "ffn.w2", p + "ffn.b2",
		}
		got := make([]*tensor.Tensor, len(names))
		for i, n := range names {
			if got[i], err = get(n); err != nil {
				return nil, err
			}
		}
		attn, err := NewMHA(cfg.Heads, got[2], got[3], got[4], got[5])
		if err != nil {
			return nil, err
		}
		attn.Causal = true
		g.Blocks = append(g.Blocks, &Block{
			LN1:  &nn.LayerNorm{Gamma: got[0], Beta: got[1], Eps: cfg.Eps},
			Attn: attn,
			LN2:  &nn.LayerNorm{Gamma: got[6], Beta: got[7], Eps: cfg.Eps},
			W1:   got[8], B1: got[9], W2: got[10], B2: got[11],
		})
	}
	if lg, err := get("lnf.gamma"); err == nil {
		lb, err2 := get("lnf.beta")
		if err2 != nil {
			return nil, err2
		}
		g.LNf = &nn.LayerNorm{Gamma: lg, Beta: lb, Eps: cfg.Eps}
	} else {
		return nil, err
	}
	if g.Head, err = get("head"); err != nil {
		return nil, err
	}
	return g, nil
}

func exec1(ctx *backend.Context, op backend.Op, attrs backend.Attrs, ins ...*tensor.Tensor) (*tensor.Tensor, error) {
	out, err := backend.Execute(ctx, op, ins, attrs)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// ins2Pool reuses the 2-element input slice backend.Execute takes, for 2-input ops on
// the INFERENCE decode/prefill hot path (T960). backend.Execute retains its inputs slice
// ONLY through ctx.Recorder.Record (execute.go), which fires only when a tape is attached;
// with a nil Recorder — every DecodeStep/Prefill context is inference-only — the slice is
// dead the instant Execute returns, so it goes straight back to the pool.
var ins2Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 2); return &s }}

// exec2 runs a 2-input op, reusing a pooled input slice when the context is NOT recording.
// Under a recorder it defers to exec1's fresh slice, because Record stores that exact slice
// in the tape node and a pooled one would be overwritten by the next op (a training-only
// correctness trap the guard closes). project — 4 calls per layer per token, shared by every
// model — is the hot caller, so one change speeds every decode.
func exec2(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return exec1(ctx, op, attrs, a, b)
	}
	sp := ins2Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1] = a, b
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1] = nil, nil // don't pin the input tensors alive in the pool
	ins2Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// ins1Pool / ins3Pool are the 1- and 3-input siblings of ins2Pool for the same
// inference hot path (T962): RoPE is a single-input op (2 per layer per token) and MHA
// a 3-input op (1 per layer). Same recorder-guarded reuse — pool only when nothing is
// taping, so backend.Execute never keeps the slice past the call.
var (
	ins1Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 1); return &s }}
	ins3Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 3); return &s }}
	// ins4Pool serves the masked-attention ops, which take query, key, value and a mask or
	// relative-position bias. Without a 4-input sibling those five call sites were the only
	// per-iteration variadic allocations in this family that PS6017 could not see — it reports
	// a variadic call only when a non-variadic sibling of that arity exists, so the gap was in
	// the helper set, not in the check.
	ins4Pool = sync.Pool{New: func() any { s := make([]*tensor.Tensor, 4); return &s }}
)

// exec1a runs a 1-input op with a pooled input slice when not recording (RoPE). See exec2.
func exec1a(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return exec1(ctx, op, attrs, a)
	}
	sp := ins1Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0] = a
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0] = nil
	ins1Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// exec3 runs a 3-input op with a pooled input slice when not recording (MHA q,k,v). See exec2.
func exec3(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b, c *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return exec1(ctx, op, attrs, a, b, c)
	}
	sp := ins3Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1], s[2] = a, b, c
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1], s[2] = nil, nil, nil
	ins3Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// exec4 is the 4-input sibling, for the masked-attention ops that take query, key, value and
// a mask or relative-position bias. Same contract as exec2/exec3: under a recorder it defers
// to the variadic form so the tape sees a slice it may retain, and otherwise it borrows a
// pooled 4-element slice, nils the entries before returning it, and calls the identical
// backend.Execute. Bit-identical by construction — only the provenance of the input slice
// changes.
func exec4(ctx *backend.Context, op backend.Op, attrs backend.Attrs, a, b, c, d *tensor.Tensor) (*tensor.Tensor, error) {
	if ctx.Recorder != nil {
		return exec1(ctx, op, attrs, a, b, c, d)
	}
	sp := ins4Pool.Get().(*[]*tensor.Tensor)
	s := *sp
	s[0], s[1], s[2], s[3] = a, b, c, d
	out, err := backend.Execute(ctx, op, s, attrs)
	s[0], s[1], s[2], s[3] = nil, nil, nil, nil
	ins4Pool.Put(sp)
	if err != nil {
		return nil, err
	}
	return out[0], nil
}

// Embed gathers token+position embeddings for the prompt through the dispatch,
// so gradients flow to TokEmb/PosEmb (differentiable, §T34): x = Embed(TokEmb,
// tokens) + Embed(PosEmb, 0..seq−1).
func (g *GPT) Embed(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	seq := len(tokens)
	if seq == 0 || seq > g.Config.Ctx {
		return nil, fmt.Errorf("nlp: prompt length %d outside (0,%d]", seq, g.Config.Ctx)
	}
	for _, t := range tokens {
		if t < 0 || t >= g.Config.Vocab {
			return nil, fmt.Errorf("nlp: token %d outside vocab %d", t, g.Config.Vocab)
		}
	}
	tokIdx := tensor.New(g.TokEmb.Dtype(), tensor.Shape{seq})
	posIdx := tensor.New(g.PosEmb.Dtype(), tensor.Shape{seq})
	for i, t := range tokens {
		tokIdx.SetF64(float64(t), i)
		posIdx.SetF64(float64(i), i)
	}
	et, err := exec1(ctx, backend.OpEmbed, nil, g.TokEmb, tokIdx)
	if err != nil {
		return nil, err
	}
	ep, err := exec1(ctx, backend.OpEmbed, nil, g.PosEmb, posIdx)
	if err != nil {
		return nil, err
	}
	return exec2(ctx, backend.OpAdd, nil, et, ep)
}

// Params returns every trainable tensor (token/pos embeddings, all block
// weights, final LN, and the LM head) for optimizers.
func (g *GPT) Params() []*tensor.Tensor {
	ps := []*tensor.Tensor{g.TokEmb, g.PosEmb}
	for _, b := range g.Blocks {
		ps = append(ps,
			b.LN1.Gamma, b.LN1.Beta,
			b.Attn.Wq, b.Attn.Wk, b.Attn.Wv, b.Attn.Wo,
			b.LN2.Gamma, b.LN2.Beta,
			b.W1, b.B1, b.W2, b.B2,
		)
	}
	ps = append(ps, g.LNf.Gamma, g.LNf.Beta, g.Head)
	return ps
}

// Safetensors returns the model's parameters under the FromSafetensors naming
// convention — the exact inverse of FromSafetensors — so a trained model can be
// checkpointed with safetensors.Save/SaveFile and reloaded bit-identically. The
// map holds the model's LIVE tensors (no copy): serialize before mutating.
func (g *GPT) Safetensors() map[string]*tensor.Tensor {
	ts := map[string]*tensor.Tensor{
		"tok_emb":   g.TokEmb,
		"pos_emb":   g.PosEmb,
		"lnf.gamma": g.LNf.Gamma,
		"lnf.beta":  g.LNf.Beta,
		"head":      g.Head,
	}
	for l, b := range g.Blocks {
		p := fmt.Sprintf("blocks.%d.", l)
		ts[p+"ln1.gamma"] = b.LN1.Gamma
		ts[p+"ln1.beta"] = b.LN1.Beta
		ts[p+"attn.wq"] = b.Attn.Wq
		ts[p+"attn.wk"] = b.Attn.Wk
		ts[p+"attn.wv"] = b.Attn.Wv
		ts[p+"attn.wo"] = b.Attn.Wo
		ts[p+"ln2.gamma"] = b.LN2.Gamma
		ts[p+"ln2.beta"] = b.LN2.Beta
		ts[p+"ffn.w1"] = b.W1
		ts[p+"ffn.b1"] = b.B1
		ts[p+"ffn.w2"] = b.W2
		ts[p+"ffn.b2"] = b.B2
	}
	return ts
}

// Forward computes logits [seq, vocab] for the prompt tokens.
func (g *GPT) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return g.ForwardFromEmbed(ctx, x)
}

// ForwardExit computes early-exit logits after the zero-based transformer block
// exitLayer. It executes only blocks 0 through exitLayer, then applies the model's
// shared final LayerNorm and LM head. LayerSkip-style checkpoints train that shared
// exit to make intermediate predictions useful; an ordinary GPT is valid input but
// may produce poor draft acceptance. Choosing the final block is bit-identical to
// [GPT.Forward].
//
// Unlike [GPT.ForwardEarlyExit], this method stops the block stack at the requested
// layer instead of also completing the mature forward. It is therefore the cheap
// draft primitive for [SelfSpeculativeDecode].
func (g *GPT) ForwardExit(ctx *backend.Context, tokens []int, exitLayer int) (*tensor.Tensor, error) {
	if g == nil {
		return nil, fmt.Errorf("nlp: ForwardExit needs a model")
	}
	if exitLayer < 0 || exitLayer >= len(g.Blocks) {
		return nil, fmt.Errorf("nlp: ForwardExit exitLayer %d outside [0,%d)", exitLayer, len(g.Blocks))
	}
	if err := validateLayerBackends(g.LayerBackends, len(g.Blocks)); err != nil {
		return nil, err
	}
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	for layer := 0; layer <= exitLayer; layer++ {
		if x, err = g.forwardBlock(ctx, x, layer); err != nil {
			return nil, err
		}
	}
	return g.Unembed(ctx, x)
}

// ForwardFromEmbed runs the transformer blocks and LM head on a precomputed embedding
// x [seq, dim], returning logits [seq, vocab]. Splitting the embedding step out lets a
// training loop inject NEFTune noise (nn.NEFTune) between the embedding and the blocks.
func (g *GPT) ForwardFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := g.hiddenFromEmbed(ctx, x)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, h, g.Head)
}

// ForwardHidden returns the final hidden states [seq, dim] — the residual stream after
// all blocks and the final LayerNorm, i.e. Forward WITHOUT the LM head (§T443). This is
// the representation auxiliary decoding heads attach to (Medusa, early-exit probes):
// Forward(tokens) ≡ ForwardHidden(tokens)·Head.
func (g *GPT) ForwardHidden(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return g.hiddenFromEmbed(ctx, x)
}

// hiddenFromEmbed is the shared block stack + final LayerNorm.
func (g *GPT) hiddenFromEmbed(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	return g.hiddenFromEmbedCapture(ctx, x, nil)
}

// ForwardResiduals is Forward's residual-stream observer (§T810, the J-lens
// foundation §R250): it runs the full block stack and hands every residual-stream
// snapshot h_l [seq, dim] to capture — layer 0 is the post-embedding block input
// h_0 (token+position embedding), layer l ∈ 1..Layers the residual AFTER block l
// (PRE final LayerNorm), so a non-nil capture fires Layers+1 times in order. The
// returned tensor is the post-LNf hidden states, identical to ForwardHidden. The
// tensors handed to capture are the LIVE ones the ongoing forward consumes:
// observers must not mutate them (a test-only injector may, on an eager backend,
// at its own risk). A nil capture records exactly the same ops as ForwardHidden —
// byte-identical, same discipline as Llama's KV tap (§T810).
func (g *GPT) ForwardResiduals(ctx *backend.Context, tokens []int, capture ResidualCapture) (*tensor.Tensor, error) {
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	return g.hiddenFromEmbedCapture(ctx, x, capture)
}

// Unembed maps residual-stream rows h [n, dim] to logits [n, vocab] through the
// model's final LayerNorm and LM head — the [LensReadoutModel] seam (§T812):
// Unembed applied to the pre-LNf residual reproduces Forward's logits exactly,
// and the J-lens borrows it to decode transported activations with the model's
// own head.
func (g *GPT) Unembed(ctx *backend.Context, h *tensor.Tensor) (*tensor.Tensor, error) {
	x, err := g.LNf.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, g.Head)
}

// hiddenFromEmbedCapture is hiddenFromEmbed with an optional residual-stream tap
// (the GPT sibling of Llama.hiddenFromEmbedTaps). The tap is a pure observer: a
// nil capture is byte-identical to the historical hiddenFromEmbed.
func (g *GPT) hiddenFromEmbedCapture(ctx *backend.Context, x *tensor.Tensor, capture ResidualCapture) (*tensor.Tensor, error) {
	if err := validateLayerBackends(g.LayerBackends, len(g.Blocks)); err != nil {
		return nil, err
	}
	if capture != nil {
		capture(0, x) // h_0: the post-embedding input to block 0
	}
	for l := range g.Blocks {
		var err error
		if x, err = g.forwardBlock(ctx, x, l); err != nil {
			return nil, err
		}
		if capture != nil {
			capture(l+1, x) // h_{l+1}: the residual stream after block l
		}
	}
	var err error
	if x, err = g.LNf.Forward(ctx, x); err != nil {
		return nil, err
	}
	return x, nil
}

// forwardBlock applies one GPT block through its configured layer backend. Keeping
// this step shared makes the normal full stack and bounded early-exit execution use
// exactly the same operator order.
func (g *GPT) forwardBlock(ctx *backend.Context, x *tensor.Tensor, layer int) (*tensor.Tensor, error) {
	b := g.Blocks[layer]
	layerCtx := contextForLayer(ctx, g.LayerBackends, layer)

	// attention sublayer: x += Attn(LN1(x))
	h, err := b.LN1.Forward(layerCtx, x)
	if err != nil {
		return nil, err
	}
	if h, err = b.Attn.Forward(layerCtx, h); err != nil {
		return nil, err
	}
	if x, err = exec2(layerCtx, backend.OpAdd, nil, x, h); err != nil {
		return nil, err
	}

	// FFN sublayer: x += W2·gelu(W1·LN2(x)+b1)+b2
	if h, err = b.LN2.Forward(layerCtx, x); err != nil {
		return nil, err
	}
	if h, err = exec1(layerCtx, backend.OpMatMul, nil, h, b.W1); err != nil {
		return nil, err
	}
	if h, err = exec1(layerCtx, backend.OpAddBias, nil, h, b.B1); err != nil {
		return nil, err
	}
	if h, err = exec1a(layerCtx, backend.OpGELU, nil, h); err != nil {
		return nil, err
	}
	if h, err = exec1(layerCtx, backend.OpMatMul, nil, h, b.W2); err != nil {
		return nil, err
	}
	if h, err = exec1(layerCtx, backend.OpAddBias, nil, h, b.B2); err != nil {
		return nil, err
	}
	return exec2(layerCtx, backend.OpAdd, nil, x, h)
}
