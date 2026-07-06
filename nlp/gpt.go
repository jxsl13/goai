package nlp

import (
	"fmt"

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
	Config GPTConfig
	TokEmb *tensor.Tensor // [vocab, dim]
	PosEmb *tensor.Tensor // [ctx, dim]
	Blocks []*Block
	LNf    *nn.LayerNorm
	Head   *tensor.Tensor // [dim, vocab]
}

// GPTConfig fixes the model geometry.
type GPTConfig struct {
	Vocab, Ctx, Dim, Heads, Layers int
	Eps                            float64
}

// Block is one pre-LN transformer block.
type Block struct {
	LN1, LN2 *nn.LayerNorm
	Attn     *MHA
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
	return exec1(ctx, backend.OpAdd, nil, et, ep)
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

// Forward computes logits [seq, vocab] for the prompt tokens.
func (g *GPT) Forward(ctx *backend.Context, tokens []int) (*tensor.Tensor, error) {
	x, err := g.Embed(ctx, tokens)
	if err != nil {
		return nil, err
	}
	for _, b := range g.Blocks {
		// attention sublayer: x += Attn(LN1(x))
		h, err := b.LN1.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		if h, err = b.Attn.Forward(ctx, h); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
		// FFN sublayer: x += W2·gelu(W1·LN2(x)+b1)+b2
		if h, err = b.LN2.Forward(ctx, x); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B1); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpGELU, nil, h); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpMatMul, nil, h, b.W2); err != nil {
			return nil, err
		}
		if h, err = exec1(ctx, backend.OpAddBias, nil, h, b.B2); err != nil {
			return nil, err
		}
		if x, err = exec1(ctx, backend.OpAdd, nil, x, h); err != nil {
			return nil, err
		}
	}
	if x, err = g.LNf.Forward(ctx, x); err != nil {
		return nil, err
	}
	return exec1(ctx, backend.OpMatMul, nil, x, g.Head)
}
