package nn_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// End-to-end char-LM proofs for the newly-added blocks (NGPTBlock, FoXBlock,
// HymbaBlock, GatedAttention, MixtureOfRecursions) and the APOLLO optimizer,
// mirroring the incumbent Mamba/RetNet e2e tests (mamba_lm_e2e_test.go,
// retnet_lm_e2e_test.go): a tiny char-level LM on the deterministic grammar
// corpus must at least HALVE its cross-entropy from the uniform-init start,
// and greedy generation must emit valid tokens deterministically. This bar
// catches gradient-scale and integration bugs that local gradcheck is blind
// to. The corpus, architecture scaffold (one-hot embedding matmul → per-layer
// stack → head), optimizer defaults (AdamW 3e-3, wd 0.01, grad-clip 1.0) and
// assertions are the incumbents', factored into one shared harness.

// lmLayer is the hidden→hidden contract every block under test satisfies:
// Forward maps [T,dim]→[T,dim] through the backend context, Params feeds the
// optimizer.
type lmLayer interface {
	Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error)
	Params() []*tensor.Tensor
}

// charLMCfg tunes the shared harness per block. Zero values mean the
// incumbent defaults (dim 48, 2 layers, window 64, 250 steps, AdamW 3e-3 with
// wd 0.01 and grad-clip 1.0, pre-norm residual wrapping with a final norm).
type charLMCfg struct {
	dim, layers, window, steps int

	// plain applies each layer directly (h = layer(h)) with NO external
	// RMSNorm, residual, or final norm — for blocks that carry their own
	// normalization and residual structure (nGPT's hypersphere update, the
	// test-local full transformer block). Default false = the incumbents'
	// RMSNorm → layer → residual wrapping plus a final RMSNorm.
	plain bool

	// makeOpt builds the optimizer over all trainable params; nil defaults to
	// the incumbents' AdamW(3e-3, 0.01). Any nn.Optimizer works — this is the
	// seam that lets APOLLO drive a real-model run.
	makeOpt func(params []*tensor.Tensor) nn.Optimizer

	// postStep runs after every optimizer step (nGPT's mandatory
	// NormalizeWeights hook); nil = nothing.
	postStep func()
}

// grammarCorpus rebuilds the incumbents' deterministic grammar corpus:
// "SUBJ VERB OBJ. " sentences over a 15-char vocabulary, tokenized by first
// appearance. Identical seed and construction to mamba_lm_e2e_test.go.
func grammarCorpus() (toks []int, vocab int) {
	rng := rand.New(rand.NewPCG(2, 0x3a3))
	subjects := []string{"the cat", "a dog"}
	verbs := []string{" sees ", " fears "}
	objects := []string{"a star", "the sun"}
	var text []byte
	for range 400 {
		text = append(text, subjects[rng.IntN(2)]...)
		text = append(text, verbs[rng.IntN(2)]...)
		text = append(text, objects[rng.IntN(2)]...)
		text = append(text, '.', ' ')
	}
	seen := map[byte]int{}
	for _, b := range text {
		if _, ok := seen[b]; !ok {
			seen[b] = len(seen)
		}
	}
	toks = make([]int, len(text))
	for i, b := range text {
		toks[i] = seen[b]
	}
	return toks, len(seen)
}

// trainCharLM builds the incumbent LM scaffold around makeLayer's blocks,
// trains it on the grammar corpus, asserts the cross-entropy at least halves
// from the first step, and finishes with the greedy-generation sanity check.
// makeLayer is called once per layer with a deterministic per-layer seed.
func trainCharLM(t *testing.T, makeLayer func(dim int, seed uint64) (lmLayer, error), cfg charLMCfg) {
	t.Helper()
	if cfg.dim == 0 {
		cfg.dim = 48
	}
	if cfg.layers == 0 {
		cfg.layers = 2
	}
	if cfg.window == 0 {
		cfg.window = 64
	}
	if cfg.steps == 0 {
		cfg.steps = 250
	}
	toks, vocab := grammarCorpus()

	emb := tensor.New(tensor.F64, tensor.Shape{vocab, cfg.dim})
	nn.XavierUniform(emb, vocab, cfg.dim, 1)
	head := nn.NewLinear(tensor.F64, cfg.dim, vocab, 2)
	params := []*tensor.Tensor{emb}
	var layers []lmLayer
	var norms []*nn.RMSNorm
	for l := range cfg.layers {
		layer, err := makeLayer(cfg.dim, uint64(10+l))
		if err != nil {
			t.Fatal(err)
		}
		layers = append(layers, layer)
		params = append(params, layer.Params()...)
		if !cfg.plain {
			nrm := nn.NewRMSNorm(tensor.F64, cfg.dim)
			norms = append(norms, nrm)
			params = append(params, nrm.Params()...)
		}
	}
	var finalNorm *nn.RMSNorm
	if !cfg.plain {
		finalNorm = nn.NewRMSNorm(tensor.F64, cfg.dim)
		params = append(params, finalNorm.Params()...)
	}
	params = append(params, head.Params()...)

	onehot := func(seq []int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{len(seq), vocab})
		for i, tk := range seq {
			x.SetF64(1, i, tk)
		}
		return x
	}
	forward := func(ctx *backend.Context, seq []int) (*tensor.Tensor, error) {
		x, err := backend.Execute(ctx, backend.OpMatMul, []*tensor.Tensor{onehot(seq), emb}, nil)
		if err != nil {
			return nil, err
		}
		h := x[0]
		for l := range cfg.layers {
			if cfg.plain {
				if h, err = layers[l].Forward(ctx, h); err != nil {
					return nil, err
				}
				continue
			}
			n, err := norms[l].Forward(ctx, h)
			if err != nil {
				return nil, err
			}
			lo, err := layers[l].Forward(ctx, n)
			if err != nil {
				return nil, err
			}
			sum, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{h, lo}, nil)
			if err != nil {
				return nil, err
			}
			h = sum[0]
		}
		if finalNorm != nil {
			if h, err = finalNorm.Forward(ctx, h); err != nil {
				return nil, err
			}
		}
		return head.Forward(ctx, h)
	}

	var opt nn.Optimizer
	if cfg.makeOpt != nil {
		opt = cfg.makeOpt(params)
	} else {
		opt = nn.NewAdamW(params, 3e-3, 0.01)
	}
	targets := tensor.New(tensor.F64, tensor.Shape{cfg.window})
	var first, last float64
	for step := range cfg.steps {
		start := (step * 89) % (len(toks) - cfg.window - 1)
		seq := toks[start : start+cfg.window]
		for i := range cfg.window {
			targets.SetF64(float64(toks[start+1+i]), i)
		}
		tape := autograd.NewTape()
		ctx := tape.Context()
		logits, err := forward(ctx, seq)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(params, tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
		if cfg.postStep != nil {
			cfg.postStep()
		}
	}
	t.Logf("char-LM: CE %.3f → %.3f (uniform = %.3f)", first, last, math.Log(float64(vocab)))
	if last >= first*0.5 {
		t.Fatalf("LM did not train: CE %.3f → %.3f (bar %.3f)", first, last, first*0.5)
	}

	// Greedy generation: deterministic, valid ids.
	gen := append([]int(nil), toks[:8]...)
	for range 24 {
		logits, err := forward(backend.NewContext(), gen)
		if err != nil {
			t.Fatal(err)
		}
		lastRow := logits.Shape()[0] - 1
		best := 0
		for v := 1; v < vocab; v++ {
			if logits.AtF64(lastRow, v) > logits.AtF64(lastRow, best) {
				best = v
			}
		}
		if best < 0 || best >= vocab {
			t.Fatalf("invalid token %d", best)
		}
		gen = append(gen, best)
	}
	if len(gen) != 32 {
		t.Fatalf("generation length %d", len(gen))
	}
}

// TestNGPTCharLM proves NGPTBlock trains a real char-LM. nGPT carries its own
// hypersphere normalization and LERP residual, so the block runs plain (no
// external RMSNorm/residual/final-norm), and — per the nGPT contract — every
// optimizer step is followed by NormalizeWeights (the weight-column
// re-projection the paper requires; GoAI optimizers have no post-step hook).
// Weight decay is disabled: decaying unit-norm columns that are immediately
// re-normalized is a no-op on the weights and harmful on the learnable scales.
func TestNGPTCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains an nGPT LM; skipped in -short")
	}
	var blocks []*nn.NGPTBlock
	trainCharLM(t, func(dim int, seed uint64) (lmLayer, error) {
		b, err := nn.NewNGPTBlock(tensor.F64, dim, 4, 2*dim, seed)
		if err != nil {
			return nil, err
		}
		blocks = append(blocks, b)
		return b, nil
	}, charLMCfg{
		plain: true,
		makeOpt: func(params []*tensor.Tensor) nn.Optimizer {
			return nn.NewAdamW(params, 1e-2, 0)
		},
		postStep: func() {
			for _, b := range blocks {
				b.NormalizeWeights()
			}
		},
	})
}

// TestFoXCharLM proves FoXBlock (Forgetting Transformer attention) trains a
// real char-LM in the incumbents' pre-norm residual scaffold. FoX has no
// positional encoding by design — its learned forget-gate decay must supply
// the recency bias the grammar needs.
func TestFoXCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a FoX LM; skipped in -short")
	}
	trainCharLM(t, func(dim int, seed uint64) (lmLayer, error) {
		return nn.NewFoXBlock(tensor.F64, dim, 4, seed)
	}, charLMCfg{})
}

// TestHymbaCharLM proves HymbaBlock (parallel attention + SSM hybrid heads
// with β-norm fusion) trains a real char-LM in the incumbents' pre-norm
// residual scaffold.
func TestHymbaCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a Hymba LM; skipped in -short")
	}
	trainCharLM(t, func(dim int, seed uint64) (lmLayer, error) {
		return nn.NewHymbaBlock(tensor.F64, dim, 4, 8, seed)
	}, charLMCfg{})
}

// gatedAttnBlock is a full pre-norm transformer block built around
// GatedAttention (which is attention-only): x + GA(norm1(x)), then the usual
// SiLU MLP sub-layer y + W2·SiLU(W1·norm2(y)). Used both as the
// TestGatedAttentionCharLM layer (plain mode: it has its own norms and
// residuals) and as evidence that the gate trains inside a real block.
type gatedAttnBlock struct {
	n1, n2 *nn.RMSNorm
	att    *nn.GatedAttention
	w1, w2 *nn.Linear
}

func newGatedAttnBlock(dim, heads int, seed uint64) (*gatedAttnBlock, error) {
	att, err := nn.NewGatedAttention(tensor.F64, dim, heads, seed)
	if err != nil {
		return nil, err
	}
	return &gatedAttnBlock{
		n1:  nn.NewRMSNorm(tensor.F64, dim),
		n2:  nn.NewRMSNorm(tensor.F64, dim),
		att: att,
		w1:  nn.NewLinear(tensor.F64, dim, 2*dim, seed+10),
		w2:  nn.NewLinear(tensor.F64, 2*dim, dim, seed+11),
	}, nil
}

func (b *gatedAttnBlock) Forward(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	ex := func(op backend.Op, in ...*tensor.Tensor) (*tensor.Tensor, error) {
		o, err := backend.Execute(ctx, op, in, nil)
		if err != nil {
			return nil, err
		}
		return o[0], nil
	}
	n, err := b.n1.Forward(ctx, x)
	if err != nil {
		return nil, err
	}
	a, err := b.att.Forward(ctx, n)
	if err != nil {
		return nil, err
	}
	h, err := ex(backend.OpAdd, x, a)
	if err != nil {
		return nil, err
	}
	n2, err := b.n2.Forward(ctx, h)
	if err != nil {
		return nil, err
	}
	m, err := b.w1.Forward(ctx, n2)
	if err != nil {
		return nil, err
	}
	if m, err = ex(backend.OpSiLU, m); err != nil {
		return nil, err
	}
	if m, err = b.w2.Forward(ctx, m); err != nil {
		return nil, err
	}
	return ex(backend.OpAdd, h, m)
}

func (b *gatedAttnBlock) Params() []*tensor.Tensor {
	p := append(b.n1.Params(), b.n2.Params()...)
	p = append(p, b.att.Params()...)
	p = append(p, b.w1.Params()...)
	return append(p, b.w2.Params()...)
}

// TestGatedAttentionCharLM proves GatedAttention trains a real char-LM inside
// a full transformer block — AND is the APOLLO real-model proof: the
// optimizer is nn.NewAPOLLO (random-projection channel-wise gradient scaling,
// low-rank Adam state), which must clear the exact same CE-halving bar AdamW
// clears for the incumbents.
func TestGatedAttentionCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a gated-attention LM with APOLLO; skipped in -short")
	}
	trainCharLM(t, func(dim int, seed uint64) (lmLayer, error) {
		return newGatedAttnBlock(dim, 4, seed)
	}, charLMCfg{
		plain: true,
		makeOpt: func(params []*tensor.Tensor) nn.Optimizer {
			return nn.NewAPOLLO(params, 3e-3, 7)
		},
	})
}

// TestMoRCharLM proves MixtureOfRecursions trains a real char-LM: each layer
// is a MoR wrapping one weight-shared MLP RecursionBlock (the same
// Linear→tanh→Linear shape mor_test.go uses), applied up to 2 recursion steps
// with a (1.0, 0.5) capacity schedule — every token takes at least one pass,
// the router picks which half recurse twice. Runs in the incumbents' pre-norm
// residual scaffold.
func TestMoRCharLM(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a Mixture-of-Recursions LM; skipped in -short")
	}
	trainCharLM(t, func(dim int, seed uint64) (lmLayer, error) {
		block := newMorMLP(tensor.F64, dim, 2*dim, seed+100)
		return nn.NewMixtureOfRecursions(block, 2, 0.5, tensor.F64, dim, seed,
			nn.WithMoRCapacities(1.0, 0.5))
	}, charLMCfg{})
}
