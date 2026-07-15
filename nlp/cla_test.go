package nlp_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// claAsGPT assembles a plain GPT that SHARES the CLA's live weight tensors —
// only valid for Share=1, where every block carries its own Wk/Wv. This is the
// strongest form of the degeneracy check: same tensors, two forward
// implementations.
func claAsGPT(t *testing.T, c *nlp.CLA) *nlp.GPT {
	t.Helper()
	cfg := c.Config
	if cfg.Share != 1 {
		t.Fatalf("claAsGPT needs Share=1, got %d", cfg.Share)
	}
	g := &nlp.GPT{
		Config: nlp.GPTConfig{Vocab: cfg.Vocab, Ctx: cfg.Ctx, Dim: cfg.Dim,
			Heads: cfg.Heads, Layers: cfg.Layers, Eps: cfg.Eps},
		TokEmb: c.TokEmb, PosEmb: c.PosEmb, LNf: c.LNf, Head: c.Head,
	}
	for _, b := range c.Blocks {
		attn, err := nlp.NewMHA(cfg.Heads, b.Wq, b.Wk, b.Wv, b.Wo)
		if err != nil {
			t.Fatal(err)
		}
		attn.Causal = true
		g.Blocks = append(g.Blocks, &nlp.Block{
			LN1: b.LN1, LN2: b.LN2, Attn: attn,
			W1: b.W1, B1: b.B1, W2: b.W2, B2: b.B2,
		})
	}
	return g
}

// The key equivalence: with Share=1 every layer is its own leader, and CLA is
// EXACTLY the plain GPT — a GPT built from the same weight tensors produces
// identical logits (within 1e-10, same f64 op sequence).
func TestCLAShare1MatchesGPT(t *testing.T) {
	cfg := nlp.CLAConfig{Vocab: 11, Ctx: 8, Dim: 8, Heads: 2, Layers: 2, Share: 1, Eps: 1e-5}
	c, err := nlp.NewCLA(cfg, 42)
	if err != nil {
		t.Fatal(err)
	}
	g := claAsGPT(t, c)
	tokens := []int{3, 1, 4, 1, 5, 9, 2, 6}

	ctx := backend.NewContext()
	got, err := c.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	want, err := g.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	if !got.Shape().Equal(want.Shape()) {
		t.Fatalf("shape %v vs GPT %v", got.Shape(), want.Shape())
	}
	for i := range tokens {
		for v := range cfg.Vocab {
			a, b := got.AtF64(i, v), want.AtF64(i, v)
			if math.Abs(a-b) > 1e-10*math.Max(1, math.Abs(b)) {
				t.Fatalf("logits[%d,%d]: CLA %.14g vs GPT %.14g", i, v, a, b)
			}
		}
	}
	t.Logf("Share=1 CLA matches plain GPT on all %d×%d logits", len(tokens), cfg.Vocab)
}

// Numeric gradcheck through EVERY parameter on a tiny Share=2 config: central
// differences on a next-token CE loss vs the tape's analytic gradients (the
// repo's gradcheck recipe, rtol 1e-4 at h=1e-6 in F64). Layers=2, Share=2 makes
// layer 0 the single leader whose k,v feed BOTH layers' OpMHA calls — this
// specifically exercises the fan-out gradient summation into Wk/Wv.
func TestCLAGradcheck(t *testing.T) {
	cfg := nlp.CLAConfig{Vocab: 3, Ctx: 4, Dim: 4, Heads: 2, Layers: 2, Share: 2, Eps: 1e-5}
	m, err := nlp.NewCLA(cfg, 11)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{0, 1, 2, 1}
	targets := tensor.FromFloat64(tensor.Shape{4}, []float64{1, 2, 1, 0})

	loss := func(ctx *backend.Context) *tensor.Tensor {
		logits, err := m.Forward(ctx, tokens)
		if err != nil {
			t.Fatal(err)
		}
		l, err := nn.CrossEntropy(ctx, logits, targets)
		if err != nil {
			t.Fatal(err)
		}
		return l
	}

	tape := autograd.NewTape()
	if err := tape.Backward(loss(tape.Context())); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for pi, p := range m.Params() {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("param %d: nil gradient", pi)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := loss(backend.NewContext()).AtF64()
			p.SetF64(o-h, coord...)
			lm := loss(backend.NewContext()).AtF64()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Fatalf("param %d grad%v: numeric %.8g vs analytic %.8g", pi, coord, num, ana)
			}
		}
	}
	t.Logf("gradcheck passed for all %d params (fan-out into shared Wk/Wv included)", len(m.Params()))
}

// The headline invariant: the decode cache holds Layers/Share K/V slots, not
// one per layer — and follower blocks really carry no Wk/Wv.
func TestCLACacheSize(t *testing.T) {
	for _, tc := range []struct{ layers, share int }{{4, 1}, {4, 2}, {4, 4}, {6, 3}} {
		cfg := nlp.CLAConfig{Vocab: 5, Ctx: 4, Dim: 4, Heads: 2, Layers: tc.layers, Share: tc.share, Eps: 1e-5}
		m, err := nlp.NewCLA(cfg, 1)
		if err != nil {
			t.Fatal(err)
		}
		cache := m.NewCache()
		want := tc.layers / tc.share
		if len(cache.K) != want || len(cache.V) != want {
			t.Fatalf("layers=%d share=%d: cache slots K=%d V=%d, want %d",
				tc.layers, tc.share, len(cache.K), len(cache.V), want)
		}
		for l, b := range m.Blocks {
			leader := l%tc.share == 0
			if (b.Wk != nil) != leader || (b.Wv != nil) != leader {
				t.Fatalf("layers=%d share=%d: block %d Wk/Wv presence wrong (leader=%v)",
					tc.layers, tc.share, l, leader)
			}
		}
	}
}

// NewCLA rejects bad geometry, including the CLA-specific Share constraints.
func TestCLAConfigValidation(t *testing.T) {
	ok := nlp.CLAConfig{Vocab: 5, Ctx: 4, Dim: 4, Heads: 2, Layers: 4, Share: 2}
	for name, mut := range map[string]func(*nlp.CLAConfig){
		"zero vocab":      func(c *nlp.CLAConfig) { c.Vocab = 0 },
		"zero ctx":        func(c *nlp.CLAConfig) { c.Ctx = 0 },
		"zero dim":        func(c *nlp.CLAConfig) { c.Dim = 0 },
		"zero layers":     func(c *nlp.CLAConfig) { c.Layers = 0 },
		"dim%heads":       func(c *nlp.CLAConfig) { c.Heads = 3 },
		"zero share":      func(c *nlp.CLAConfig) { c.Share = 0 },
		"layers%share":    func(c *nlp.CLAConfig) { c.Share = 3 },
		"share too large": func(c *nlp.CLAConfig) { c.Share = 8 },
		"negative heads":  func(c *nlp.CLAConfig) { c.Heads = -2 },
	} {
		cfg := ok
		mut(&cfg)
		if _, err := nlp.NewCLA(cfg, 1); err == nil {
			t.Errorf("%s: want error, got nil", name)
		}
	}
	if _, err := nlp.NewCLA(ok, 1); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

// §T35-style parity for the group-shared cache: incremental DecodeStep must
// reproduce the full-forward logits at every position (same causal attention
// math, f64) — including across group boundaries (Layers=4, Share=2 → two
// groups, followers reading the leader's just-updated slot).
func TestCLADecodeMatchesForward(t *testing.T) {
	cfg := nlp.CLAConfig{Vocab: 11, Ctx: 12, Dim: 8, Heads: 2, Layers: 4, Share: 2, Eps: 1e-5}
	m, err := nlp.NewCLA(cfg, 7)
	if err != nil {
		t.Fatal(err)
	}
	tokens := []int{3, 1, 4, 1, 5, 9, 2, 6, 5, 3}
	ctx := backend.NewContext()

	full, err := m.Forward(ctx, tokens)
	if err != nil {
		t.Fatal(err)
	}
	cache := m.NewCache()
	for pos, tok := range tokens {
		step, err := m.DecodeStep(ctx, cache, tok, pos)
		if err != nil {
			t.Fatal(err)
		}
		if got := cache.K[0].Shape()[0]; got != pos+1 {
			t.Fatalf("cache rows %d after step %d", got, pos)
		}
		for v := range cfg.Vocab {
			got, want := step.AtF64(0, v), full.AtF64(pos, v)
			if math.Abs(got-want) > 1e-11*math.Max(1, math.Abs(want)) {
				t.Fatalf("pos %d cls %d: incremental %.14g vs full %.14g", pos, v, got, want)
			}
		}
	}
	t.Logf("CLA decode matches full forward across %d positions (%d KV slots for %d layers)",
		len(tokens), len(cache.K), cfg.Layers)
}

// End-to-end: a Share=2 CLA trains on the in-repo grammar corpus (the char-LM
// e2e bar, nn/newblocks_lm_e2e_test.go): next-token cross-entropy on a fixed
// eval window must at least HALVE from the random-init start.
func TestCLAGrammarE2E(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a CLA model; skipped in -short")
	}
	toks, vocab, _ := diffusionGrammar()
	const window = 64
	cfg := nlp.CLAConfig{Vocab: vocab, Ctx: window, Dim: 48, Heads: 4, Layers: 2, Share: 2, Eps: 1e-5}
	m, err := nlp.NewCLA(cfg, 1)
	if err != nil {
		t.Fatal(err)
	}

	// fixed eval snapshot: window at offset 0, next-token targets
	evalIn := toks[:window]
	evalTgt := tensor.New(tensor.F64, tensor.Shape{window})
	for i := range window {
		evalTgt.SetF64(float64(toks[i+1]), i)
	}
	evalLoss := func() float64 {
		logits, err := m.Forward(backend.NewContext(), evalIn)
		if err != nil {
			t.Fatal(err)
		}
		l, err := nn.CrossEntropy(backend.NewContext(), logits, evalTgt)
		if err != nil {
			t.Fatal(err)
		}
		return l.AtF64()
	}
	first := evalLoss()

	opt := nn.NewAdamW(m.Params(), 3e-3, 0.01)
	for step := range 300 {
		start := (step * 89) % (len(toks) - window - 1)
		seq := toks[start : start+window]
		tgt := tensor.New(tensor.F64, tensor.Shape{window})
		for i := range window {
			tgt.SetF64(float64(toks[start+i+1]), i)
		}
		tape := autograd.NewTape()
		logits, err := m.Forward(tape.Context(), seq)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(tape.Context(), logits, tgt)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	last := evalLoss()
	t.Logf("CLA (Share=2) eval CE %.3f → %.3f", first, last)
	if last >= first*0.5 {
		t.Fatalf("CLA did not train: eval CE %.3f → %.3f (bar %.3f)", first, last, first*0.5)
	}
}

// ExampleCLA builds a small Cross-Layer Attention model (4 layers sharing k,v
// in pairs), forwards a prompt, and decodes one token through the group-shared
// KV-cache — which holds Layers/Share = 2 slots, the memory win CLA is for.
// Everything is seeded, so the output is deterministic.
func ExampleCLA() {
	cfg := nlp.CLAConfig{Vocab: 7, Ctx: 8, Dim: 8, Heads: 2, Layers: 4, Share: 2, Eps: 1e-5}
	m, _ := nlp.NewCLA(cfg, 1)
	fmt.Println("params:", len(m.Params())) // followers carry no Wk/Wv

	logits, _ := m.Forward(backend.NewContext(), []int{0, 1, 2})
	fmt.Println("logits:", logits.Shape())

	cache := m.NewCache()
	fmt.Println("kv slots:", len(cache.K), "for", cfg.Layers, "layers")
	step, _ := m.DecodeStep(backend.NewContext(), cache, 3, 0)
	fmt.Println("step:", step.Shape(), "cached tokens:", cache.K[0].Shape()[0])
	// Output:
	// params: 49
	// logits: (3, 7)
	// kv slots: 2 for 4 layers
	// step: (1, 7) cached tokens: 1
}
