//go:build darwin && cgo

package llamagpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T483: the optimizer zoo on a REAL transformer training workload — every optimizer here was
// validated against its paper on small fixtures; this is the first time each trains an actual
// GPT (same architecture, same init seed, same data order — only the optimizer differs). Each
// must at least halve the initial cross-entropy in 120 steps; the final losses are logged as the
// comparison table. Sophia is excluded: its Hessian update needs the Gauss-Newton-Bartlett
// estimator (a second backward on resampled labels) — a faithful harness of its own.
func TestOptimizerZooTrainsGPT(t *testing.T) {
	if testing.Short() {
		t.Skip("trains eight models; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	const steps, window = 120, 128

	type stepper interface{ Step(nn.GradFn) error }
	cases := []struct {
		name string
		mk   func(ps []*tensor.Tensor) stepper
		eval func(o stepper) // optional pre-eval hook
	}{
		{"AdamW", func(ps []*tensor.Tensor) stepper { return nn.NewAdamW(ps, 3e-3, 0.01) }, nil},
		{"Adafactor", func(ps []*tensor.Tensor) stepper { return nn.NewAdafactor(ps) }, nil},
		{"LAMB", func(ps []*tensor.Tensor) stepper { return nn.NewLAMB(ps, 0.01) }, nil},
		// Muon's contract is 2-D matrices only — the paper's recipe pairs it with Adam for the
		// 1-D params (norm gains, biases); this composite is that recipe.
		{"Muon(2D)+AdamW(1D)", func(ps []*tensor.Tensor) stepper {
			var mats, vecs []*tensor.Tensor
			for _, p := range ps {
				if p.Ndim() == 2 {
					mats = append(mats, p)
				} else {
					vecs = append(vecs, p)
				}
			}
			return &compositeOpt{a: nn.NewMuon(mats, 0.02), b: nn.NewAdamW(vecs, 3e-3, 0.01)}
		}, nil},
		{"AdEMAMix", func(ps []*tensor.Tensor) stepper { return nn.NewAdEMAMix(ps, 3e-3) }, nil},
		{"ScheduleFreeAdamW", func(ps []*tensor.Tensor) stepper { return nn.NewScheduleFreeAdamW(ps, 3e-3) },
			func(o stepper) { o.(*nn.ScheduleFree).Eval() }},
		{"Shampoo", func(ps []*tensor.Tensor) stepper { return nn.NewShampoo(ps, 0.03, nn.WithShampooRootEvery(20)) }, nil},
		{"SOAP", func(ps []*tensor.Tensor) stepper { return nn.NewSOAP(ps, 3e-3) }, nil},
	}

	results := make([]string, 0, len(cases))
	for _, tc := range cases {
		m := trainableGPT(t, cfg, 1) // identical init for every optimizer
		opt := tc.mk(m.Params())
		targets := tensor.New(tensor.F32, tensor.Shape{window})
		var first, last float64
		for step := range steps {
			start := (step * 97) % (len(toks) - window - 1)
			tokens := toks[start : start+window]
			for i := range window {
				targets.SetF64(float64(toks[start+1+i]), i)
			}
			tape := autograd.NewTapeOn(be)
			c := tape.Context()
			logits, err := m.Forward(c, tokens)
			if err != nil {
				t.Fatal(err)
			}
			loss, err := nn.CrossEntropy(c, logits, targets)
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
			clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
			if err := opt.Step(clipped); err != nil {
				t.Fatalf("%s: %v", tc.name, err)
			}
		}
		if tc.eval != nil {
			tc.eval(opt)
		}
		results = append(results, tc.name+": "+formatCE(first, last))
		if last >= first*0.5 {
			t.Errorf("%s failed to halve the loss on a real GPT: CE %.3f → %.3f", tc.name, first, last)
		}
	}
	for _, r := range results {
		t.Log(r)
	}
}

func formatCE(first, last float64) string {
	return "CE " + trim(first) + " → " + trim(last)
}

func trim(f float64) string {
	s := []byte("0.000")
	v := int(f*1000 + 0.5)
	s[0] = byte('0' + v/1000)
	s[2] = byte('0' + (v/100)%10)
	s[3] = byte('0' + (v/10)%10)
	s[4] = byte('0' + v%10)
	return string(s)
}

// compositeOpt drives two optimizers over disjoint parameter sets with one Step call.
type compositeOpt struct {
	a, b interface{ Step(nn.GradFn) error }
}

func (c *compositeOpt) Step(g nn.GradFn) error {
	if err := c.a.Step(g); err != nil {
		return err
	}
	return c.b.Step(g)
}

// §T503: Sophia joins the zoo with a FAITHFUL Gauss-Newton-Bartlett Hessian harness — the
// estimator the §T483 zoo deferred: every 10 steps, labels are RESAMPLED from the model's own
// softmax, a separate backward on that sampled loss yields ∇L̂, and ĥ = B·(∇L̂⊙∇L̂) feeds
// UpdateHessian. Same harness, same halve-the-loss bar as the zoo.
func TestSophiaTrainsGPT(t *testing.T) {
	if testing.Short() {
		t.Skip("trains a model; skipped in -short")
	}
	if !metal.Available() {
		t.Skip("metal: no gpu")
	}
	be, ok := backend.Get(backend.Metal)
	if !ok {
		t.Skip("metal not registered")
	}
	corpus := specCorpus(600)
	toks, vocab := charTokens(corpus)
	cfg := nlp.GPTConfig{Vocab: vocab, Ctx: 256, Dim: 96, Heads: 4, Layers: 3, Eps: 1e-5}
	const steps, window = 120, 128

	m := trainableGPT(t, cfg, 1)
	opt := nn.NewSophia(m.Params(), 5e-4) // Sophia runs at ~10× smaller lr than AdamW
	rng := rand.New(rand.NewPCG(13, 0x50f1a))

	gnbUpdate := func(tokens []int) {
		// forward once to get the model's distribution, sample labels from it.
		probeCtx := backend.NewContext().WithBackend(be)
		logits, err := m.Forward(probeCtx, tokens)
		if err != nil {
			t.Fatal(err)
		}
		sampled := tensor.New(tensor.F32, tensor.Shape{len(tokens)})
		for i := range tokens {
			var mx float64 = math.Inf(-1)
			for v := range vocab {
				if x := logits.AtF64(i, v); x > mx {
					mx = x
				}
			}
			var sum float64
			ps := make([]float64, vocab)
			for v := range vocab {
				ps[v] = math.Exp(logits.AtF64(i, v) - mx)
				sum += ps[v]
			}
			u := rng.Float64() * sum
			pick, cum := 0, 0.0
			for v, p := range ps {
				cum += p
				if u <= cum {
					pick = v
					break
				}
			}
			sampled.SetF64(float64(pick), i)
		}
		// backward on the SAMPLED-label loss; ĥ = B·grad².
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		lg, err := m.Forward(ctx, tokens)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.CrossEntropy(ctx, lg, sampled)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		hess := map[*tensor.Tensor]*tensor.Tensor{}
		for _, p := range m.Params() {
			g := tape.Grad(p)
			if g == nil {
				continue
			}
			h := tensor.New(tensor.F64, p.Shape())
			for i := range p.Numel() {
				idx := tensor.Unravel(i, p.Shape())
				gv := g.AtF64(idx...)
				h.SetF64(float64(window)*gv*gv, idx...)
			}
			hess[p] = h
		}
		if err := opt.UpdateHessian(func(p *tensor.Tensor) *tensor.Tensor { return hess[p] }); err != nil {
			t.Fatal(err)
		}
	}

	targets := tensor.New(tensor.F32, tensor.Shape{window})
	var first, last float64
	for step := range steps {
		start := (step * 97) % (len(toks) - window - 1)
		tokens := toks[start : start+window]
		if step%10 == 0 {
			gnbUpdate(tokens)
		}
		for i := range window {
			targets.SetF64(float64(toks[start+1+i]), i)
		}
		tape := autograd.NewTapeOn(be)
		ctx := tape.Context()
		logits, err := m.Forward(ctx, tokens)
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
		clipped, _ := nn.ClipGradNorm(m.Params(), tape.Grad, 1.0)
		if err := opt.Step(clipped); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("Sophia (GNB every 10): CE %.3f → %.3f", first, last)
	if last >= first*0.5 {
		t.Errorf("Sophia failed to halve the loss: CE %.3f → %.3f", first, last)
	}
}
