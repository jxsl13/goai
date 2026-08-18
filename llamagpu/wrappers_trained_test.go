//go:build darwin && cgo

package llamagpu_test

import (
	"fmt"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §T484: the composable optimizer WRAPPERS on the real GPT (the §T483 zoo's sibling): Lookahead,
// Grokfast and GaLore wrap/replace AdamW behind the same Step(GradFn) contract; CautiousAdamW is
// the masked variant; SAM exercises its real two-pass contract — the gradient closure is called
// at w AND at the perturbed w_adv, so it must re-run forward+backward against the live tensors.
// Each must halve the initial cross-entropy in 120 steps on the identical task.
func TestOptimizerWrappersTrainGPT(t *testing.T) {
	if testing.Short() {
		t.Skip("trains the optimizer wrappers; skipped in -short")
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

	// one training step's forward+backward for the CURRENT parameters; returns the grad lookup.
	gradsFor := func(m *nlp.GPT, step int) func() (nn.GradFn, error) {
		return func() (nn.GradFn, error) {
			start := (step * 97) % (len(toks) - window - 1)
			tokens := toks[start : start+window]
			targets := tensor.New(tensor.F32, tensor.Shape{window})
			for i := range window {
				targets.SetF64(float64(toks[start+1+i]), i)
			}
			tape := autograd.NewTapeOn(be)
			c := tape.Context()
			logits, err := m.Forward(c, tokens)
			if err != nil {
				return nil, err
			}
			loss, err := nn.CrossEntropy(c, logits, targets)
			if err != nil {
				return nil, err
			}
			if err := tape.Backward(loss); err != nil {
				return nil, err
			}
			lastLoss = loss.AtF64()
			return tape.Grad, nil
		}
	}

	type runner func(m *nlp.GPT) error // performs `steps` optimizer steps
	stdLoop := func(mk func(ps []*tensor.Tensor) interface{ Step(nn.GradFn) error }) runner {
		return func(m *nlp.GPT) error {
			opt := mk(m.Params())
			for step := range steps {
				g, err := gradsFor(m, step)()
				if err != nil {
					return err
				}
				clipped, _ := nn.ClipGradNorm(m.Params(), g, 1.0)
				if err := opt.Step(clipped); err != nil {
					return err
				}
				track(step)
			}
			return nil
		}
	}
	cases := []struct {
		name string
		run  runner
	}{
		{"Lookahead(AdamW)", stdLoop(func(ps []*tensor.Tensor) interface{ Step(nn.GradFn) error } {
			return nn.NewLookahead(nn.NewAdamW(ps, 3e-3, 0.01), ps)
		})},
		{"Grokfast(AdamW)", stdLoop(func(ps []*tensor.Tensor) interface{ Step(nn.GradFn) error } {
			// λ=2 amplification ≈ doubles the effective step — compensate the lr (paper retunes it).
			return nn.NewGrokfast(nn.NewAdamW(ps, 1.5e-3, 0.01), ps)
		})},
		{"CautiousAdamW", stdLoop(func(ps []*tensor.Tensor) interface{ Step(nn.GradFn) error } {
			return nn.NewCautiousAdamW(ps, 3e-3)
		})},
		{"GaLore", stdLoop(func(ps []*tensor.Tensor) interface{ Step(nn.GradFn) error } {
			return nn.NewGaLore(ps, 3e-3)
		})},
		// QuantBits=0 is the specified full-precision collapse. Feed Q-GaLore
		// and a GaLore shadow the same clipped GPT gradients, then prove every
		// update remains bit-identical. Comparing independent Metal training
		// runs would instead admit GPU reduction-order noise.
		{"Q-GaLore(QuantBits=0)", func(m *nlp.GPT) error {
			qParams := m.Params()
			shadowParams := trainableGPT(t, cfg, 1).Params()
			q := nn.NewQGaLore(qParams, 3e-3, nn.WithQGaLoreQuantBits(0))
			g := nn.NewGaLore(shadowParams, 3e-3)
			for step := range steps {
				grad, err := gradsFor(m, step)()
				if err != nil {
					return err
				}
				clipped, _ := nn.ClipGradNorm(qParams, grad, 1.0)
				qGrad := make(map[*tensor.Tensor]*tensor.Tensor, len(qParams))
				gGrad := make(map[*tensor.Tensor]*tensor.Tensor, len(qParams))
				for i, p := range qParams {
					v := clipped(p)
					qGrad[p] = v
					gGrad[shadowParams[i]] = v
				}
				if err := q.Step(func(p *tensor.Tensor) *tensor.Tensor { return qGrad[p] }); err != nil {
					return err
				}
				if err := g.Step(func(p *tensor.Tensor) *tensor.Tensor { return gGrad[p] }); err != nil {
					return err
				}
				for pi, qp := range qParams {
					gp := shadowParams[pi]
					if !qp.Shape().Equal(gp.Shape()) {
						return fmt.Errorf("step %d parameter %d shape %v != %v", step, pi, qp.Shape(), gp.Shape())
					}
					for ei := range qp.Numel() {
						idx := tensor.Unravel(ei, qp.Shape())
						if qv, gv := qp.AtF64(idx...), gp.AtF64(idx...); qv != gv {
							return fmt.Errorf("step %d parameter %d element %d: Q-GaLore %.17g != GaLore %.17g", step, pi, ei, qv, gv)
						}
					}
				}
				track(step)
			}
			return nil
		}},
		{"SAM(AdamW)", func(m *nlp.GPT) error { // SAM's own two-pass contract
			opt := nn.NewSAM(m.Params(), nn.NewAdamW(m.Params(), 3e-3, 0.01))
			for step := range steps {
				if err := opt.Step(gradsFor(m, step)); err != nil {
					return err
				}
				track(step)
			}
			return nil
		}},
	}

	for _, tc := range cases {
		m := trainableGPT(t, cfg, 1)
		firstLoss, lastTracked = 0, 0
		if err := tc.run(m); err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		t.Logf("%s: CE %.3f → %.3f", tc.name, firstLoss, lastTracked)
		if lastTracked >= firstLoss*0.5 {
			t.Errorf("%s failed to halve the loss: CE %.3f → %.3f", tc.name, firstLoss, lastTracked)
		}
	}
}

// shared loss tracking for the wrapper loops (single-goroutine test).
var lastLoss, firstLoss, lastTracked float64

func track(step int) {
	if step == 0 {
		firstLoss = lastLoss
	}
	lastTracked = lastLoss
}
