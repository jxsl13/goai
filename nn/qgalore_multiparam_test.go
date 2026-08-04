package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// buildQGaLoreMultiFixture makes nP independent [dim,dim] matrix params with deterministic
// weights and gradients (fresh tensors each call so two optimizers can't alias state).
func buildQGaLoreMultiFixture(nP, dim int) ([]*tensor.Tensor, nn.GradFn) {
	params := make([]*tensor.Tensor, nP)
	grads := map[*tensor.Tensor]*tensor.Tensor{}
	for i := range params {
		params[i] = tensor.New(tensor.F64, tensor.Shape{dim, dim})
		wd := params[i].Storage().F64()
		for j := range wd {
			wd[j] = float64((j*7+i*13)%101) * 1e-3
		}
		g := tensor.New(tensor.F64, tensor.Shape{dim, dim})
		gd := g.Storage().F64()
		for j := range gd {
			gd[j] = float64((j*3+i*5)%97) * 1e-3
		}
		grads[params[i]] = g
	}
	return params, func(p *tensor.Tensor) *tensor.Tensor { return grads[p] }
}

// TestQGaLoreMultiParamDeterministic guards the parallel two-phase Step: stepping a MULTI-parameter
// QGaLore in full paper mode (seeded weight stochastic rounding) must be reproducible run-to-run.
// Because the per-parameter compute fans out across goroutines while the rng-consuming weight
// quantization is applied serially in parameter order, any regression that let the parallel phase
// touch the shared rng — or reordered the draws — would surface here as a mismatch between two
// identically-seeded runs. Single-parameter tests can't exercise the multi-parameter draw order.
func TestQGaLoreMultiParamDeterministic(t *testing.T) {
	const nP, dim, steps = 5, 96, 40
	run := func() []float64 {
		params, gf := buildQGaLoreMultiFixture(nP, dim)
		opt := nn.NewQGaLore(params, 1e-3, nn.WithQGaLoreGap(1), nn.WithQGaLoreSeed(7))
		for s := 0; s < steps; s++ {
			if err := opt.Step(gf); err != nil {
				t.Fatal(err)
			}
		}
		var out []float64
		for _, p := range params {
			out = append(out, p.Storage().F64()...)
		}
		return out
	}
	a, b := run(), run()
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("QGaLore multi-param Step not reproducible at %d: %v != %v", i, a[i], b[i])
		}
	}
}
