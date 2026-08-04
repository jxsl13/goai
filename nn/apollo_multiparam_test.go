package nn_test

import (
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func buildAPOLLOMultiFixture(nP, dim int) ([]*tensor.Tensor, nn.GradFn) {
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

func runAPOLLOMulti(t *testing.T, nP, dim, steps int) []float64 {
	params, gf := buildAPOLLOMultiFixture(nP, dim)
	opt := nn.NewAPOLLO(params, 1e-3, 9, nn.WithAPOLLOGap(3))
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

// TestAPOLLOMultiParamDeterministic guards the parallel two-phase Step: a MULTI-parameter APOLLO
// (seeded projections, reseeds crossing a Gap boundary) must be reproducible run-to-run. The
// per-parameter compute fans out across goroutines while every a.rng draw stays serial in Phase A,
// so any regression that let the parallel phase touch the shared rng, or reordered the reseed draws,
// would surface as a mismatch. Single-parameter tests can't exercise the multi-parameter draw order.
func TestAPOLLOMultiParamDeterministic(t *testing.T) {
	a := runAPOLLOMulti(t, 5, 96, 40)
	b := runAPOLLOMulti(t, 5, 96, 40)
	if len(a) != len(b) {
		t.Fatalf("length mismatch %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("APOLLO multi-param Step not reproducible at %d: %v != %v", i, a[i], b[i])
		}
	}
}
