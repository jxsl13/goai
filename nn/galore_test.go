package nn_test

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 convergence: on a convex quadratic ½‖W−W*‖² with a full-rank
// projection (r = min dim → no information lost), GaLore's project→Adam→project-back
// drives the loss far down, guarding the whole projected-update path end to end.
func TestGaLoreConvergesQuadratic(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	rows, cols := 5, 7
	star := make([]float64, rows*cols)
	init := make([]float64, rows*cols)
	for i := range star {
		star[i] = rng.NormFloat64()
		init[i] = rng.NormFloat64()
	}
	w := tensor.FromFloat64(tensor.Shape{rows, cols}, append([]float64(nil), init...))
	// full-rank projection (r = min(rows,cols)) so the quadratic is fully covered.
	opt := nn.NewGaLore([]*tensor.Tensor{w}, 0.1, nn.WithGaLoreRank(rows), nn.WithGaLoreScale(1.0))

	loss := func() float64 {
		var s float64
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - star[i]
			s += d * d
		}
		return 0.5 * s
	}
	l0 := loss()
	for range 2000 {
		g := tensor.New(tensor.F32, w.Shape())
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			g.SetF64(w.AtF64(idx...)-star[i], idx...)
		}
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
	}
	if lf := loss(); lf > 0.1*l0 {
		t.Errorf("GaLore did not converge: loss %.4f → %.4f (want < 10%% of initial)", l0, lf)
	}
}

// GaLore keeps the Adam moment state in a rank-r subspace, so a large weight trains
// with a fraction of the optimizer memory: for a 4096×4096 layer at rank 128 the
// second-moment state shrinks from 4096·4096 to 128·4096 — 32× smaller — while the
// weights stay full-rank (Zhao et al. 2024).
func ExampleGaLore() {
	w := tensor.New(tensor.F32, tensor.Shape{4096, 4096})
	_ = nn.NewGaLore([]*tensor.Tensor{w}, 1e-3, nn.WithGaLoreRank(128))
	fmt.Println(4096*4096/(128*4096), "x smaller")
	// Output: 32 x smaller
}
