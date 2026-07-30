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

// TestKANFusedBitExactVsDispatch pins the inference spline path (ctx.Recorder == nil,
// KANLayer.fusedSpline) bit-identical to the two-einsum dispatch path (a recording tape
// context): the fused contraction reproduces the einsums' i-outer/c-inner summation order
// exactly, so the two outputs must agree to the last bit.
func TestKANFusedBitExactVsDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 4))
	for _, tc := range []struct{ batch, in, out int }{{8, 12, 10}, {16, 20, 24}, {5, 7, 3}} {
		l, err := nn.NewKAN(tc.in, tc.out, 3)
		if err != nil {
			t.Fatal(err)
		}
		x := randMat(rng, tc.batch, tc.in)
		fused, err := l.Forward(backend.NewContext(), x) // Recorder == nil → fusedSpline
		if err != nil {
			t.Fatal(err)
		}
		disp, err := l.Forward(autograd.NewTape().Context(), x) // Recorder != nil → two einsums
		if err != nil {
			t.Fatal(err)
		}
		var maxd float64
		for idx := range fused.Numel() {
			co := tensor.Unravel(idx, fused.Shape())
			if d := math.Abs(fused.AtF64(co...) - disp.AtF64(co...)); d > maxd {
				maxd = d
			}
		}
		if maxd != 0 {
			t.Fatalf("batch=%d in=%d out=%d: fused vs einsum differ by %g (want bit-exact 0)", tc.batch, tc.in, tc.out, maxd)
		}
	}
}
