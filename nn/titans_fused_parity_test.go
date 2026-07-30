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

// The fused linear-memory inference path (ctx.Recorder==nil) must be BIT-IDENTICAL to the
// dispatch path (recorder set → op-graph loop) for the same inputs.
func TestTitansLinearFusedBitExactVsDispatch(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 29))
	randMat := func(rows, cols int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		s := x.Storage().F64()
		for i := range s {
			s[i] = rng.NormFloat64()
		}
		return x
	}
	sigCol := func(seq int) *tensor.Tensor { // gate values in (0,1)
		x := tensor.New(tensor.F64, tensor.Shape{seq, 1})
		s := x.Storage().F64()
		for i := range s {
			s[i] = 1 / (1 + math.Exp(-rng.NormFloat64()))
		}
		return x
	}

	for _, tc := range []struct{ seq, dim int }{{1, 8}, {5, 16}, {33, 24}, {96, 40}} {
		m, err := nn.NewNeuralMemory(tensor.F64, tc.dim, tc.dim, 3, nn.WithTitansLinearMemory())
		if err != nil {
			t.Fatal(err)
		}
		// give M0 nonzero (meta-learned) values so the memory-init path is exercised
		m0 := m.M0.Storage().F64()
		for i := range m0 {
			m0[i] = 0.1 * rng.NormFloat64()
		}
		q, k, v := randMat(tc.seq, tc.dim), randMat(tc.seq, tc.dim), randMat(tc.seq, tc.dim)
		eta, theta, alpha := sigCol(tc.seq), sigCol(tc.seq), sigCol(tc.seq)

		disp, err := m.Scan(autograd.NewTape().Context(), q, k, v, eta, theta, alpha)
		if err != nil {
			t.Fatal(err)
		}
		fused, err := m.Scan(backend.NewContext(), q, k, v, eta, theta, alpha)
		if err != nil {
			t.Fatal(err)
		}
		if disp.Numel() != fused.Numel() {
			t.Fatalf("seq=%d dim=%d: numel mismatch", tc.seq, tc.dim)
		}
		for i := range disp.Numel() {
			c := tensor.Unravel(i, disp.Shape())
			a, b := disp.AtF64(c...), fused.AtF64(c...)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Fatalf("seq=%d dim=%d idx=%d: dispatch %v != fused %v (not bit-identical)", tc.seq, tc.dim, i, a, b)
			}
		}
	}
}
