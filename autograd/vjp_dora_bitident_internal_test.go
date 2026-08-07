package autograd

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestDoRAWeightBackwardBitIdentical proves the PS1006 row-major restructure is BIT-identical to the
// original column-walk algorithm (gradcheck only bounds finite-difference error; this pins the exact
// bits). It recomputes ∂V/∂m with the ORIGINAL nested-loop order and asserts every f64 output word
// matches the registered VJP's.
func TestDoRAWeightBackwardBitIdentical(t *testing.T) {
	vjp := vjps[backend.OpDoRAWeight]
	rng := rand.New(rand.NewSource(99))
	for _, d := range []struct{ rows, cols int }{{5, 4}, {64, 48}, {129, 31}} {
		rows, cols := d.rows, d.cols
		v := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		g := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		m := tensor.New(tensor.F64, tensor.Shape{cols})
		vs, gs, ms := v.Storage().F64(), g.Storage().F64(), m.Storage().F64()
		for i := range vs {
			vs[i] = rng.NormFloat64()
		}
		for i := range gs {
			gs[i] = rng.NormFloat64()
		}
		for i := range ms {
			ms[i] = rng.NormFloat64()
		}
		// Force a degenerate zero column to exercise the n==0 branch.
		if cols > 2 {
			for i := 0; i < rows; i++ {
				vs[i*cols+2] = 0
			}
		}
		out, err := vjp(nil, []*tensor.Tensor{v, m}, nil, nil, g)
		if err != nil {
			t.Fatal(err)
		}
		gotDV, gotDM := out[0].Storage().F64(), out[1].Storage().F64()

		// ORIGINAL algorithm (column-walk, fused per column).
		refDV := make([]float64, rows*cols)
		refDM := make([]float64, cols)
		for j := 0; j < cols; j++ {
			var ss, s float64
			for i := 0; i < rows; i++ {
				x := vs[i*cols+j]
				ss += x * x
				s += gs[i*cols+j] * x
			}
			n := math.Sqrt(ss)
			if n == 0 {
				continue
			}
			mj := ms[j]
			refDM[j] = s / n
			for i := 0; i < rows; i++ {
				refDV[i*cols+j] = mj / n * (gs[i*cols+j] - vs[i*cols+j]*s/(n*n))
			}
		}
		for i := range refDV {
			if math.Float64bits(gotDV[i]) != math.Float64bits(refDV[i]) {
				t.Fatalf("[%dx%d] dV[%d]: got %v want %v (not bit-identical)", rows, cols, i, gotDV[i], refDV[i])
			}
		}
		for j := range refDM {
			if math.Float64bits(gotDM[j]) != math.Float64bits(refDM[j]) {
				t.Fatalf("[%dx%d] dM[%d]: got %v want %v (not bit-identical)", rows, cols, j, gotDM[j], refDM[j])
			}
		}
	}
}
