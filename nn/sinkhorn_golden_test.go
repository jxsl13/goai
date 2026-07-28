package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// sinkhornRef is the unblocked scaling iteration, transcribed to pin the SUMMATION ORDER
// the shipped implementation must keep. Register-blocking the output index is bit-safe
// only because each accumulator still sums its reduction axis in ascending order; a probe
// that unrolls the reduction axis instead reorders the adds and this gate goes red.
func sinkhornRef(cost *tensor.Tensor, r, c []float64, eps float64, iters int) *tensor.Tensor {
	m, n := cost.Shape()[0], cost.Shape()[1]
	minC := math.Inf(1)
	for i := range m {
		for j := range n {
			if v := cost.AtF64(i, j); v < minC {
				minC = v
			}
		}
	}
	k := make([][]float64, m)
	for i := range m {
		k[i] = make([]float64, n)
		for j := range n {
			k[i][j] = math.Exp(-(cost.AtF64(i, j) - minC) / eps)
		}
	}
	u, v := make([]float64, m), make([]float64, n)
	for i := range u {
		u[i] = 1
	}
	for j := range v {
		v[j] = 1
	}
	for range iters {
		for i := range m {
			var kv float64
			for j := range n {
				kv += k[i][j] * v[j]
			}
			if kv > 0 {
				u[i] = r[i] / kv
			}
		}
		for j := range n {
			var ktu float64
			for i := range m {
				ktu += k[i][j] * u[i]
			}
			if ktu > 0 {
				v[j] = c[j] / ktu
			}
		}
	}
	p := tensor.New(cost.Dtype(), tensor.Shape{m, n})
	for i := range m {
		for j := range n {
			p.SetF64(u[i]*k[i][j]*v[j], i, j)
		}
	}
	return p
}

// TestSinkhornBitIdenticalToUnblocked covers sizes that exercise every remainder class of
// a 4-way unroll on BOTH axes — an unroll whose tail is wrong stays invisible at sizes
// divisible by 4, which is exactly the size a hand-written check tends to pick.
func TestSinkhornBitIdenticalToUnblocked(t *testing.T) {
	for _, sz := range [][2]int{{4, 4}, {5, 7}, {6, 6}, {7, 5}, {13, 11}, {16, 16}, {17, 3}, {3, 17}, {1, 1}, {2, 9}} {
		m, n := sz[0], sz[1]
		cost := tensor.New(tensor.F64, tensor.Shape{m, n})
		for i := range m {
			for j := range n {
				cost.SetF64(0.5+math.Abs(math.Sin(float64(i*n+j)*1.3)), i, j)
			}
		}
		r, c := make([]float64, m), make([]float64, n)
		for i := range r {
			r[i] = 1 / float64(m)
		}
		for j := range c {
			c[j] = 1 / float64(n)
		}
		got, err := nn.Sinkhorn(cost, r, c, 0.4, 25)
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		want := sinkhornRef(cost, r, c, 0.4, 25)
		for i := range got.Numel() {
			idx := tensor.Unravel(i, got.Shape())
			g, w := got.AtF64(idx...), want.AtF64(idx...)
			if math.Float64bits(g) != math.Float64bits(w) {
				t.Fatalf("%dx%d at %v: got %g want %g (not bit-identical)", m, n, idx, g, w)
			}
		}
	}
}
