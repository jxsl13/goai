package linalg

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// The flat arms must agree with the accessor arms BIT-FOR-BIT, including on a strided view
// where flatRowMajor declines and both norms must fall back. Tolerance 0: the change moves no
// arithmetic, only how each element is fetched.
func TestNormFlatArmsBitIdentical(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, sh := range [][2]int{{1, 1}, {3, 7}, {64, 32}, {33, 33}} {
		m, n := sh[0], sh[1]
		d := make([]float64, m*n)
		for i := range d {
			d[i] = rng.NormFloat64() * 3
		}
		a := tensor.FromFloat64(tensor.Shape{m, n}, d)
		if _, ok := flatRowMajor(a); !ok {
			t.Fatalf("%dx%d: expected the flat arm to be taken", m, n)
		}
		// accessor-arm references, computed here in the pre-change form
		var wantFro float64
		for i := range m {
			for j := range n {
				v := a.AtF64(i, j)
				wantFro += v * v
			}
		}
		wantFro = math.Sqrt(wantFro)
		wantInf := 0.0
		for i := range m {
			var r float64
			for j := range n {
				r += math.Abs(a.AtF64(i, j))
			}
			if r > wantInf {
				wantInf = r
			}
		}
		gotFro, err := NormFro(a)
		if err != nil {
			t.Fatal(err)
		}
		gotInf, err := NormInf(a)
		if err != nil {
			t.Fatal(err)
		}
		if math.Float64bits(gotFro) != math.Float64bits(wantFro) {
			t.Fatalf("%dx%d NormFro %016x, want %016x", m, n, math.Float64bits(gotFro), math.Float64bits(wantFro))
		}
		if math.Float64bits(gotInf) != math.Float64bits(wantInf) {
			t.Fatalf("%dx%d NormInf %016x, want %016x", m, n, math.Float64bits(gotInf), math.Float64bits(wantInf))
		}
	}
}
