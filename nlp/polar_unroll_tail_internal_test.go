package nlp

import (
	"math"
	"testing"
)

// The unrolled applyInverse must agree with the plain form at EVERY d, including sizes where the
// 4-wide body leaves a remainder — d=1,2,3 exercise only the tail, and 6,7 exercise both.
func TestPolarApplyInverseUnrollTail(t *testing.T) {
	for _, d := range []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 17} {
		p, err := newPolarRotation(d, 3)
		if err != nil {
			t.Fatal(err)
		}
		y := make([]float64, d)
		for i := range y {
			y[i] = float64(i%7) + 0.25
		}
		got, err := p.applyInverse(y)
		if err != nil {
			t.Fatal(err)
		}
		// Reference: the plain i-outer form, same order, no unroll.
		want := make([]float64, d)
		for i := range d {
			for j := range d {
				want[j] += p.q[i][j] * y[i]
			}
		}
		for j := range want {
			if math.Float64bits(got[j]) != math.Float64bits(want[j]) {
				t.Fatalf("d=%d out[%d]: unrolled %v (%016x) != plain %v (%016x)",
					d, j, got[j], math.Float64bits(got[j]), want[j], math.Float64bits(want[j]))
			}
		}
	}
}
