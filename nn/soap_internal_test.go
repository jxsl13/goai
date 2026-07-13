package nn

import (
	"math"
	"testing"
)

// §V16 tier-1: the eigenbasis rotations are exact inverses (Q_L, Q_R orthogonal), so
// rotateBack(Q_L, rotateForward(Q_L, G, Q_R), Q_R) = G.
func TestSOAPRotationInverse(t *testing.T) {
	// symmetric PSD matrices → real orthogonal eigenbases
	l := [][]float64{{4, 1, 0}, {1, 3, 1}, {0, 1, 2}}
	r := [][]float64{{2, 0.5}, {0.5, 1}}
	ql, qr := eigenBasis(l), eigenBasis(r)
	g := [][]float64{{1, -2}, {0.5, 3}, {-1, 0.25}} // 3×2

	back := rotateBack(ql, rotateForward(ql, g, qr), qr)
	for i := range 3 {
		for j := range 2 {
			if math.Abs(back[i][j]-g[i][j]) > 1e-10 {
				t.Errorf("rotate round-trip [%d,%d] = %.10g, want %.10g", i, j, back[i][j], g[i][j])
			}
		}
	}
	// Q_L columns are orthonormal: QᵀQ = I
	for a := range 3 {
		for b := range 3 {
			var d float64
			for i := range 3 {
				d += ql[i][a] * ql[i][b]
			}
			want := 0.0
			if a == b {
				want = 1
			}
			if math.Abs(d-want) > 1e-10 {
				t.Errorf("(Q_LᵀQ_L)[%d,%d] = %.10g, want %.10g", a, b, d, want)
			}
		}
	}
}
