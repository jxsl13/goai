package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/internal/linalg"
)

// TestGaLoreGramBitIdentical pins the triangle + k-outer rewrite of both Gram matrices
// against the original (i,j)-outer full-matrix formulation, on raw float64 bits.
//
// Both rewrites are order-preserving for a reason worth stating: the triangle mirror is
// valid because g[i]·g[j] and g[j]·g[i] sum the same products over the same ascending k
// (multiplication commutes, the sum is not reassociated), and the k-outer form interchanges
// the loops without touching the order in which any single entry accumulates.
func TestGaLoreGramBitIdentical(t *testing.T) {
	for _, dims := range [][2]int{{4, 7}, {7, 4}, {1, 5}, {5, 1}, {6, 6}, {13, 9}, {9, 13}} {
		m, n := dims[0], dims[1]
		g := make([][]float64, m)
		for i := range g {
			g[i] = make([]float64, n)
			for j := range g[i] {
				g[i][j] = math.Sin(float64(i*n+j)*0.71) + 0.3*math.Cos(float64(i+j)*1.7)
			}
		}
		// reference: the original full-matrix loops
		wantGG := make([][]float64, m)
		for i := range m {
			wantGG[i] = make([]float64, m)
			for j := range m {
				var s float64
				for k := range n {
					s += g[i][k] * g[j][k]
				}
				wantGG[i][j] = s
			}
		}
		wantGTG := make([][]float64, n)
		for i := range n {
			wantGTG[i] = make([]float64, n)
			for j := range n {
				var s float64
				for k := range m {
					s += g[k][i] * g[k][j]
				}
				wantGTG[i][j] = s
			}
		}
		// the shipped path: galoreProjection picks GGᵀ when m<=n, else GᵀG. Drive both by
		// transposing, so each branch is exercised for every shape.
		gotVecs, left := galoreProjection(g, min(m, n))
		if left != (m <= n) {
			t.Fatalf("m=%d n=%d: side %v", m, n, left)
		}
		var ref [][]float64
		if left {
			ref = wantGG
		} else {
			ref = wantGTG
		}
		_, wantVecs := linalg.SymEig(ref)
		if len(gotVecs) > len(wantVecs) {
			t.Fatalf("m=%d n=%d: got %d vectors, reference has %d", m, n, len(gotVecs), len(wantVecs))
		}
		for i := range gotVecs {
			for j := range gotVecs[i] {
				if math.Float64bits(gotVecs[i][j]) != math.Float64bits(wantVecs[i][j]) {
					t.Fatalf("m=%d n=%d vec[%d][%d]: got %v want %v (not bit-identical)",
						m, n, i, j, gotVecs[i][j], wantVecs[i][j])
				}
			}
		}
	}
}
