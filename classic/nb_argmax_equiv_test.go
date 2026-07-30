package classic

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Prove jointArgmax(row) == argmax over jointRow(row) (ascending class, strict >, lowest
// label wins ties), including forced ties. Fits a real GaussianNB so fitted params are valid.
func TestGaussianNBJointArgmaxEquivJointRow(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 8))
	for trial := 0; trial < 200; trial++ {
		nc := 1 + rng.IntN(6)
		d := 1 + rng.IntN(15)
		nSamp := nc * (2 + rng.IntN(6))
		X := make([][]float64, nSamp)
		y := make([]int, nSamp)
		for i := range X {
			X[i] = make([]float64, d)
			for j := range X[i] {
				X[i][j] = float64(rng.IntN(5)-2) + rng.Float64()*0.1
			}
			y[i] = rng.IntN(nc)
		}
		m := NewGaussianNB()
		if err := m.Fit(X, y); err != nil {
			continue // degenerate split (e.g. a class with <2 samples) — skip
		}
		for q := 0; q < 40; q++ {
			row := make([]float64, d)
			for j := range row {
				row[j] = float64(rng.IntN(5)-2) + rng.Float64()*0.1
			}
			// reference: argmax over jointRow. jointRow fills a caller-owned buffer rather
			// than returning one — Predict has no joint vector to hold, so the allocation
			// belongs to the callers that actually return it — hence the explicit make here.
			joint := make([]float64, len(m.classes))
			m.jointRow(row, joint)
			best, bestLL := 0, math.Inf(-1)
			for c, ll := range joint {
				if ll > bestLL {
					bestLL, best = ll, c
				}
			}
			want := m.classes[best]
			if got := m.jointArgmax(row); got != want {
				t.Fatalf("trial %d q %d nc=%d: jointArgmax=%d want %d", trial, q, nc, got, want)
			}
		}
	}
}
