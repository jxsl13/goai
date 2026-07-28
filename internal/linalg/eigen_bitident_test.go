package linalg

import (
	"math"
	"math/rand"
	"testing"
)

// symEigRows is the ORIGINAL row-of-slices Jacobi implementation, kept here as the
// oracle for the flat rewrite. Every operation, operand and ordering matches; only
// the buffer layout differs, so the two must agree BIT for bit. If this ever needs
// updating alongside SymEig, the rewrite it guards has changed semantics and that is
// the bug.
func symEigRows(a [][]float64) ([]float64, [][]float64) {
	n := len(a)
	m := make([][]float64, n)
	v := make([][]float64, n)
	for i := range n {
		m[i] = append([]float64(nil), a[i]...)
		v[i] = make([]float64, n)
		v[i][i] = 1
	}
	for range 100 {
		off := 0.0
		for i := range n {
			for j := i + 1; j < n; j++ {
				off += m[i][j] * m[i][j]
			}
		}
		if off < 1e-28 {
			break
		}
		for p := range n {
			for q := p + 1; q < n; q++ {
				if math.Abs(m[p][q]) < 1e-300 {
					continue
				}
				theta := (m[q][q] - m[p][p]) / (2 * m[p][q])
				t := math.Copysign(1, theta) / (math.Abs(theta) + math.Sqrt(theta*theta+1))
				c := 1 / math.Sqrt(t*t+1)
				s := t * c
				for k := range n {
					mkp, mkq := m[k][p], m[k][q]
					m[k][p] = c*mkp - s*mkq
					m[k][q] = s*mkp + c*mkq
				}
				for k := range n {
					mpk, mqk := m[p][k], m[q][k]
					m[p][k] = c*mpk - s*mqk
					m[q][k] = s*mpk + c*mqk
				}
				for k := range n {
					vkp, vkq := v[k][p], v[k][q]
					v[k][p] = c*vkp - s*vkq
					v[k][q] = s*vkp + c*vkq
				}
			}
		}
	}
	vals := make([]float64, n)
	for i := range n {
		vals[i] = m[i][i]
	}
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	for i := range n {
		best := i
		for j := i + 1; j < n; j++ {
			if vals[order[j]] > vals[order[best]] {
				best = j
			}
		}
		order[i], order[best] = order[best], order[i]
	}
	sorted := make([]float64, n)
	vecs := make([][]float64, n)
	for k, oi := range order {
		sorted[k] = vals[oi]
		col := make([]float64, n)
		for r := range n {
			col[r] = v[r][oi]
		}
		vecs[k] = col
	}
	return sorted, vecs
}

func TestSymEigFlatBitIdenticalToRows(t *testing.T) {
	for _, n := range []int{1, 2, 3, 8, 17} {
		rng := rand.New(rand.NewSource(int64(n)))
		a := make([][]float64, n)
		for i := range n {
			a[i] = make([]float64, n)
		}
		for i := range n {
			for j := 0; j <= i; j++ {
				v := rng.NormFloat64()
				a[i][j], a[j][i] = v, v
			}
		}
		gotVals, gotVecs := SymEig(a)
		wantVals, wantVecs := symEigRows(a)
		for k := range n {
			if math.Float64bits(gotVals[k]) != math.Float64bits(wantVals[k]) {
				t.Fatalf("n=%d val[%d]: got %v want %v", n, k, gotVals[k], wantVals[k])
			}
			for r := range n {
				if math.Float64bits(gotVecs[k][r]) != math.Float64bits(wantVecs[k][r]) {
					t.Fatalf("n=%d vec[%d][%d]: got %v want %v", n, k, r, gotVecs[k][r], wantVecs[k][r])
				}
			}
		}
	}
}
