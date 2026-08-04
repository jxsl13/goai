package nn

import (
	"math"
	"math/rand"
	"testing"
)

// §V16 tier-1: invMatrixRoot computes M^{−1/4} exactly for a diagonal (hence any
// symmetric-PSD) matrix — diag(16,81) ⇒ diag(16^{−1/4}, 81^{−1/4}) = diag(1/2, 1/3) —
// and the reconstruction (M^{−1/4})⁴·M = I holds.
func TestInvMatrixRoot(t *testing.T) {
	m := [][]float64{{16, 0}, {0, 81}}
	r := invMatrixRoot(m, 4, 1e-12)
	want := [][]float64{{0.5, 0}, {0, 1.0 / 3.0}}
	for i := range 2 {
		for j := range 2 {
			if math.Abs(r[i][j]-want[i][j]) > 1e-9 {
				t.Errorf("M^{-1/4}[%d,%d] = %.10g, want %.10g", i, j, r[i][j], want[i][j])
			}
		}
	}
	// (M^{-1/4})⁴ · M = I : r⁴ = M^{-1}, so r⁴·M = I.
	r2 := matMul2(r, r)
	r4 := matMul2(r2, r2)
	id := matMul2(r4, m)
	for i := range 2 {
		for j := range 2 {
			want := 0.0
			if i == j {
				want = 1
			}
			if math.Abs(id[i][j]-want) > 1e-8 {
				t.Errorf("(M^{-1/4})⁴·M [%d,%d] = %.10g, want %.10g", i, j, id[i][j], want)
			}
		}
	}
}

func matMul2(a, b [][]float64) [][]float64 {
	n := len(a)
	out := make([][]float64, n)
	for i := range n {
		out[i] = make([]float64, n)
		for j := range n {
			for k := range n {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func benchIMR(b *testing.B, n int) {
	rng := rand.New(rand.NewSource(2))
	mat := make([][]float64, n)
	for i := range n {
		mat[i] = make([]float64, n)
		for j := 0; j <= i; j++ {
			v := rng.NormFloat64()*0.1 + boolf(i == j)
			mat[i][j], mat[j][i] = v, v
		}
	}
	b.ResetTimer()
	for range b.N {
		_ = invMatrixRoot(mat, 4, 1e-12)
	}
}
func boolf(b bool) float64 {
	if b {
		return 4
	}
	return 0
}
func BenchmarkInvMatrixRoot_256(b *testing.B) { benchIMR(b, 256) }
func BenchmarkInvMatrixRoot_512(b *testing.B) { benchIMR(b, 512) }
