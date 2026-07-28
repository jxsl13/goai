package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// retention carries a devirtualized flat path with a generic AtF64 fallback; the ULP
// audit (R-01KYM4HGM1EEY) found it blind. Per PROC-011 this oracle reproduces the
// kernel's ALGORITHM: the decayed inner product P_nm = (Σ_i Q[n,i]·K[m,i])·γ^(n−m)
// for m ≤ n, then O[n] = Σ_m P_nm·V[m], accumulated in that order. The fast path
// precomputes γ^t into a table and indexes pow[n−m]; math.Pow with the same argument
// yields the same bits, so the oracle computes it directly.
func TestRetentionBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][3]int{{1, 1, 1}, {3, 2, 4}, {6, 4, 4}, {9, 3, 5}} {
		l, dk, dv := sz[0], sz[1], sz[2]
		q := bench.RandF64(tensor.Shape{l, dk}, uint64(l*100+dk))
		k := bench.RandF64(tensor.Shape{l, dk}, uint64(l*100+dk)+7)
		v := bench.RandF64(tensor.Shape{l, dv}, uint64(l*100+dv)+13)
		ra := backend.RetentionAttrs{Gamma: 0.9}
		out, err := backend.Execute(ctx, backend.OpRetention, []*tensor.Tensor{q, k, v}, ra)
		if err != nil {
			t.Fatal(err)
		}
		obuf := make([]float64, dv)
		for n := range l {
			p := make([]float64, n+1)
			for m := 0; m <= n; m++ {
				var a float64
				for i := range dk {
					a += q.AtF64(n, i) * k.AtF64(m, i)
				}
				p[m] = a * math.Pow(ra.Gamma, float64(n-m))
			}
			for j := range obuf {
				obuf[j] = 0
			}
			for m := 0; m <= n; m++ {
				for j := range dv {
					obuf[j] += p[m] * v.AtF64(m, j)
				}
			}
			for j := range dv {
				if got := out[0].AtF64(n, j); math.Float64bits(got) != math.Float64bits(obuf[j]) {
					t.Fatalf("l=%d dk=%d dv=%d O[%d,%d]: got %v want %v", l, dk, dv, n, j, got, obuf[j])
				}
			}
		}
	}
}
