package ref_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// The flat working buffer must produce a BIT-identical factor to the row-slice
// layout it replaced. The oracle below recomputes the factorization in the original
// [][]float64 form with the same operation order, so any divergence is a layout or
// indexing bug rather than a rounding difference — there is no rounding difference
// available, the arithmetic is unchanged.
func TestCholeskyFlatBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{1, 2, 3, 9, 32} {
		rng := rand.New(rand.NewSource(int64(n)))
		d := make([]float64, n*n)
		for i := range n {
			for j := 0; j <= i; j++ {
				v := rng.NormFloat64() * 0.1
				d[i*n+j], d[j*n+i] = v, v
			}
			d[i*n+i] += float64(n) + 1
		}
		a := tensor.FromFloat64(tensor.Shape{n, n}, d)
		out, err := backend.Execute(ctx, backend.OpCholesky, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		// Reference: original row-of-slices layout, identical operation order.
		l := make([][]float64, n)
		for i := range n {
			l[i] = make([]float64, n)
		}
		for j := range n {
			dg := d[j*n+j]
			for k := range j {
				dg -= l[j][k] * l[j][k]
			}
			ljj := math.Sqrt(dg)
			l[j][j] = ljj
			for i := j + 1; i < n; i++ {
				s := d[i*n+j]
				for k := range j {
					s -= l[i][k] * l[j][k]
				}
				l[i][j] = s / ljj
			}
		}
		for i := range n {
			for j := 0; j <= i; j++ {
				got, want := out[0].AtF64(i, j), l[i][j]
				if math.Float64bits(got) != math.Float64bits(want) {
					t.Fatalf("n=%d L[%d,%d]: got bits %#x (%v), want %#x (%v)",
						n, i, j, math.Float64bits(got), got, math.Float64bits(want), want)
				}
			}
		}
	}
}
