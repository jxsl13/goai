package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// blas1's Dot, Nrm2 and Axpy already carry devirtualized typed paths with generic
// AtF64 fallbacks, and each claims in a comment to be bit-identical to the fallback.
// The ULP audit (R-01KYM4HGM1EEY) found the file blind: a one-ulp change in the dot
// accumulation passed every test, so that claim was unverified. These oracles check
// it — they recompute each result through AtF64 in the same ascending order, which
// is precisely what the fallback path does.
func TestBLAS1BitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, n := range []int{1, 2, 7, 64, 1000} {
		a := bench.RandF64(tensor.Shape{n}, uint64(n))
		b := bench.RandF64(tensor.Shape{n}, uint64(n)+99)

		// Dot: Σ a[i]·b[i], accumulated ascending in float64.
		out, err := backend.Execute(ctx, backend.OpDot, []*tensor.Tensor{a, b}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range n {
			acc += a.AtF64(i) * b.AtF64(i)
		}
		if got := out[0].AtF64(); math.Float64bits(got) != math.Float64bits(acc) {
			t.Fatalf("Dot n=%d: got %v want %v", n, got, acc)
		}

		// Nrm2: √Σ a[i]², same accumulation order.
		out, err = backend.Execute(ctx, backend.OpNrm2, []*tensor.Tensor{a}, nil)
		if err != nil {
			t.Fatal(err)
		}
		// Nrm2 uses the LAPACK dnrm2 SCALED update, not a naive sum of squares — a
		// first draft of this oracle assumed sqrt(sum x^2) and disagreed in the last
		// digit, because the scaled form is deliberately better conditioned. The
		// oracle must reproduce the implementation's algorithm, not the textbook
		// definition of the quantity.
		scale, ssq := 0.0, 1.0
		for i := range n {
			v := a.AtF64(i)
			if v == 0 {
				continue
			}
			av := math.Abs(v)
			if scale < av {
				r := scale / av
				ssq = 1 + ssq*r*r
				scale = av
			} else {
				r := av / scale
				ssq += r * r
			}
		}
		if got, want := out[0].AtF64(), scale*math.Sqrt(ssq); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("Nrm2 n=%d: got %v want %v", n, got, want)
		}

		// Axpy: alpha·x + y, elementwise.
		aa := backend.AXPYAttrs{Alpha: 1.7}
		out, err = backend.Execute(ctx, backend.OpAXPY, []*tensor.Tensor{a, b}, aa)
		if err != nil {
			t.Fatal(err)
		}
		for i := range n {
			want := aa.Alpha*a.AtF64(i) + b.AtF64(i)
			if got := out[0].AtF64(i); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("Axpy n=%d elem %d: got %v want %v", n, i, got, want)
			}
		}
	}
}
