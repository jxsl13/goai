package nn

import (
	"math"
	"testing"
)

// §V16.2 (partition of unity): the order-k B-spline basis functions sum to 1 at every x in the
// grid span [gridMin, gridMax], including the endpoints (the top endpoint is closed so it holds
// there too). Checked at the paper defaults (G=5, k=3) and a couple of other (G,k).
func TestKANPartitionOfUnity(t *testing.T) {
	cases := []struct {
		G, k   int
		lo, hi float64
	}{
		{5, 3, -1, 1}, // paper defaults
		{4, 2, -2, 3},
		{7, 1, 0, 1},
		{3, 4, -1, 2},
	}
	for _, c := range cases {
		knots := buildKnots(c.lo, c.hi, c.G, c.k)
		scratch := make([]float64, len(knots))
		// sample the closed span including both endpoints
		const n = 40
		for s := 0; s <= n; s++ {
			x := c.lo + (c.hi-c.lo)*float64(s)/float64(n)
			vals := evalBSplineBasis(x, knots, c.k, scratch)
			if got := len(vals); got != c.G+c.k {
				t.Fatalf("G=%d k=%d: nbasis=%d, want %d", c.G, c.k, got, c.G+c.k)
			}
			var sum float64
			for _, v := range vals {
				sum += v
			}
			if math.Abs(sum-1) > 1e-10 {
				t.Errorf("G=%d k=%d x=%.4f: Σ basis = %.15g, want 1", c.G, c.k, x, sum)
			}
		}
	}
}

// §V16.2 (hand-checked spline): with k=1 (piecewise-linear hat functions), G=2 over [0,1], the knot
// vector is [-0.5,0,0.5,1,1.5] and at x=0.25 the three basis values are exactly [0.5,0.5,0]. A spline
// with coefficients [2,4,6] therefore evaluates to 0.5·2+0.5·4+0·6 = 3, all hand-computable.
func TestKANHandSpline(t *testing.T) {
	knots := buildKnots(0, 1, 2, 1)
	wantKnots := []float64{-0.5, 0, 0.5, 1, 1.5}
	for i, w := range wantKnots {
		if math.Abs(knots[i]-w) > 1e-12 {
			t.Fatalf("knots[%d]=%v, want %v", i, knots[i], w)
		}
	}
	scratch := make([]float64, len(knots))
	vals := evalBSplineBasis(0.25, knots, 1, scratch)
	want := []float64{0.5, 0.5, 0}
	if len(vals) != 3 {
		t.Fatalf("nbasis=%d, want 3", len(vals))
	}
	for i := range want {
		if math.Abs(vals[i]-want[i]) > 1e-10 {
			t.Errorf("basis[%d]=%.15g, want %v", i, vals[i], want[i])
		}
	}
	coef := []float64{2, 4, 6}
	var spline float64
	for i := range coef {
		spline += vals[i] * coef[i]
	}
	if math.Abs(spline-3) > 1e-10 {
		t.Errorf("hand spline = %.15g, want 3", spline)
	}
}
