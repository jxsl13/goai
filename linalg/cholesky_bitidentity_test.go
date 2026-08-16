package linalg_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// TestCholeskyIsBitIdentical freezes every bit of the factor and of a multi-column solve. Jamming
// four rows through one pass of the pivot row claims to change no value — each row keeps dot4's
// own four accumulators over the same ascending k and the same combination — and a factorization
// that drifted in the last bits would still pass a residual check, so the digest is the only gate
// that would notice.
//
// The sizes straddle the jam: 3 runs entirely in the by-one tail, 7 takes one jammed pass plus a
// remainder of 3, and 129 is an odd size well past it.
func TestCholeskyIsBitIdentical(t *testing.T) {
	cases := []struct {
		n, k int
		f32  bool
		want uint64
	}{
		{3, 2, false, archgold.Pick(17768641141587771019, 11980659079766952600)},
		{7, 3, false, archgold.Pick(6519991207317368223, 17211136390019119064)},
		{64, 5, false, archgold.Pick(977471996829047676, 6074487709568889395)},
		{129, 4, false, archgold.Pick(7848359424020427753, 5307133137334945473)},
		// F32 input takes the OTHER arm of the factorization — the one that reads through
		// AtF64 — which the F64 rows never enter.
		{7, 3, true, archgold.Pick(14982212773978087913, 6458871279157759121)},
		{64, 5, true, archgold.Pick(4047594521655782583, 12445304480102178508)},
	}
	for _, c := range cases {
		dt := tensor.F64
		if c.f32 {
			dt = tensor.F32
		}
		a := tensor.New(dt, tensor.Shape{c.n, c.n})
		as := make([]float64, c.n*c.n)
		for i := range c.n {
			for j := range c.n {
				v := math.Sin(float64(i*31+j*17)) * 0.5
				as[i*c.n+j] = v
				as[j*c.n+i] = v
			}
			as[i*c.n+i] = float64(c.n) + 1 // diagonally dominant: SPD, no pivot degeneracy
		}
		for i, v := range as {
			a.SetF64(v, i/c.n, i%c.n)
		}
		b := tensor.New(dt, tensor.Shape{c.n, c.k})
		for i := range c.n * c.k {
			b.SetF64(math.Cos(float64(i*13+5)), i/c.k, i%c.k)
		}
		l, err := linalg.Cholesky(a)
		if err != nil {
			t.Fatal(err)
		}
		x, err := linalg.CholSolve(a, b)
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, tn := range []*tensor.Tensor{l, x} {
			cn := tn.Contiguous()
			r, c2 := cn.Shape()[0], cn.Shape()[1]
			for i := range r * c2 {
				u := math.Float64bits(cn.AtF64(i/c2, i%c2))
				for s := 0; s < 64; s += 8 {
					h = (h ^ (u>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("n=%d k=%d f32=%v digest %d, want %d", c.n, c.k, c.f32, h, c.want)
		}
	}
}
