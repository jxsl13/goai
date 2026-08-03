package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMoEBalanceExpReuseIsBitIdentical locks the one-exp-per-logit softmax to the two-pass form it
// replaced. The reference below is that older form written out in full — exp computed once for the
// sum and AGAIN for the normalize — and the comparison is on raw bits, not a tolerance, because the
// claim is that storing a value and reloading it is exact rather than merely close.
//
// Shapes are chosen so the parallel path fires and so one case has a token count that does not
// divide evenly across workers; the fold after the softmax is untouched, so any divergence here
// could only come from the softmax.
func TestMoEBalanceExpReuseIsBitIdentical(t *testing.T) {
	const alpha = 0.013
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, dims := range [][2]int{{2048, 32}, {1021, 17}} {
			tks, n := dims[0], dims[1]
			x := tensor.New(dt, tensor.Shape{tks, n})
			for i := range tks * n {
				x.SetF64(math.Sin(float64(i)*0.017)*3, i/n, i%n)
			}
			as := tensor.New(tensor.F64, tensor.Shape{tks})
			for tt := range tks {
				as.SetF64(float64((tt*7)%n), tt)
			}
			out, err := backend.Execute(backend.NewContext(), backend.OpMoEBalance,
				[]*tensor.Tensor{x, as}, backend.MoEBalanceAttrs{Alpha: alpha})
			if err != nil {
				t.Fatalf("%v %dx%d: %v", dt, tks, n, err)
			}

			// The pre-change arithmetic, verbatim.
			P := make([]float64, n)
			f := make([]float64, n)
			for tt := range tks {
				m := math.Inf(-1)
				for i := range n {
					if v := x.AtF64(tt, i); v > m {
						m = v
					}
				}
				var sum float64
				for i := range n {
					sum += math.Exp(x.AtF64(tt, i) - m)
				}
				for i := range n {
					P[i] += math.Exp(x.AtF64(tt, i)-m) / sum
				}
				f[int(as.AtF64(tt))]++
			}
			var want float64
			for i := range n {
				want += (f[i] / float64(tks)) * (P[i] / float64(tks))
			}
			want *= alpha * float64(n)
			if dt == tensor.F32 {
				want = float64(float32(want))
			}

			if got := out[0].AtF64(); math.Float64bits(got) != math.Float64bits(want) {
				t.Fatalf("%v %dx%d: one-exp %v, two-exp %v — not bit-identical",
					dt, tks, n, got, want)
			}
		}
	}
}
