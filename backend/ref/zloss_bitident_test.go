package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// zloss carries a devirtualized flat-row path with a generic AtF64 fallback and
// claims bit-identity between them; the ULP audit (R-01KYM4HGM1EEY) found it blind.
// Per PROC-011 this oracle reproduces the kernel's ALGORITHM — stable log-sum-exp
// with a per-row max, then the square of the lse, then coeff·total/b — rather than
// any textbook restatement of the z-loss.
func TestZLossBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][2]int{{1, 1}, {2, 3}, {5, 8}, {16, 4}} {
		b, c := sz[0], sz[1]
		z := bench.RandF64(tensor.Shape{b, c}, uint64(b*10+c))
		za := backend.ZLossAttrs{Coeff: 1e-3}
		out, err := backend.Execute(ctx, backend.OpZLoss, []*tensor.Tensor{z}, za)
		if err != nil {
			t.Fatal(err)
		}
		var total float64
		for i := range b {
			m := math.Inf(-1)
			for j := range c {
				if v := z.AtF64(i, j); v > m {
					m = v
				}
			}
			var sum float64
			for j := range c {
				sum += math.Exp(z.AtF64(i, j) - m)
			}
			lse := m + math.Log(sum)
			total += lse * lse
		}
		want := za.Coeff * total / float64(b)
		if got := out[0].AtF64(); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("b=%d c=%d: got bits %#x (%v), want %#x (%v)",
				b, c, math.Float64bits(got), got, math.Float64bits(want), want)
		}
	}
}
