package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Distill already carries a devirtualized flat path with hoisted p/q buffers, but
// the ULP audit (R-01KYM4HGM1EEY) found it blind: a one-ulp change in the KL
// accumulation passed every test. This pins the SHIPPED fast path against an
// independent recomputation through AtF64 — the accessor route the fast path
// bypasses — on raw bits. It guards an optimization that already exists rather than
// one being introduced.
func softmaxRowOracle(z *tensor.Tensor, i, c int, temp float64) []float64 {
	m := math.Inf(-1)
	for j := range c {
		if v := z.AtF64(i, j) / temp; v > m {
			m = v
		}
	}
	out := make([]float64, c)
	var sum float64
	for j := range c {
		e := math.Exp(z.AtF64(i, j)/temp - m)
		out[j] = e
		sum += e
	}
	for j := range c {
		out[j] /= sum
	}
	return out
}

func TestDistillBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, sz := range [][2]int{{1, 2}, {3, 5}, {8, 16}, {17, 3}} {
		b, c := sz[0], sz[1]
		zs := bench.RandF64(tensor.Shape{b, c}, uint64(b*10+c))
		zt := bench.RandF64(tensor.Shape{b, c}, uint64(b*10+c+7))
		da := backend.DistillAttrs{}.WithDefaults()
		out, err := backend.Execute(ctx, backend.OpDistill, []*tensor.Tensor{zs, zt}, da)
		if err != nil {
			t.Fatal(err)
		}
		temp := da.Temperature
		var total float64
		for i := range b {
			p := softmaxRowOracle(zt, i, c, temp)
			q := softmaxRowOracle(zs, i, c, temp)
			var kl float64
			for j := range c {
				if p[j] > 0 {
					kl += p[j] * (math.Log(p[j]) - math.Log(q[j]))
				}
			}
			total += temp * temp * kl
		}
		want := total / float64(b)
		if got := out[0].AtF64(); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("b=%d c=%d: got bits %#x (%v), want %#x (%v)",
				b, c, math.Float64bits(got), got, math.Float64bits(want), want)
		}
	}
}
