package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// Oracles for the GRPO/CPO/IPO devirtualization, written BEFORE it (PROC-009):
// the ULP audit found all three blind — a one-ulp perturbation of each accumulation
// passed the whole backend/ref suite. Each oracle recomputes the loss through the
// AtF64 accessor path the kernels are about to stop using, so it shares no code with
// the implementation, and compares on raw bits rather than a tolerance.
func softplusRef(u float64) float64 {
	return math.Max(u, 0) + math.Log1p(math.Exp(-math.Abs(u)))
}

func TestPreferenceLossesBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, batch := range []int{1, 2, 7, 64, 129} {
		a := bench.RandF64(tensor.Shape{batch}, 1)
		b2 := bench.RandF64(tensor.Shape{batch}, 2)
		c := bench.RandF64(tensor.Shape{batch}, 3)
		d := bench.RandF64(tensor.Shape{batch}, 4)

		// GRPO: lpNew, lpOld, adv, lpRef
		ga := backend.GRPOAttrs{}.WithDefaults()
		got, err := backend.Execute(ctx, backend.OpGRPO, []*tensor.Tensor{a, b2, c, d}, ga)
		if err != nil {
			t.Fatal(err)
		}
		var tot float64
		for i := range batch {
			r := math.Exp(a.AtF64(i) - b2.AtF64(i))
			av := d.AtF64(i)
			surr := math.Min(r*av, math.Max(1-ga.Epsilon, math.Min(1+ga.Epsilon, r))*av)
			dd := c.AtF64(i) - a.AtF64(i)
			tot += surr - ga.Beta*(math.Exp(dd)-dd-1)
		}
		checkBitsScalar(t, "GRPO", batch, got[0].AtF64(), -tot/float64(batch))

		// CPO: w, l
		ca := backend.CPOAttrs{}.WithDefaults()
		got, err = backend.Execute(ctx, backend.OpCPO, []*tensor.Tensor{a, b2}, ca)
		if err != nil {
			t.Fatal(err)
		}
		tot = 0
		for i := range batch {
			lw := a.AtF64(i)
			z := ca.Beta * (lw - b2.AtF64(i))
			tot += softplusRef(-z) + ca.Alpha*(-lw)
		}
		checkBitsScalar(t, "CPO", batch, got[0].AtF64(), tot/float64(batch))

		// IPO: pc, rc, pl, rl
		ia := backend.IPOAttrs{}.WithDefaults()
		got, err = backend.Execute(ctx, backend.OpIPO, []*tensor.Tensor{a, b2, c, d}, ia)
		if err != nil {
			t.Fatal(err)
		}
		tot = 0
		target := 1.0 / (2.0 * ia.Beta)
		for i := range batch {
			h := (a.AtF64(i) - c.AtF64(i)) - (b2.AtF64(i) - d.AtF64(i))
			dd := h - target
			tot += dd * dd
		}
		checkBitsScalar(t, "IPO", batch, got[0].AtF64(), tot/float64(batch))
	}
}

func checkBitsScalar(t *testing.T, name string, batch int, got, want float64) {
	t.Helper()
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("%s batch=%d: got bits %#x (%v), want %#x (%v)",
			name, batch, math.Float64bits(got), got, math.Float64bits(want), want)
	}
}
