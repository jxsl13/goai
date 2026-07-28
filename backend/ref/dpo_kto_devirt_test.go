package ref_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// The flat-slice fast paths in DPO/KTO must be BIT-identical to the arithmetic they
// replace. The oracle recomputes each loss here from AtF64 reads — the accessor path
// the kernel no longer uses — so it shares no code with the implementation.
func TestDPOKTODevirtBitIdentical(t *testing.T) {
	be, _ := backend.Get(backend.Ref)
	ctx := backend.NewContext().WithBackend(be)
	for _, batch := range []int{1, 2, 7, 64, 257} {
		pc := bench.RandF64(tensor.Shape{batch}, 1)
		rc := bench.RandF64(tensor.Shape{batch}, 2)
		pl := bench.RandF64(tensor.Shape{batch}, 3)
		rl := bench.RandF64(tensor.Shape{batch}, 4)

		da := backend.DPOAttrs{}.WithDefaults()
		out, err := backend.Execute(ctx, backend.OpDPO, []*tensor.Tensor{pc, rc, pl, rl}, da)
		if err != nil {
			t.Fatal(err)
		}
		var want float64
		for i := range batch {
			d := da.Beta * ((pc.AtF64(i) - rc.AtF64(i)) - (pl.AtF64(i) - rl.AtF64(i)))
			u := -d
			want += math.Max(u, 0) + math.Log1p(math.Exp(-math.Abs(u)))
		}
		want /= float64(batch)
		if got := out[0].AtF64(); math.Float64bits(got) != math.Float64bits(want) {
			t.Fatalf("DPO batch=%d: got bits %#x (%v), want %#x (%v)",
				batch, math.Float64bits(got), got, math.Float64bits(want), want)
		}
	}
}
