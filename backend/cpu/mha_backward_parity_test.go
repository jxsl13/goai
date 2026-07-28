package cpu_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// TestCPUMHABackwardBitIdenticalToRefF64 guards the production BACKWARD attention
// path in F64. The audit (R-01KYM6C5AWFKZ) called this the most consequential gap:
// a gradient defect degrades training silently rather than producing a visibly
// wrong forward output, and a one-ulp mutation in cpu's backward accumulation
// passed the whole backend/cpu suite.
//
// F64 ONLY, and the restriction is measured rather than cautious. cpu and ref agree
// bit-for-bit on all three gradients in F64 (0 of 48 each), but DISAGREE in F32 —
// 12, 10 and 20 of 48 elements for dQ, dK and dV respectively. That divergence is
// recorded separately; whatever its cause, parity cannot guard the F32 backward
// while it stands, and pretending otherwise would produce a test that fails for
// reasons unrelated to the defect it is meant to catch.
func TestCPUMHABackwardBitIdenticalToRefF64(t *testing.T) {
	rb, ok := backend.Get(backend.Ref)
	if !ok {
		t.Skip("ref backend unavailable")
	}
	cb, ok := backend.Get(backend.CPU)
	if !ok {
		t.Skip("cpu backend unavailable")
	}
	cases := []struct{ seq, heads, dk int }{
		{1, 1, 1}, {4, 1, 3}, {6, 2, 4}, {7, 3, 2}, {16, 4, 8},
	}
	for _, c := range cases {
		for _, causal := range []bool{false, true} {
			sh := tensor.Shape{c.seq, c.heads * c.dk}
			seed := uint64(c.seq*10 + c.dk)
			in := []*tensor.Tensor{
				bench.RandF64(sh, seed), bench.RandF64(sh, seed+1),
				bench.RandF64(sh, seed+2), bench.RandF64(sh, seed+3),
			}
			at := backend.AttnAttrs{Heads: c.heads, Causal: causal}
			want, err := backend.Execute(backend.NewContext().WithBackend(rb), backend.OpMHABackward, in, at)
			if err != nil {
				t.Fatal(err)
			}
			got, err := backend.Execute(backend.NewContext().WithBackend(cb), backend.OpMHABackward, in, at)
			if err != nil {
				t.Fatal(err)
			}
			if len(got) != len(want) {
				t.Fatalf("seq=%d heads=%d dk=%d: %d outputs != %d", c.seq, c.heads, c.dk, len(got), len(want))
			}
			for o := range want {
				for i := range want[o].Numel() {
					co := tensor.Unravel(i, want[o].Shape())
					g, w := got[o].AtF64(co...), want[o].AtF64(co...)
					if math.Float64bits(g) != math.Float64bits(w) {
						t.Fatalf("seq=%d heads=%d dk=%d causal=%v grad[%d] elem %d: cpu %v != ref %v",
							c.seq, c.heads, c.dk, causal, o, i, g, w)
					}
				}
			}
		}
	}
}
