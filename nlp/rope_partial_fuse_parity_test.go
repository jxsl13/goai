package nlp

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// partialRoPE has a fused gather/scatter arm that replaces seven layout dispatches, claimed
// bit-identical to the dispatch path. The arms are selected by ctx.Recorder — a plain context
// takes the fused arm, a taped one keeps the dispatches — so one raw-bit comparison covers the
// layout algebra and the tape guard at once.
//
// Swept over geometries rather than one shape, because the whole claim is about index
// arithmetic: a single (seq, heads, rotaryDim) triple can agree by coincidence when an offset
// is wrong, and rotaryDim == hd takes an entirely different early-return branch.
//
// Every rotaryDim here is EVEN — RoPE rotates channel pairs and the cpu kernel rejects an odd
// head dim outright. The first version of this sweep included an odd one and failed on the
// backend's own validation, which is a bad test rather than a finding.
func TestPartialRoPEFusedMatchesDispatch(t *testing.T) {
	cases := []struct{ seq, heads, hd, rotaryDim int }{
		{1, 4, 8, 4},  // decode: one row, half rotated
		{5, 4, 8, 4},  // prefill
		{3, 2, 6, 2},  // rotary smaller than half
		{2, 3, 6, 2},  // three heads, rotary a third of the head
		{6, 2, 10, 4}, // longer sequence, rotary under half
		{1, 1, 8, 6},  // single head, most of it rotated
		{4, 2, 8, 8},  // rotaryDim == hd: the early-return path
	}
	for _, tc := range cases {
		rng := rand.New(rand.NewSource(9))
		x := tensor.New(tensor.F32, tensor.Shape{tc.seq, tc.heads * tc.hd})
		xs := x.Storage().F32()
		for i := range xs {
			xs[i] = float32(rng.NormFloat64())
		}
		rope := backend.RoPEAttrs{Base: 10000, PosOffset: 2}
		run := func(ctx *backend.Context) []float32 {
			out, err := partialRoPE(ctx, x, tc.heads, tc.rotaryDim, rope)
			if err != nil {
				t.Fatal(err)
			}
			return append([]float32(nil), out.Contiguous().Storage().F32()...)
		}
		fused := run(backend.NewContext())
		dispatch := run(autograd.NewTape().Context())
		if len(fused) != len(dispatch) {
			t.Fatalf("%v: length %d vs %d", tc, len(fused), len(dispatch))
		}
		for i := range fused {
			if math.Float32bits(fused[i]) != math.Float32bits(dispatch[i]) {
				t.Fatalf("%+v: element %d differs: fused %08x, dispatch %08x — the fused layout "+
					"algebra disagrees with the slice/concat path",
					tc, i, math.Float32bits(fused[i]), math.Float32bits(dispatch[i]))
			}
		}
	}
}
