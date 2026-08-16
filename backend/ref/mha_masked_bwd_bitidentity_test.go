package ref

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestMHAMaskedBackwardIsBitIdentical freezes all four gradients. Jamming the dV and dQ/dK loops
// four keys per pass claims to change no value, and NOTHING ELSE HERE COULD SEE IT: the generic
// AtF64 arm is unreachable because f64Data succeeds for both registered dtypes, the parallel
// test compares the kernel with ITSELF, and the F32 test is a 1e-5 tolerance
// (PERF-TOLERANCE-ORACLE-001, PERF-MASKED-BACKWARD-TARGET-001).
//
// The shapes reach both jams' edges and both mask variants: sk=32 is a whole number of groups,
// sk=19 and sk=13 leave remainders of 3 and 1, and perHead selects the second copy of each loop.
func TestMHAMaskedBackwardIsBitIdentical(t *testing.T) {
	for _, c := range []struct {
		sq, sk, dk, heads int
		perHead           bool
		want              uint64
	}{
		{16, 32, 16, 2, false, archgold.Pick(16308141056924124793, 15380585875560371548)},
		{9, 19, 8, 2, false, archgold.Pick(16345993731896056167, 5634437938168354934)},
		{7, 13, 8, 2, true, archgold.Pick(10045763495713164437, 16635919694738124331)},
		{16, 32, 16, 2, true, archgold.Pick(4901236805177551999, 2734271971446364057)},
	} {
		dm := c.heads * c.dk
		mk := func(rows, cols int, off float64) *tensor.Tensor {
			tn := tensor.New(tensor.F64, tensor.Shape{rows, cols})
			s := tn.Storage().F64()
			for i := range s {
				s[i] = math.Sin(float64(i)*0.019+off) * 0.4
			}
			return tn
		}
		q, k, v := mk(c.sq, dm, 0), mk(c.sk, dm, 1.1), mk(c.sk, dm, 2.3)
		g := mk(c.sq, dm, 3.7)
		var mask *tensor.Tensor
		if c.perHead {
			mask = tensor.New(tensor.F64, tensor.Shape{c.heads, c.sq, c.sk})
		} else {
			mask = tensor.New(tensor.F64, tensor.Shape{c.sq, c.sk})
		}
		n := mask.Numel()
		for i := range n {
			// About a fifth masked, so groups of four straddle the mask rather than aligning
			// with it — the case a block-aligned mask can never produce.
			val := 0.0
			if (i*7)%9 < 2 {
				val = math.Inf(-1)
			}
			co := tensor.Unravel(i, mask.Shape())
			mask.SetF64(val, co...)
		}
		out, err := std.table[kernelKey{backend.OpMHAMaskedBackward, tensor.F64}](
			backend.NewContext(), []*tensor.Tensor{q, k, v, mask, g},
			backend.AttnAttrs{Heads: c.heads})
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, tn := range out {
			if tn == nil {
				continue
			}
			cn := tn.Contiguous()
			for i := range cn.Numel() {
				u := math.Float64bits(cn.AtF64(tensor.Unravel(i, cn.Shape())...))
				for s := 0; s < 64; s += 8 {
					h = (h ^ (u>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("sq=%d sk=%d perHead=%v digest %d, want %d", c.sq, c.sk, c.perHead, h, c.want)
		}
	}
}
