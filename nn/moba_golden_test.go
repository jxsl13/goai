package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

type mobaGeom struct {
	seq, dm, heads, block, topK int
	sum                         uint64
}

func mobaChecksum(t *testing.T, g mobaGeom) uint64 {
	t.Helper()
	mk := func(seed float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{g.seq, g.dm})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(seed + 1.7*float64(i))
		}
		return x
	}
	out, err := nn.MoBAAttention(mk(1), mk(2), mk(3), g.heads, g.block, g.topK, 0)
	if err != nil {
		t.Fatalf("%+v: %v", g, err)
	}
	h := uint64(14695981039346656037)
	for i := range out.Numel() {
		b := math.Float64bits(out.AtF64(tensor.Unravel(i, out.Shape())...))
		for s := 0; s < 64; s += 8 {
			h = (h ^ ((b >> s) & 0xff)) * 1099511628211
		}
	}
	return h
}

// TestMoBABitIdentical guards optimizations to MoBAAttention. Constants captured from the
// implementation before the P·V rewrite; a single moved bit fails here. Geometries cover a
// partial trailing block, topK at and above the block count, and blockSize 1.
func TestMoBABitIdentical(t *testing.T) {
	for _, g := range mobaGolden {
		if got := mobaChecksum(t, g); got != g.sum {
			t.Fatalf("seq=%d dm=%d heads=%d block=%d topK=%d: checksum %d, want %d",
				g.seq, g.dm, g.heads, g.block, g.topK, got, g.sum)
		}
	}
}
