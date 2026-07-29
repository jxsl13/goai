package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// retChunkSum hashes RetentionChunkwise's output bits. The existing duality test compares
// against the parallel form at 1e-10, which cannot see a reassociation; this pins the exact
// bits so the 4-way blocking of the output loop is held to its bit-identity claim.
func retChunkSum(seq, d, cs int, gamma float64) uint64 {
	mk := func(seed float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{seq, d})
		s := x.Storage().F64()
		for i := range s {
			s[i] = math.Sin(seed + 1.7*float64(i))
		}
		return x
	}
	out, err := nn.RetentionChunkwise(mk(1), mk(2), mk(3), gamma, cs)
	if err != nil {
		return 0
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

func TestRetentionChunkwiseBitIdentical(t *testing.T) {
	for _, g := range retChunkGolden {
		seq, d, cs := int(g[0]), int(g[1]), int(g[2])
		if got := retChunkSum(seq, d, cs, 0.9); got != g[3] {
			t.Fatalf("seq=%d d=%d cs=%d: checksum %d, want %d", seq, d, cs, got, g[3])
		}
	}
}
