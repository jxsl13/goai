package nn_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// MoBA picks the top-K past blocks per query and attends only to those. The selection is a
// set membership test in the innermost loop, and replacing the map that held it with a
// generation-stamped slice must not move a single output bit — a different SET would change
// which keys are scored, which any tolerance test would happily accept as "close enough."
//
// Shapes that matter here: a sequence that is NOT a multiple of the block size, so the last
// block is short and blockLen differs; and a topK large enough to select every past block on
// some queries and small enough to reject on others, so both arms of the membership test run.
func mobaDigest(t *testing.T, seq, dm, heads, blockSize, topK int) uint64 {
	t.Helper()
	mk := func(fn func(i int) float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{seq, dm})
		s := x.Storage().F64()
		for i := range s {
			s[i] = fn(i)
		}
		return x
	}
	q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.01) })
	k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.013) })
	v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.017) })
	out, err := nn.MoBAAttention(q, k, v, heads, blockSize, topK, 0)
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	n := out.Numel()
	for _, val := range out.Storage().F64()[:n] {
		u := math.Float64bits(val)
		for s := 0; s < 64; s += 8 {
			h = (h ^ (u>>s)&0xff) * 1099511628211
		}
	}
	return h
}

func TestMoBAIsBitIdentical(t *testing.T) {
	cases := []struct {
		seq, dm, heads, blockSize, topK int
		want                            uint64
	}{
		{37, 32, 2, 8, 2, archgold.Pick(4168332973237042244, 8328200919833524082)},   // 37 = 4 blocks of 8 plus a short one; topK rejects
		{37, 32, 2, 8, 16, archgold.Pick(10924912427930123379, 828959329962388704)},  // topK exceeds the block count: every past block selected
		{64, 32, 4, 16, 2, archgold.Pick(6028010401060052134, 15043508461982351155)}, // block-aligned
		{9, 16, 1, 4, 1, archgold.Pick(13685344069298453308, 2052641934387284795)},   // topK=1: only the current block is ever in the set
	}
	for _, c := range cases {
		got := mobaDigest(t, c.seq, c.dm, c.heads, c.blockSize, c.topK)
		if got != c.want {
			t.Errorf("seq=%d heads=%d block=%d topK=%d: digest %d",
				c.seq, c.heads, c.blockSize, c.topK, got)
		}
	}
}
