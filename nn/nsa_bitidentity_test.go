package nn_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestNSABranchesAreBitIdentical freezes all three branch outputs. Jamming the score dot and the
// P·V accumulation four keys per pass claims to change no value, and every other NSA test is a
// 1e-12 tolerance — the parallel-equivalence test compares the jammed code against ITSELF, so
// none of them would notice a reassociation (PERF-TOLERANCE-ORACLE-001).
//
// The shapes reach both jams' edges: seq=96 with block 16 leaves whole groups, and seq=67 and
// seq=41 leave remainders of 3 and 1 against a four-wide pass, with the selection mask making
// groups straddle kept and dropped keys.
func TestNSABranchesAreBitIdentical(t *testing.T) {
	for _, c := range []struct {
		seq, dm, heads, blk, sel, win int
		want                          uint64
	}{
		{96, 128, 4, 16, 4, 32, archgold.Pick(17751161931858554633, 8790598743836276329)},
		{67, 64, 2, 8, 3, 16, archgold.Pick(1408904477776929825, 17381712554975422382)},
		{41, 32, 2, 4, 2, 8, archgold.Pick(17544698765182081482, 16852215822321138725)},
		// A block size that is NOT a multiple of four, so a four-wide group straddles the
		// selection mask and the jam has to fall back inside the group.
		{53, 32, 2, 6, 3, 10, archgold.Pick(17668651896359838193, 16170567050796504078)},
	} {
		mk := func(f func(i int) float64) *tensor.Tensor {
			tt := tensor.New(tensor.F64, tensor.Shape{c.seq, c.dm})
			s := tt.Storage().F64()
			for i := range s {
				s[i] = f(i)
			}
			return tt
		}
		q := mk(func(i int) float64 { return math.Sin(float64(i) * 0.021) })
		k := mk(func(i int) float64 { return math.Cos(float64(i) * 0.017) })
		v := mk(func(i int) float64 { return math.Sin(float64(i) * 0.013) })
		cmp, slc, win, err := nn.NSABranches(q, k, v, c.heads, c.blk, c.sel, c.win, 0)
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, tn := range []*tensor.Tensor{cmp, slc, win} {
			for _, val := range tn.Storage().F64() {
				u := math.Float64bits(val)
				for s := 0; s < 64; s += 8 {
					h = (h ^ (u>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("seq=%d dm=%d digest %d, want %d", c.seq, c.dm, h, c.want)
		}
	}
}
