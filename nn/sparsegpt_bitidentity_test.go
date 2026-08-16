package nn_test

import (
	"github.com/jxsl13/goai/internal/archgold"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestSparseGPTPruneIsBitIdentical freezes the pruned weights and the mask. Making the block
// body row-outer and banding it claims to change no value — rows never interact, so choosing a
// row's mask before compensating that row rather than after choosing every row's is the same
// arithmetic in the same order — and a pruner is somewhere a changed decision would look like
// a different-but-plausible sparsity pattern rather than a bug.
//
// One shape is below the fan-out gate and two clear it; the N:M case is included because it
// takes a different kOf and a different mask rule than unstructured sparsity.
func TestSparseGPTPruneIsBitIdentical(t *testing.T) {
	cases := []struct {
		out, in, samples int
		nm               bool
		want             uint64
	}{
		{4, 8, 16, false, archgold.Pick(13628133690511030499, 8973122112660946674)},
		{48, 96, 64, false, archgold.Pick(6530497775513801175, 10559994915875960948)},
		{32, 64, 48, true, archgold.Pick(18248866684583792410, 11466859589314545682)},
	}
	for _, c := range cases {
		w := tensor.New(tensor.F64, tensor.Shape{c.out, c.in})
		ws := w.Storage().F64()
		for i := range ws {
			ws[i] = math.Sin(float64(i)*0.37) * 0.5
		}
		x := tensor.New(tensor.F64, tensor.Shape{c.in, c.samples})
		xs := x.Storage().F64()
		for i := range xs {
			xs[i] = math.Cos(float64(i)*0.11) + 0.01*float64(i%7)
		}
		var pruned, mask *tensor.Tensor
		var err error
		if c.nm {
			pruned, mask, err = nn.SparseGPTPruneNM(w, x, 2, 4)
		} else {
			pruned, mask, err = nn.SparseGPTPrune(w, x, 0.5)
		}
		if err != nil {
			t.Fatal(err)
		}
		h := uint64(14695981039346656037)
		for _, tn := range []*tensor.Tensor{pruned, mask} {
			for _, v := range tn.Storage().F64() {
				b := math.Float64bits(v)
				for s := 0; s < 64; s += 8 {
					h = (h ^ (b>>s)&0xff) * 1099511628211
				}
			}
		}
		if h != c.want {
			t.Fatalf("%dx%d nm=%v digest %d, want %d", c.out, c.in, c.nm, h, c.want)
		}
	}
}
