package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func wandaSum(t *testing.T, cout, cin, samples int, sparsity float64, nm, f32 bool) uint64 {
	t.Helper()
	// w is [cin, cout] and x is [tokens, cin] — the shapes wandaInputs builds.
	dt := tensor.F64
	if f32 {
		dt = tensor.F32
	}
	w := tensor.New(dt, tensor.Shape{cin, cout})
	for i := range w.Numel() {
		w.SetF64(math.Sin(float64(i)*0.37), tensor.Unravel(i, w.Shape())...)
	}
	x := tensor.New(dt, tensor.Shape{samples, cin})
	for i := range x.Numel() {
		x.SetF64(math.Cos(float64(i)*0.013), tensor.Unravel(i, x.Shape())...)
	}
	var pruned, mask *tensor.Tensor
	var err error
	if nm {
		pruned, mask, err = nn.WandaPruneNM(w, x, 2, 4)
	} else {
		pruned, mask, err = nn.WandaPrune(w, x, sparsity)
	}
	if err != nil {
		t.Fatal(err)
	}
	h := uint64(14695981039346656037)
	for _, tn := range []*tensor.Tensor{pruned, mask} {
		for i := range tn.Numel() {
			b := math.Float64bits(tn.AtF64(tensor.Unravel(i, tn.Shape())...))
			for s := 0; s < 64; s += 8 {
				h = (h ^ ((b >> s) & 0xff)) * 1099511628211
			}
		}
	}
	return h
}

// TestWandaBitIdentical pins both prune paths. The column sort moved from sort.Slice to
// slices.SortFunc and the block sort from sort.SliceStable to slices.SortStableFunc; the
// first is safe because its comparator is TOTAL (score, then index), the second because
// stability is preserved. Ties are what a sort swap can reorder, so the fixture deliberately
// includes shapes where scores repeat.
func TestWandaBitIdentical(t *testing.T) {
	for _, c := range wandaGolden {
		if got := wandaSum(t, c.cout, c.cin, c.samples, c.sparsity, c.nm, c.f32); got != c.sum {
			t.Fatalf("cout=%d cin=%d samples=%d nm=%v f32=%v: checksum %d, want %d",
				c.cout, c.cin, c.samples, c.nm, c.f32, got, c.sum)
		}
	}
}

type wandaCase struct {
	cout, cin, samples int
	sparsity           float64
	nm, f32            bool
	sum                uint64
}
