package nn

import (
	"fmt"
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// slowFocalOneHot is the per-element reference (independent index math via
// AtF64/SetF64) that the typed fillFocalOneHot fast path must match cell-for-cell.
// Kept ONLY as a test oracle — an off-by-one in the fast path's i*classes+c flat
// index would diverge from SetF64's flatOffset math and fail here.
func slowFocalOneHot(dtype tensor.Dtype, targets *tensor.Tensor, batch, classes int) (*tensor.Tensor, error) {
	oh := tensor.New(dtype, tensor.Shape{batch, classes})
	for i := range batch {
		c := int(targets.AtF64(i))
		if c < 0 || c >= classes {
			return nil, fmt.Errorf("nn: FocalLoss target %d out of range [0,%d)", c, classes)
		}
		oh.SetF64(1, i, c)
	}
	return oh, nil
}

// TestFillFocalOneHotBitIdentical pins the typed fast path (contiguous F32/F64
// targets, which tensor.New produces → the branch IS exercised, not vacuously
// falling through) against the per-element oracle across both dtypes.
func TestFillFocalOneHotBitIdentical(t *testing.T) {
	const batch, classes = 257, 13 // non-round to expose stride/index errors
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		targets := tensor.New(dt, tensor.Shape{batch})
		rng := rand.New(rand.NewPCG(1, 2))
		for i := range batch {
			targets.SetF64(float64(rng.IntN(classes)), i)
		}
		want, err := slowFocalOneHot(dt, targets, batch, classes)
		if err != nil {
			t.Fatalf("dtype %v: oracle: %v", dt, err)
		}
		got := tensor.New(dt, tensor.Shape{batch, classes})
		if err := fillFocalOneHot(got, targets, batch, classes); err != nil {
			t.Fatalf("dtype %v: fast: %v", dt, err)
		}
		for i := range batch {
			for c := range classes {
				if got.AtF64(i, c) != want.AtF64(i, c) {
					t.Fatalf("dtype %v: cell [%d,%d]: fast %v != oracle %v", dt, i, c, got.AtF64(i, c), want.AtF64(i, c))
				}
			}
		}
	}
}

// TestFillFocalOneHotOutOfRangeSameError checks the fast path reports the SAME
// first-out-of-range error as the general path (the loss's guarantee, §V16).
func TestFillFocalOneHotOutOfRangeSameError(t *testing.T) {
	targets := tensor.New(tensor.F64, tensor.Shape{3})
	targets.SetF64(0, 0)
	targets.SetF64(5, 1) // out of range for classes=2
	targets.SetF64(1, 2)
	err := fillFocalOneHot(tensor.New(tensor.F64, tensor.Shape{3, 2}), targets, 3, 2)
	if err == nil || !strings.Contains(err.Error(), "target 5 out of range [0,2)") {
		t.Fatalf("want first-out-of-range error, got %v", err)
	}
}
