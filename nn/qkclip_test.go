package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

func qkClipSetup(seed int) (x, wq, wk *tensor.Tensor) {
	const seq, dm = 6, 8
	mk := func(s int, rows, cols int, scale float64) *tensor.Tensor {
		t := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range t.Numel() {
			t.SetF64(math.Sin(float64(s*100+i)*1.1)*scale, tensor.Unravel(i, t.Shape())...)
		}
		return t
	}
	return mk(seed, seq, dm, 1), mk(seed+1, dm, dm, 1.2), mk(seed+2, dm, dm, 1.2)
}

func cloneT(t *tensor.Tensor) *tensor.Tensor {
	out := tensor.New(t.Dtype(), t.Shape())
	for i := range t.Numel() {
		idx := tensor.Unravel(i, t.Shape())
		out.SetF64(t.AtF64(idx...), idx...)
	}
	return out
}

func project(t *testing.T, x, w *tensor.Tensor) *tensor.Tensor {
	out, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return out[0]
}

// §T550: the defining QK-Clip property — after clipping with threshold τ, every head's
// recomputed max attention logit is ≤ τ (up to fp), heads already within budget keep
// their weights BIT-identical, and the clipped head's logits shrink by exactly
// τ/maxLogit (both projections scaled by the square root).
func TestQKClipCapsLogits(t *testing.T) {
	const heads, dk, scale = 2, 4, 0.5
	x, wq, wk := qkClipSetup(1)
	q, k := project(t, x, wq), project(t, x, wk)
	before, err := nn.MaxAttentionLogits(q, k, heads, scale, true)
	if err != nil {
		t.Fatal(err)
	}
	// pick τ between the two heads' maxima so exactly one head clips.
	lo, hi := math.Min(before[0], before[1]), math.Max(before[0], before[1])
	if hi <= lo {
		t.Fatalf("degenerate setup: %v", before)
	}
	tau := (lo + hi) / 2
	wqOrig := cloneT(wq)

	clipped, err := nn.QKClip(wq, wk, heads, before, tau)
	if err != nil {
		t.Fatal(err)
	}
	if clipped != 1 {
		t.Fatalf("clipped %d heads, want exactly 1 (τ between the maxima)", clipped)
	}
	q2, k2 := project(t, x, wq), project(t, x, wk)
	after, err := nn.MaxAttentionLogits(q2, k2, heads, scale, true)
	if err != nil {
		t.Fatal(err)
	}
	for h := range heads {
		if after[h] > tau*(1+1e-12) {
			t.Fatalf("head %d still exceeds τ: %g > %g", h, after[h], tau)
		}
		if before[h] <= tau {
			// untouched head: weights bit-identical.
			for c := h * dk; c < (h+1)*dk; c++ {
				for r := range wq.Shape()[0] {
					if wq.AtF64(r, c) != wqOrig.AtF64(r, c) {
						t.Fatalf("head %d under τ was modified at (%d,%d)", h, r, c)
					}
				}
			}
		} else if math.Abs(after[h]-tau) > 1e-9*tau {
			t.Fatalf("clipped head %d: max logit %g, want exactly τ=%g (γ² scaling)", h, after[h], tau)
		}
	}
}

// Training-loop usage: probe the projected activations after each optimizer step
// and clip heads whose max logit drifted past the budget.
func ExampleQKClip() {
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 0, 0, 1, 0, 2, 1, 0})
	wq := tensor.FromFloat64(tensor.Shape{4, 4}, []float64{
		4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0.1,
	})
	wk := cloneT(wq)
	q := cloneT(x) // for the example, pretend x is already projected
	k := cloneT(x)
	maxes, _ := nn.MaxAttentionLogits(q, k, 2, 1, false)
	n, _ := nn.QKClip(wq, wk, 2, maxes, 100)
	fmt.Println(n) // both heads' probe logits are ≤ 100: nothing to clip
	// Output: 0
}
