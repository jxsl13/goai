package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// TestOptimizerFastPathParity guards the typed flat-loop fast paths (§V22,
// T652): every optimizer with a contiguous-f64/f32 fast path must produce a
// BIT-IDENTICAL trajectory to its generic widening-accessor fallback. It runs
// each optimizer twice over parameters with identical logical values — once on
// contiguous tensors (fast path) and once on transposed views (non-contiguous,
// so flatF64/flatF32 return nil and the generic path runs) — and requires exact
// equality after every step.
func TestOptimizerFastPathParity(t *testing.T) {
	opts := map[string]func(ps []*tensor.Tensor) nn.Optimizer{
		"SGD":       func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewSGD(ps, 1e-2, 0.9) },
		"Adam":      func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewAdamW(ps, 1e-3, 0.01) },
		"Lion":      func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewLion(ps, 1e-4, nn.WithLionWeightDecay(0.01)) },
		"LAMB":      func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewLAMB(ps, 1e-3, nn.WithLAMBWeightDecay(0.01)) },
		"AdEMAMix":  func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewAdEMAMix(ps, 1e-3) },
		"Adafactor": func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewAdafactor(ps) },
		"SOAP":      func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewSOAP(ps, 1e-3) },
		"Shampoo":   func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewShampoo(ps, 1e-3) },
		"Sophia":    func(ps []*tensor.Tensor) nn.Optimizer { return nn.NewSophia(ps, 1e-3) },
		"ScheduleFree": func(ps []*tensor.Tensor) nn.Optimizer {
			return nn.NewScheduleFreeAdamW(ps, 1e-3, nn.WithScheduleFreeWeightDecay(0.01))
		},
	}
	const steps = 4
	pval := func(i int) float64 { return math.Sin(float64(i)*0.7) + 0.1 }
	gval := func(i, s int) float64 { return math.Cos(float64(i)*0.3+float64(s)) * 0.05 }

	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for name, mk := range opts {
			t.Run(fmt.Sprintf("%s/%v", name, dt), func(t *testing.T) {
				// One 2-D parameter (matrix paths) and one 3-D parameter
				// (vector/diagonal paths of SOAP, Shampoo, Adafactor).
				shapes := []tensor.Shape{{3, 5}, {2, 3, 4}}
				fast := make([]*tensor.Tensor, len(shapes))
				slow := make([]*tensor.Tensor, len(shapes))
				for pi, s := range shapes {
					fast[pi] = tensor.New(dt, s)
					slow[pi] = mustTransposedView(t, dt, s)
					if slow[pi].IsContiguous() {
						t.Fatalf("view for %v is contiguous; parity test would not hit the generic path", s)
					}
					for i := range s.Numel() {
						idx := tensor.Unravel(i, s)
						fast[pi].SetF64(pval(i), idx...)
						slow[pi].SetF64(pval(i), idx...)
					}
				}
				optFast := mk(fast)
				optSlow := mk(slow)
				for s := range steps {
					gFast := map[*tensor.Tensor]*tensor.Tensor{}
					gSlow := map[*tensor.Tensor]*tensor.Tensor{}
					for pi, shape := range shapes {
						cg := tensor.New(dt, shape)
						vg := mustTransposedView(t, dt, shape)
						for i := range shape.Numel() {
							idx := tensor.Unravel(i, shape)
							cg.SetF64(gval(i, s), idx...)
							vg.SetF64(gval(i, s), idx...)
						}
						gFast[fast[pi]] = cg
						gSlow[slow[pi]] = vg
					}
					if err := optFast.Step(func(p *tensor.Tensor) *tensor.Tensor { return gFast[p] }); err != nil {
						t.Fatalf("fast step %d: %v", s, err)
					}
					if err := optSlow.Step(func(p *tensor.Tensor) *tensor.Tensor { return gSlow[p] }); err != nil {
						t.Fatalf("slow step %d: %v", s, err)
					}
					for pi, shape := range shapes {
						for i := range shape.Numel() {
							idx := tensor.Unravel(i, shape)
							fv, sv := fast[pi].AtF64(idx...), slow[pi].AtF64(idx...)
							if fv != sv {
								t.Fatalf("step %d param %d idx %v: fast path %.17g != generic path %.17g", s, pi, idx, fv, sv)
							}
						}
					}
				}
			})
		}
	}
}

// mustTransposedView returns a non-contiguous view with logical shape s: a
// tensor allocated with reversed dims, viewed through a dim-reversing Permute.
func mustTransposedView(t *testing.T, dt tensor.Dtype, s tensor.Shape) *tensor.Tensor {
	t.Helper()
	rev := make(tensor.Shape, len(s))
	perm := make([]int, len(s))
	for i := range s {
		rev[i] = s[len(s)-1-i]
		perm[i] = len(s) - 1 - i
	}
	base := tensor.New(dt, rev)
	view, err := base.Permute(perm...)
	if err != nil {
		t.Fatalf("permute %v: %v", perm, err)
	}
	return view
}
