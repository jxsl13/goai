package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// maRef independently computes ĝ = g + λ·μ for the Grokfast-MA filter, where μ is the mean (or
// sum) over a maxlen-W deque of the most recent gradients, gated by warmup (amplify only when the
// deque is full). Returns the sequence of filtered gradients.
func maRef(grads [][]float64, w int, lambda float64, warmup, filterSum bool) [][]float64 {
	dim := len(grads[0])
	var deque [][]float64
	out := make([][]float64, len(grads))
	for step, g := range grads {
		deque = append(deque, g)
		if len(deque) > w {
			deque = deque[1:]
		}
		gh := make([]float64, dim)
		amplify := !warmup || len(deque) == w
		for k := range dim {
			var s float64
			for _, e := range deque {
				s += e[k]
			}
			mu := s
			if !filterSum {
				mu = s / float64(len(deque))
			}
			gh[k] = g[k]
			if amplify {
				gh[k] += lambda * mu
			}
		}
		out[step] = gh
		_ = step
	}
	return out
}

// §V16 tier-1 (§R153 Grokfast-MA): the moving-average filtered gradient matches an independent
// deque recomputation across window eviction, with warmup off (amplify every step).
func TestGrokfastMARecurrenceNoWarmup(t *testing.T) {
	const dim, w, lambda = 2, 2, 3.0
	grads := [][]float64{{1, -2}, {0.5, -1}, {2, 0}, {-1, 1}}
	want := maRef(grads, w, lambda, false, false)

	p := tensor.New(tensor.F64, tensor.Shape{dim})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfastMA(cap, []*tensor.Tensor{p},
		nn.WithGrokfastMAWindow(w), nn.WithGrokfastMALambda(lambda), nn.WithGrokfastMAWarmup(false))
	for step, gv := range grads {
		g := tensor.FromFloat64(tensor.Shape{dim}, gv)
		if err := gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
		for k := range dim {
			if got := cap.last[p][k]; math.Abs(got-want[step][k]) > 1e-12 {
				t.Errorf("step %d ĝ[%d] = %.12g, want %.12g", step, k, got, want[step][k])
			}
		}
	}
}

// §V16 tier-1: with warmup (default) the first W−1 steps pass the raw gradient through; from the
// W-th step on the full-window mean is amplified. Matches the reference gating exactly.
func TestGrokfastMAWarmupGating(t *testing.T) {
	const dim, w, lambda = 2, 3, 4.0
	grads := [][]float64{{1, 1}, {2, -1}, {0.5, 2}, {-1, 0.5}, {3, -2}}
	want := maRef(grads, w, lambda, true, false)

	p := tensor.New(tensor.F64, tensor.Shape{dim})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfastMA(cap, []*tensor.Tensor{p},
		nn.WithGrokfastMAWindow(w), nn.WithGrokfastMALambda(lambda)) // warmup defaults true
	for step, gv := range grads {
		g := tensor.FromFloat64(tensor.Shape{dim}, gv)
		_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
		// steps 0,1 (deque not full) must be raw; steps ≥2 amplified.
		raw := step < w-1
		for k := range dim {
			got := cap.last[p][k]
			if raw && math.Abs(got-gv[k]) > 1e-12 {
				t.Errorf("warmup step %d ĝ[%d]=%g not raw %g", step, k, got, gv[k])
			}
			if math.Abs(got-want[step][k]) > 1e-12 {
				t.Errorf("step %d ĝ[%d] = %.12g, want %.12g", step, k, got, want[step][k])
			}
		}
	}
}

// §V16 tier-1: FilterSum uses the window SUM (not mean) as the slow component.
func TestGrokfastMASum(t *testing.T) {
	const dim, w, lambda = 2, 3, 1.0
	grads := [][]float64{{1, 2}, {3, 4}, {5, 6}, {7, 8}}
	want := maRef(grads, w, lambda, false, true)
	p := tensor.New(tensor.F64, tensor.Shape{dim})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfastMA(cap, []*tensor.Tensor{p}, nn.WithGrokfastMAWindow(w),
		nn.WithGrokfastMALambda(lambda), nn.WithGrokfastMAWarmup(false), nn.WithGrokfastMASum(true))
	for step, gv := range grads {
		g := tensor.FromFloat64(tensor.Shape{dim}, gv)
		_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
		for k := range dim {
			if got := cap.last[p][k]; math.Abs(got-want[step][k]) > 1e-12 {
				t.Errorf("sum step %d ĝ[%d] = %.12g, want %.12g", step, k, got, want[step][k])
			}
		}
	}
}

// §V16 tier-1: λ=0 recovers the base optimizer exactly (ĝ = g), regardless of window state.
func TestGrokfastMALambdaZeroNoOp(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{2})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfastMA(cap, []*tensor.Tensor{p}, nn.WithGrokfastMALambda(0), nn.WithGrokfastMAWarmup(false))
	for _, gv := range [][]float64{{1, 2}, {-3, 4}, {5, -6}} {
		g := tensor.FromFloat64(tensor.Shape{2}, gv)
		_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
		for k := range 2 {
			if got := cap.last[p][k]; math.Abs(got-gv[k]) > 1e-12 {
				t.Errorf("λ=0: ĝ[%d] = %g, want raw %g", k, got, gv[k])
			}
		}
	}
}

// §V2/convergence: wrapping SGD, Grokfast-MA still minimizes a convex quadratic.
func TestGrokfastMAConvergesSGD(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{3}, []float64{5, -3, 8})
	target := []float64{1, 1, 1}
	base := nn.NewSGD([]*tensor.Tensor{w}, 0.02, 0)
	gf := nn.NewGrokfastMA(base, []*tensor.Tensor{w},
		nn.WithGrokfastMALambda(2), nn.WithGrokfastMAWindow(5), nn.WithGrokfastMAWarmup(false))
	loss := func() float64 {
		var s float64
		for k := range 3 {
			d := w.AtF64(k) - target[k]
			s += d * d
		}
		return s
	}
	first := loss()
	for range 400 {
		err := gf.Step(func(p *tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, p.Shape())
			for k := range 3 {
				g.SetF64(2*(p.AtF64(k)-target[k]), k)
			}
			return g
		})
		if err != nil {
			t.Fatal(err)
		}
	}
	if last := loss(); last > 0.01*first {
		t.Errorf("did not converge: loss %.4g → %.4g", first, last)
	}
}

func ExampleGrokfastMA() {
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	base := nn.NewSGD([]*tensor.Tensor{w}, 0.1, 0)
	// Window 1, warmup off: μ = the single current gradient, so ĝ = g·(1+λ).
	gf := nn.NewGrokfastMA(base, []*tensor.Tensor{w},
		nn.WithGrokfastMALambda(2), nn.WithGrokfastMAWindow(1), nn.WithGrokfastMAWarmup(false))
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})
	_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	// SGD step with ĝ = g·(1+2) = 3: w ← 0 − 0.1·3 = −0.3.
	fmt.Printf("%.1f %.1f\n", w.AtF64(0), w.AtF64(1))
	// Output:
	// -0.3 -0.3
}
