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

// captureOpt is a mock Optimizer that records the (filtered) gradient it is handed.
type captureOpt struct {
	params []*tensor.Tensor
	last   map[*tensor.Tensor][]float64
}

func (c *captureOpt) Step(grad nn.GradFn) error {
	c.last = make(map[*tensor.Tensor][]float64)
	for _, p := range c.params {
		g := grad(p)
		flat := make([]float64, p.Numel())
		for i := range flat {
			flat[i] = g.AtF64(tensor.Unravel(i, p.Shape())...)
		}
		c.last[p] = flat
	}
	return nil
}

// §V16 tier-1: the filtered gradient ĝ = g + λ·μ with μ_t = α·μ_{t−1} + (1−α)·g_t
// (μ seeded with g_0) matches an independent recomputation over several steps.
func TestGrokfastEMARecurrence(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{3})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	const alpha, lambda = 0.9, 2.0
	gf := nn.NewGrokfast(cap, []*tensor.Tensor{p}, nn.WithGrokfastAlpha(alpha), nn.WithGrokfastLambda(lambda))

	grads := [][]float64{{1, -2, 0.5}, {0.5, -1, 1}, {2, 0, -0.5}, {-1, 1, 0.25}}
	mu := []float64{0, 0, 0}
	for step, gv := range grads {
		g := tensor.FromFloat64(tensor.Shape{3}, gv)
		if err := gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
		// independent recurrence
		for k := range 3 {
			if step == 0 {
				mu[k] = gv[k] // seed with first gradient
			}
			mu[k] = alpha*mu[k] + (1-alpha)*gv[k]
			want := gv[k] + lambda*mu[k]
			if got := cap.last[p][k]; math.Abs(got-want) > 1e-12 {
				t.Errorf("step %d ĝ[%d] = %.12g, want %.12g", step, k, got, want)
			}
		}
	}
}

// §V16 tier-1: on the first step μ = g_0, so ĝ = g_0·(1+λ).
func TestGrokfastInitFirstGrad(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{2})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfast(cap, []*tensor.Tensor{p}, nn.WithGrokfastLambda(2), nn.WithGrokfastAlpha(0.98))
	g0 := tensor.FromFloat64(tensor.Shape{2}, []float64{3, -4})
	_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g0 })
	for k, want := range []float64{3 * 3, -4 * 3} { // g·(1+λ)=3g
		if got := cap.last[p][k]; math.Abs(got-want) > 1e-12 {
			t.Errorf("ĝ[%d] = %g, want %g", k, got, want)
		}
	}
}

// §V16 tier-1: λ=0 recovers the base optimizer exactly (ĝ = g).
func TestGrokfastLambdaZeroNoOp(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{2})
	cap := &captureOpt{params: []*tensor.Tensor{p}}
	gf := nn.NewGrokfast(cap, []*tensor.Tensor{p}, nn.WithGrokfastLambda(0))
	for _, gv := range [][]float64{{1, 2}, {-3, 4}} {
		g := tensor.FromFloat64(tensor.Shape{2}, gv)
		_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
		for k := range 2 {
			if got := cap.last[p][k]; math.Abs(got-gv[k]) > 1e-12 {
				t.Errorf("λ=0: ĝ[%d] = %g, want raw %g", k, got, gv[k])
			}
		}
	}
}

// §V2/convergence: wrapping SGD, Grokfast still minimizes a convex quadratic.
func TestGrokfastConvergesSGD(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{3}, []float64{5, -3, 8})
	target := []float64{1, 1, 1}
	base := nn.NewSGD([]*tensor.Tensor{w}, 0.02, 0)
	gf := nn.NewGrokfast(base, []*tensor.Tensor{w}, nn.WithGrokfastLambda(2), nn.WithGrokfastAlpha(0.9))

	loss := func() float64 {
		var s float64
		for k := range 3 {
			d := w.AtF64(k) - target[k]
			s += d * d
		}
		return s
	}
	first := loss()
	for range 300 {
		err := gf.Step(func(p *tensor.Tensor) *tensor.Tensor {
			g := tensor.New(tensor.F64, p.Shape())
			for k := range 3 {
				g.SetF64(2*(p.AtF64(k)-target[k]), k) // ∇ Σ(w−t)²
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

func ExampleGrokfast() {
	w := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	base := nn.NewSGD([]*tensor.Tensor{w}, 0.1, 0)
	// Wrap the base optimizer; the first step amplifies the gradient by (1+λ).
	gf := nn.NewGrokfast(base, []*tensor.Tensor{w}, nn.WithGrokfastLambda(2), nn.WithGrokfastAlpha(0.98))
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})
	_ = gf.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	// SGD step with ĝ = g·(1+2) = 3: w ← 0 − 0.1·3 = −0.3.
	fmt.Printf("%.1f %.1f\n", w.AtF64(0), w.AtF64(1))
	// Output:
	// -0.3 -0.3
}
