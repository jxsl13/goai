package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// constGrad returns a fixed gradient for p regardless of its value, so the SGD
// trajectory is analytically known.
func constGrad(p *tensor.Tensor, g float64) nn.GradFn {
	grad := tensor.New(tensor.F64, p.Shape())
	for i := range grad.Numel() {
		grad.SetF64(g, tensor.Unravel(i, grad.Shape())...)
	}
	return func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		return grad
	}
}

// §V16 tier-1 (§R98): after k inner SGD steps Lookahead interpolates the slow weights
// toward the fast ones and resets to them: with a constant gradient g and step lr,
// θ_k = θ₀ − k·lr·g, so the parameters become φ = (1−α)·θ₀ + α·θ_k = θ₀ − α·k·lr·g.
func TestLookaheadInterpolation(t *testing.T) {
	const lr, g, alpha, k = 0.1, 1.0, 0.5, 3
	p0 := []float64{1, 2, 3}
	p := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), p0...))
	la := nn.NewLookahead(nn.NewSGD([]*tensor.Tensor{p}, lr, 0), []*tensor.Tensor{p},
		nn.WithLookaheadK(k), nn.WithLookaheadAlpha(alpha))
	gf := constGrad(p, g)
	for range k {
		if err := la.Step(gf); err != nil {
			t.Fatal(err)
		}
	}
	for i := range p0 {
		want := p0[i] - alpha*k*lr*g // = θ₀ − α·k·lr·g
		if got := p.AtF64(i); math.Abs(got-want) > 1e-12 {
			t.Errorf("param[%d] = %v, want φ = %v", i, got, want)
		}
	}
}

// §V16 tier-1: before the k-th step no slow update happens — Lookahead(SGD) is just SGD
// mid-cycle.
func TestLookaheadNoSyncMidCycle(t *testing.T) {
	const lr, g, k = 0.1, 1.0, 3
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 5})
	la := nn.NewLookahead(nn.NewSGD([]*tensor.Tensor{p}, lr, 0), []*tensor.Tensor{p}, nn.WithLookaheadK(k))
	gf := constGrad(p, g)
	for range k - 1 { // 2 of 3 steps → plain SGD, no interpolation yet
		_ = la.Step(gf)
	}
	for i, p0 := range []float64{1, 5} {
		want := p0 - float64(k-1)*lr*g // plain SGD
		if got := p.AtF64(i); math.Abs(got-want) > 1e-12 {
			t.Errorf("mid-cycle param[%d] = %v, want plain-SGD %v", i, got, want)
		}
	}
}

// §V16 tier-1: two full cycles compose — the slow weights EMA repeats every k steps.
// After cycle 1 φ₁ = θ₀ − α·k·lr·g; cycle 2 starts from φ₁ so θ₂ₖ = φ₁ − k·lr·g and
// φ₂ = (1−α)φ₁ + α·θ₂ₖ.
func TestLookaheadTwoCycles(t *testing.T) {
	const lr, g, alpha, k = 0.1, 1.0, 0.5, 3
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{2})
	la := nn.NewLookahead(nn.NewSGD([]*tensor.Tensor{p}, lr, 0), []*tensor.Tensor{p},
		nn.WithLookaheadK(k), nn.WithLookaheadAlpha(alpha))
	gf := constGrad(p, g)
	for range 2 * k {
		_ = la.Step(gf)
	}
	phi1 := 2 - alpha*k*lr*g
	theta2k := phi1 - k*lr*g
	phi2 := (1-alpha)*phi1 + alpha*theta2k
	if got := p.AtF64(0); math.Abs(got-phi2) > 1e-12 {
		t.Errorf("after two cycles param = %v, want %v", got, phi2)
	}
}

// Lookahead wraps a base optimizer with slow weights that follow the fast ones every k
// steps. Wrapping SGD (lr 0.1) with k=2, α=0.5 and a constant gradient of 1: the fast
// weight moves to 0.8 over two steps, and the slow weight lands halfway, at 0.90.
func ExampleLookahead() {
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	la := nn.NewLookahead(nn.NewSGD([]*tensor.Tensor{p}, 0.1, 0), []*tensor.Tensor{p},
		nn.WithLookaheadK(2), nn.WithLookaheadAlpha(0.5))
	gf := constGrad(p, 1)
	_ = la.Step(gf)
	_ = la.Step(gf)
	fmt.Printf("%.2f\n", p.AtF64(0))
	// Output: 0.90
}
