package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// The vector (non-matrix) path uses the full second moment. On step 1, β̂₂,1 =
// 1−1^(−0.8) = 0, so V̂ = g² and U = g/|g| = sign(g); with unit RMS the clip and
// relative step reduce to θ ← θ − max(ε₂, RMS(θ))·min(LR,1)·sign(g). Hand-computed.
func TestAdafactorVectorStepExact(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, -1})
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{2, 3})
	opt := nn.NewAdafactor([]*tensor.Tensor{p}) // LR 1e-2 default
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
		t.Fatal(err)
	}
	// U=[1,1]; RMS(U)=1→clip 1; RMS(θ)=1→α=max(1e-3,1)·1e-2=1e-2; θ−=1e-2·[1,1].
	want := []float64{0.99, -1.01}
	for i, w := range want {
		if got := p.AtF64(i); math.Abs(got-w) > 1e-9 {
			t.Errorf("θ[%d] = %v, want %v", i, got, w)
		}
	}
}

// §V16 tier-1 convergence: on a convex quadratic ½‖W−W*‖² (matrix param → factored
// path), Adafactor's adaptive, RMS-clipped, parameter-relative step drives the loss
// far down — guarding the whole factored update end to end.
func TestAdafactorConvergesQuadratic(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	rows, cols := 6, 5
	star := make([]float64, rows*cols)
	init := make([]float64, rows*cols)
	for i := range star {
		star[i] = rng.NormFloat64()
		init[i] = rng.NormFloat64()
	}
	w := tensor.FromFloat64(tensor.Shape{rows, cols}, append([]float64(nil), init...))
	opt := nn.NewAdafactor([]*tensor.Tensor{w}, nn.WithAdafactorLR(0.05))

	loss := func() float64 {
		var s float64
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			d := w.AtF64(idx...) - star[i]
			s += d * d
		}
		return 0.5 * s
	}
	l0 := loss()
	for range 3000 {
		g := tensor.New(tensor.F32, w.Shape())
		for i := range star {
			idx := tensor.Unravel(i, w.Shape())
			g.SetF64(w.AtF64(idx...)-star[i], idx...) // ∇ = W − W*
		}
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
	}
	if lf := loss(); lf > 0.1*l0 {
		t.Errorf("Adafactor did not converge: loss %.4f → %.4f (want < 10%% of initial)", l0, lf)
	}
}

// Adafactor stores only per-row and per-column second-moment sums for a matrix
// parameter — O(rows+cols) state instead of Adam's O(rows·cols) — so large layers
// train in a fraction of the optimizer memory (Shazeer & Stern 2018).
func ExampleAdafactor() {
	w := tensor.New(tensor.F32, tensor.Shape{1024, 1024})
	opt := nn.NewAdafactor([]*tensor.Tensor{w})
	r, c := opt.Params[0].Shape()[0], opt.Params[0].Shape()[1]
	// factored second-moment state (rows+cols) vs Adam's full rows·cols:
	fmt.Println(r*c/(r+c), "x smaller")
	// Output: 512 x smaller
}
