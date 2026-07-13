package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the Sophia trajectory reproduces the paper's Algorithm 3 recurrence
// (m EMA, decoupled weight decay, then θ −= η·clip(m/max(γ·h,ε),1)) step by step,
// checked against an independent straight-line recomputation of the same equations at
// rtol 1e-12. The Hessian is refreshed every k=2 steps via UpdateHessian.
func TestSophiaAlgorithmParity(t *testing.T) {
	const (
		lr, b1, b2, gamma, eps, wd = 0.1, 0.9, 0.99, 0.05, 1e-12, 0.05
		k                          = 2
	)
	p0 := []float64{1.0, -2.0, 0.5}
	hhat := []float64{4.0, 0.01, 1.0} // second coord has tiny curvature → clips
	grads := [][]float64{
		{0.5, 0.5, -0.3},
		{0.2, 0.4, -0.1},
		{-0.4, 0.6, 0.2},
		{0.1, 0.5, 0.0},
	}

	// independent reference recurrence (a second implementation of Algorithm 3)
	clip1 := func(z float64) float64 { return math.Max(-1, math.Min(1, z)) }
	ref := append([]float64(nil), p0...)
	rm := make([]float64, len(p0))
	rh := make([]float64, len(p0))
	want := make([][]float64, len(grads))
	for s := range grads {
		if s%k == 0 { // Hessian refresh at t mod k == 1 (0-indexed: s%k==0)
			for i := range rh {
				rh[i] = b2*rh[i] + (1-b2)*hhat[i]
			}
		}
		for i := range ref {
			rm[i] = b1*rm[i] + (1-b1)*grads[s][i]
			ref[i] = ref[i]*(1-lr*wd) - lr*clip1(rm[i]/math.Max(gamma*rh[i], eps))
		}
		want[s] = append([]float64(nil), ref...)
	}

	// optimizer under test
	p := tensor.FromFloat64(tensor.Shape{len(p0)}, append([]float64(nil), p0...))
	opt := nn.NewSophia([]*tensor.Tensor{p}, lr,
		nn.WithSophiaBetas(b1, b2), nn.WithSophiaGamma(gamma),
		nn.WithSophiaEps(eps), nn.WithSophiaWeightDecay(wd))
	hfn := func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		return tensor.FromFloat64(p.Shape(), hhat)
	}
	for s := range grads {
		if s%k == 0 {
			if err := opt.UpdateHessian(hfn); err != nil {
				t.Fatal(err)
			}
		}
		gs := grads[s]
		if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		for i := range p.Numel() {
			got := p.AtF64(i)
			if math.Abs(got-want[s][i]) > 1e-12*math.Max(1, math.Abs(want[s][i])) {
				t.Errorf("step %d p[%d]: got %.14g want %.14g", s, i, got, want[s][i])
			}
		}
	}
}

// §V16 tier-1: Sophia's defining safety property — the clipped update can never move
// a coordinate by more than the learning rate, no matter how large the ratio, so a
// coordinate with (near-)zero estimated curvature still takes only a ±η sign step.
func TestSophiaClipsToLearningRate(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 0})
	opt := nn.NewSophia([]*tensor.Tensor{p}, 0.1) // no Hessian populated → h=0 → max(γ·0,ε)=ε
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{1e6, -1e6})
	if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
		if q != p {
			return nil
		}
		return g
	}); err != nil {
		t.Fatal(err)
	}
	if got := p.AtF64(0); math.Abs(got-(-0.1)) > 1e-12 {
		t.Errorf("huge +grad must clip to −lr: got %v want -0.1", got)
	}
	if got := p.AtF64(1); math.Abs(got-0.1) > 1e-12 {
		t.Errorf("huge −grad must clip to +lr: got %v want 0.1", got)
	}
}

// §V16 tier-1: with zero gradient the only motion is the decoupled AdamW-style weight
// decay θ ← θ·(1−η·λ), applied exactly.
func TestSophiaDecoupledWeightDecay(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{2.0})
	opt := nn.NewSophia([]*tensor.Tensor{p}, 0.1, nn.WithSophiaWeightDecay(0.5))
	zero := tensor.FromFloat64(tensor.Shape{1}, []float64{0})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return zero }); err != nil {
		t.Fatal(err)
	}
	if got, want := p.AtF64(0), 2.0*(1-0.1*0.5); math.Abs(got-want) > 1e-12 {
		t.Errorf("weight decay: got %v want %v", got, want)
	}
}

// Sophia clips every per-coordinate update to ±learning-rate. With no Hessian
// estimate populated the denominator is the ε floor, so any nonzero-gradient
// coordinate takes a full sign step: here θ=0.5 with a positive gradient steps down
// by exactly the learning rate 0.1.
func ExampleSophia() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{0.5})
	opt := nn.NewSophia([]*tensor.Tensor{w}, 0.1)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{1}, []float64{2.0})
	})
	fmt.Printf("%.2f\n", w.AtF64(0))
	// Output: 0.40
}
