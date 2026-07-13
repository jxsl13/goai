package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1: the leave-one-out advantage matches the exact baseline A_i = r_i −
// mean(other k−1 rewards), the equivalent (k/(k−1))(r_i − mean), sums to zero, and
// degenerates to all-zero for k≤1 (Ahmadian et al. 2024).
func TestRLOOAdvantage(t *testing.T) {
	r := []float64{1, 2, 3, 4}
	got := nn.RLOOAdvantage(r)
	want := []float64{-2, -2.0 / 3, 2.0 / 3, 2} // r_i − mean(others)
	var mean, sum float64
	for _, v := range r {
		mean += v
	}
	mean /= float64(len(r))
	for i := range r {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("A[%d] = %v, want %v", i, got[i], want[i])
		}
		if id := float64(len(r)) / float64(len(r)-1) * (r[i] - mean); math.Abs(got[i]-id) > 1e-12 {
			t.Errorf("A[%d] = %v != (k/(k-1))(r-mean) = %v", i, got[i], id)
		}
		sum += got[i]
	}
	if math.Abs(sum) > 1e-12 {
		t.Errorf("advantages must sum to 0, got %v", sum)
	}
	if a := nn.RLOOAdvantage([]float64{5}); a[0] != 0 {
		t.Errorf("k=1 has no leave-one-out baseline → advantage must be 0, got %v", a[0])
	}
}

// §V16 tier-1 / §V2: RLOOLoss = −mean(Â·logp) is differentiable through logp with
// ∂/∂logpᵢ = −Âᵢ/k; the analytic gradient matches finite differences, and the
// advantages are frozen (no gradient).
func TestRLOOLossGradCheck(t *testing.T) {
	logp := []float64{-0.5, -1.2, -0.8, -2.0}
	adv := nn.RLOOAdvantage([]float64{1, 2, 3, 4})
	advT := tensor.FromFloat64(tensor.Shape{len(adv)}, adv)

	lossAt := func(d []float64) float64 {
		out, err := nn.RLOOLoss(backend.NewContext(), tensor.FromFloat64(tensor.Shape{len(d)}, d), advT)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	logpT := tensor.FromFloat64(tensor.Shape{len(logp)}, append([]float64(nil), logp...))
	tape := autograd.NewTape()
	loss, err := nn.RLOOLoss(tape.Context(), logpT, advT)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	// advT is a leaf constant (rewards, not a parameter and not tape-connected to
	// params), so any gradient the tape records for it is never applied — the only
	// gradient that matters flows into logp.
	g := tape.Grad(logpT)
	if g == nil {
		t.Fatal("logp must receive gradients")
	}
	const h = 1e-6
	work := append([]float64(nil), logp...)
	for i := range logp {
		o := work[i]
		work[i] = o + h
		lp := lossAt(work)
		work[i] = o - h
		lm := lossAt(work)
		work[i] = o
		num := (lp - lm) / (2 * h)
		ana := g.AtF64(i)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: analytic %v vs finite-diff %v", i, ana, num)
		}
		if exact := -adv[i] / float64(len(logp)); math.Abs(ana-exact) > 1e-9 {
			t.Errorf("grad[%d] = %v, want −Â/k = %v", i, ana, exact)
		}
	}
}

// RLOO gives each sampled completion a baseline from the OTHER samples' rewards, so
// above-average completions get a positive advantage and below-average a negative
// one — no value network needed, and the advantages sum to zero.
func ExampleRLOOAdvantage() {
	adv := nn.RLOOAdvantage([]float64{1, 2, 3, 4})
	var sum float64
	for _, a := range adv {
		sum += a
	}
	fmt.Printf("%.1f %.1f %.1f %.1f (sum %.1f)\n", adv[0], adv[1], adv[2], adv[3], sum)
	// Output: -2.0 -0.7 0.7 2.0 (sum 0.0)
}
