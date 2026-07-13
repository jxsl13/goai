package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// reference recomputes the Schedule-Free y (train point) and x (eval average)
// trajectories independently from Algorithm 1 with a constant lr (⇒ c_t = 1/t).
func sfReference(theta []float64, grads [][]float64, lr, beta, wd, b2, eps float64, adam bool) (ys, xs [][]float64) {
	n := len(theta)
	z := append([]float64(nil), theta...)
	x := append([]float64(nil), theta...)
	v := make([]float64, n)
	for t := 1; t <= len(grads); t++ {
		yNow := make([]float64, n)
		for i := range z {
			yNow[i] = (1-beta)*z[i] + beta*x[i]
			g := grads[t-1][i] + wd*yNow[i]
			step := g
			if adam {
				v[i] = b2*v[i] + (1-b2)*g*g
				step = g / (math.Sqrt(v[i]/(1-math.Pow(b2, float64(t)))) + eps)
			}
			z[i] -= lr * step
		}
		ck := 1.0 / float64(t) // constant lr ⇒ γ²-weight ⇒ 1/t
		yNext := make([]float64, n)
		xNext := make([]float64, n)
		for i := range z {
			x[i] = (1-ck)*x[i] + ck*z[i]
			xNext[i] = x[i]
			yNext[i] = (1-beta)*z[i] + beta*x[i]
		}
		ys = append(ys, yNext)
		xs = append(xs, xNext)
	}
	return ys, xs
}

func runSF(t *testing.T, adam bool) {
	const lr, beta, wd, b2, eps = 0.05, 0.9, 0.1, 0.999, 1e-8
	theta := []float64{0.5, -1.0, 2.0}
	grads := [][]float64{
		{0.3, -0.2, 0.5}, {0.1, 0.4, -0.3}, {-0.5, 0.2, 0.1}, {0.2, -0.1, 0.4},
	}
	ys, xs := sfReference(theta, grads, lr, beta, wd, b2, eps, adam)

	p := tensor.FromFloat64(tensor.Shape{len(theta)}, append([]float64(nil), theta...))
	opts := []nn.ScheduleFreeOption{nn.WithScheduleFreeWeightDecay(wd)}
	var opt *nn.ScheduleFree
	if adam {
		opt = nn.NewScheduleFreeAdamW(p2s(p), lr, opts...)
	} else {
		opt = nn.NewScheduleFreeSGD(p2s(p), lr, opts...)
	}
	for s := range grads {
		gs := grads[s]
		if err := opt.Step(func(q *tensor.Tensor) *tensor.Tensor {
			if q != p {
				return nil
			}
			return tensor.FromFloat64(p.Shape(), gs)
		}); err != nil {
			t.Fatal(err)
		}
		for i := range p.Numel() { // param holds y_{s+1}
			if got := p.AtF64(i); math.Abs(got-ys[s][i]) > 1e-12*math.Max(1, math.Abs(ys[s][i])) {
				t.Errorf("step %d y[%d]: got %.14g want %.14g", s, i, got, ys[s][i])
			}
		}
	}
	// Eval must expose the averaged iterate x (the deployable point), Train restore y
	opt.Eval()
	last := len(grads) - 1
	for i := range p.Numel() {
		if got := p.AtF64(i); math.Abs(got-xs[last][i]) > 1e-12*math.Max(1, math.Abs(xs[last][i])) {
			t.Errorf("eval x[%d]: got %.14g want %.14g", i, got, xs[last][i])
		}
	}
	opt.Train()
	for i := range p.Numel() {
		if got := p.AtF64(i); math.Abs(got-ys[last][i]) > 1e-12*math.Max(1, math.Abs(ys[last][i])) {
			t.Errorf("train y[%d]: got %.14g want %.14g", i, got, ys[last][i])
		}
	}
}

func p2s(p *tensor.Tensor) []*tensor.Tensor { return []*tensor.Tensor{p} }

// §V16 tier-1: Schedule-Free SGD reproduces the paper's z/x/y recurrence (grad at y,
// z-step, γ²-weighted average, y write-back) step by step vs an independent
// recomputation, and Eval/Train expose x and y exactly.
func TestScheduleFreeSGDParity(t *testing.T) { runSF(t, false) }

// §V16 tier-1: the AdamW variant matches the same recurrence with the bias-corrected
// second-moment preconditioner on the z-step.
func TestScheduleFreeAdamWParity(t *testing.T) { runSF(t, true) }

// §V16 tier-1: before any step z=x=θ, so the interpolation point y=θ — the parameter
// starts exactly at its initial value and Eval also yields θ.
func TestScheduleFreeInitIsTheta(t *testing.T) {
	theta := []float64{1.5, -0.5}
	p := tensor.FromFloat64(tensor.Shape{2}, append([]float64(nil), theta...))
	opt := nn.NewScheduleFreeSGD([]*tensor.Tensor{p}, 0.1)
	opt.Eval()
	for i := range p.Numel() {
		if got := p.AtF64(i); math.Abs(got-theta[i]) > 1e-12 {
			t.Errorf("init eval x[%d]=%v, want θ=%v", i, got, theta[i])
		}
	}
}

// §V16 tier-1: gradients must be taken at y, so Step in eval mode is a usage error and
// is rejected rather than silently training on the wrong point.
func TestScheduleFreeStepInEvalModeErrors(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{1}, []float64{1})
	opt := nn.NewScheduleFreeSGD([]*tensor.Tensor{p}, 0.1)
	opt.Eval()
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return tensor.FromFloat64(tensor.Shape{1}, []float64{1}) }); err == nil {
		t.Error("Step in eval mode must error")
	}
}

// Schedule-Free replaces the learning-rate schedule with an online average. Starting
// at θ=1 with a constant gradient of 2, one SGD step with lr=0.1 moves the base
// iterate to z=0.8; the first average and interpolation both equal 0.8, so the
// parameter — the point you keep training from — becomes 0.80.
func ExampleScheduleFree() {
	w := tensor.FromFloat64(tensor.Shape{1}, []float64{1.0})
	opt := nn.NewScheduleFreeSGD([]*tensor.Tensor{w}, 0.1)
	_ = opt.Step(func(*tensor.Tensor) *tensor.Tensor {
		return tensor.FromFloat64(tensor.Shape{1}, []float64{2.0})
	})
	fmt.Printf("%.2f\n", w.AtF64(0))
	// Output: 0.80
}
