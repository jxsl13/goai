package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// lambRefRun is an INDEPENDENT scalar reimplementation of LAMB (Algorithm 2) over a fixed grad
// sequence for a single parameter tensor — the differential reference for TestLAMBParity.
func lambRefRun(theta0 []float64, gradSeq [][]float64, lr, b1, b2, eps, wd float64) []float64 {
	n := len(theta0)
	theta := append([]float64(nil), theta0...)
	m := make([]float64, n)
	v := make([]float64, n)
	for t, g := range gradSeq {
		c1 := 1 - math.Pow(b1, float64(t+1))
		c2 := 1 - math.Pow(b2, float64(t+1))
		u := make([]float64, n)
		var wn2, un2 float64
		for i := range theta {
			m[i] = b1*m[i] + (1-b1)*g[i]
			v[i] = b2*v[i] + (1-b2)*g[i]*g[i]
			mh := m[i] / c1
			vh := v[i] / c2
			u[i] = mh/(math.Sqrt(vh)+eps) + wd*theta[i]
			wn2 += theta[i] * theta[i]
			un2 += u[i] * u[i]
		}
		tr := 1.0
		if wn2 > 0 && un2 > 0 {
			tr = math.Sqrt(wn2) / math.Sqrt(un2)
		}
		for i := range theta {
			theta[i] -= lr * tr * u[i]
		}
	}
	return theta
}

// §V16 tier-1 (Algorithm 2): the optimizer matches the independent reference over a multi-step
// trajectory (with bias correction and decoupled weight decay).
func TestLAMBParity(t *testing.T) {
	theta0 := []float64{1, -2, 0.5, 3}
	gradSeq := [][]float64{
		{0.1, -0.2, 0.3, -0.05},
		{-0.4, 0.1, 0.2, 0.6},
		{0.05, 0.05, -0.5, -0.1},
	}
	const lr, b1, b2, eps, wd = 0.1, 0.9, 0.999, 1e-6, 0.01
	want := lambRefRun(theta0, gradSeq, lr, b1, b2, eps, wd)

	p := tensor.FromFloat64(tensor.Shape{len(theta0)}, append([]float64(nil), theta0...))
	opt := nn.NewLAMB([]*tensor.Tensor{p}, lr, nn.WithLAMBBetas(b1, b2), nn.WithLAMBEps(eps), nn.WithLAMBWeightDecay(wd))
	for _, g := range gradSeq {
		gt := tensor.FromFloat64(tensor.Shape{len(g)}, g)
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return gt }); err != nil {
			t.Fatal(err)
		}
	}
	for i := range want {
		if math.Abs(p.AtF64(i)-want[i]) > 1e-12 {
			t.Errorf("θ[%d]=%.12g, want %.12g", i, p.AtF64(i), want[i])
		}
	}
}

// §V16 tier-1 (the defining trust-ratio property): with no weight decay the per-layer update
// norm is EXACTLY lr·‖θ‖ — the trust ratio ‖θ‖/‖u‖ cancels ‖u‖, so the step size is tied to
// the weight norm and independent of the raw gradient magnitude. This is LAMB's whole point.
func TestLAMBTrustRatioProperty(t *testing.T) {
	for _, scale := range []float64{1, 100, 0.01} { // gradient magnitude must NOT change ‖Δθ‖
		start := []float64{3, -4, 12} // ‖θ‖ = 13
		p := tensor.FromFloat64(tensor.Shape{3}, append([]float64(nil), start...))
		opt := nn.NewLAMB([]*tensor.Tensor{p}, 0.1) // wd=0
		g := tensor.FromFloat64(tensor.Shape{3}, []float64{scale, -2 * scale, 0.5 * scale})
		if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
			t.Fatal(err)
		}
		var d2 float64
		for i, s := range start {
			d := p.AtF64(i) - s
			d2 += d * d
		}
		wnorm := 13.0
		if want := 0.1 * wnorm; math.Abs(math.Sqrt(d2)-want) > 1e-9 {
			t.Errorf("scale %g: ‖Δθ‖=%.10g, want lr·‖θ‖=%.10g", scale, math.Sqrt(d2), want)
		}
	}
}

// §V-PROP: LAMB decreases a least-squares loss over successive steps.
func TestLAMBTrainingDecreasesLoss(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4, 2}, []float64{1, 2, -1, 0.5, 3, -2, 0, 1})
	yt := tensor.FromFloat64(tensor.Shape{4, 1}, []float64{3, -0.5, 1, 2})
	l := nn.NewLinear(tensor.F64, 2, 1, 11)
	opt := nn.NewLAMB(l.Params(), 0.1)
	var prev float64
	for step := range 6 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		pred, err := l.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.MSE(ctx, pred, yt)
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64()
		if step > 0 && lv >= prev {
			t.Fatalf("loss did not decrease at step %d (%v → %v)", step, prev, lv)
		}
		prev = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
}

// §V2 (fallback): a whole zero-norm parameter tensor takes trust ratio 1 (θ←θ−lr·u), so a
// fresh zero-initialized weight still moves.
func TestLAMBZeroWeightFallback(t *testing.T) {
	p := tensor.New(tensor.F64, tensor.Shape{2}) // all zero ⇒ ‖θ‖=0
	opt := nn.NewLAMB([]*tensor.Tensor{p}, 0.1)  // wd=0 ⇒ u = m̂/(√v̂+ε)
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{0.5, -0.5})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g }); err != nil {
		t.Fatal(err)
	}
	// t=1: m̂=g, v̂=g² ⇒ u_i = g_i/(|g_i|+ε) ≈ ±1; trust=1 ⇒ Δθ_i = −lr·u_i ≈ ∓0.1.
	for i, gi := range []float64{0.5, -0.5} {
		want := -0.1 * (gi / (math.Abs(gi) + 1e-6))
		if math.Abs(p.AtF64(i)-want) > 1e-9 {
			t.Errorf("zero-weight step θ[%d]=%.10g, want %.10g", i, p.AtF64(i), want)
		}
	}
}

func TestLAMBEdge(t *testing.T) {
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})
	opt := nn.NewLAMB([]*tensor.Tensor{p}, 0.1)
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return nil }); err != nil {
		t.Fatal(err)
	}
	if p.AtF64(0) != 1 || p.AtF64(1) != 2 {
		t.Error("nil grad must leave params unchanged")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{3})
	if err := opt.Step(func(*tensor.Tensor) *tensor.Tensor { return bad }); err == nil {
		t.Error("shape mismatch must error")
	}
}

func ExampleLAMB() {
	// LAMB's per-layer trust ratio ties the update norm to the weight norm: ‖Δθ‖ = lr·‖θ‖.
	p := tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4}) // ‖θ‖ = 5
	opt := nn.NewLAMB([]*tensor.Tensor{p}, 0.1)
	g := tensor.FromFloat64(tensor.Shape{2}, []float64{1, 1})
	opt.Step(func(*tensor.Tensor) *tensor.Tensor { return g })
	fmt.Printf("%.4f\n", math.Hypot(p.AtF64(0)-3, p.AtF64(1)-4))
	// Output:
	// 0.5000
}
