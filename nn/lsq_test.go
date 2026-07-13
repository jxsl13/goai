package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func scalar1x1(v float64) *tensor.Tensor {
	s := tensor.New(tensor.F64, tensor.Shape{1, 1})
	s.SetF64(v, 0, 0)
	return s
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1902.08153 Eq.1/2): forward is
// round(clip(v/s, −Qn, Qp))·s.
func TestLSQForwardParity(t *testing.T) {
	v := toT([][]float64{{0.3, 1.7, 9.0}, {-2.2, -9.0, 3.2}})
	const s = 0.75
	const qn, qp = 8, 7
	got, err := nn.LSQQuantize(backend.NewContext(), v, scalar1x1(s), qn, qp, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 2 {
		for j := range 3 {
			x := math.Max(-8, math.Min(7, v.AtF64(i, j)/s))
			want := math.Round(x) * s
			if math.Abs(got.AtF64(i, j)-want) > 1e-12 {
				t.Errorf("v̂[%d,%d]=%g want %g", i, j, got.AtF64(i, j), want)
			}
		}
	}
}

// §V2: every output lies on the quantization grid {s·k : k ∈ [−Qn,Qp] integer}.
func TestLSQQuantizationGrid(t *testing.T) {
	v := toT([][]float64{{0.31, 1.72, 12.0}, {-2.29, -12.0, 3.24}})
	const s = 0.5
	const qn, qp = 8, 7
	got, _ := nn.LSQQuantize(backend.NewContext(), v, scalar1x1(s), qn, qp, 1)
	for i := range got.Numel() {
		c := tensor.Unravel(i, got.Shape())
		k := got.AtF64(c...) / s
		if math.Abs(k-math.Round(k)) > 1e-9 || k < -qn-1e-9 || k > qp+1e-9 {
			t.Errorf("v̂=%g ⇒ level %g not an integer in [%d,%d]", got.AtF64(c...), k, -qn, qp)
		}
	}
}

// §V2: values beyond the range saturate at ±Qp·s / −Qn·s.
func TestLSQClipSaturation(t *testing.T) {
	v := toT([][]float64{{100, -100}})
	const s = 0.5
	got, _ := nn.LSQQuantize(backend.NewContext(), v, scalar1x1(s), 8, 7, 1)
	if got.AtF64(0, 0) != 7*s || got.AtF64(0, 1) != -8*s {
		t.Errorf("saturation = [%g %g], want [%g %g]", got.AtF64(0, 0), got.AtF64(0, 1), 7*s, -8*s)
	}
}

// §V16 tier-1 (§2 bounds): signed weights (2^{b−1},2^{b−1}−1); unsigned (0,2^b−1).
func TestLSQBounds(t *testing.T) {
	cases := []struct {
		bits   int
		signed bool
		qn, qp int
	}{{8, true, 128, 127}, {8, false, 0, 255}, {4, true, 8, 7}, {2, false, 0, 3}}
	for _, c := range cases {
		if qn, qp := nn.LSQBounds(c.bits, c.signed); qn != c.qn || qp != c.qp {
			t.Errorf("LSQBounds(%d,%v)=(%d,%d), want (%d,%d)", c.bits, c.signed, qn, qp, c.qn, c.qp)
		}
	}
}

// §V16 tier-1 (§2.2 scale, §2.1 init): gradient scale g = 1/√(N·Qp); init s = 2·mean(|v|)/√Qp.
func TestLSQGradScaleAndInit(t *testing.T) {
	if g := nn.LSQGradScale(100, 7); math.Abs(g-1/math.Sqrt(700)) > 1e-12 {
		t.Errorf("LSQGradScale = %g, want %g", g, 1/math.Sqrt(700))
	}
	v := toT([][]float64{{1, -3}, {2, -2}}) // mean|v| = 2
	if s := nn.LSQInitStep(v, 7); math.Abs(s-2*2/math.Sqrt(7)) > 1e-12 {
		t.Errorf("LSQInitStep = %g, want %g", s, 2*2/math.Sqrt(7))
	}
}

// §V16 tier-1 (STE input gradient, Eq.3): ∂v̂/∂v = 1 inside the clip interval, 0 outside.
// finite differences are INVALID through round; we check the analytic (tape) gradient
// against the defined straight-through convention.
func TestLSQInputGradientSTE(t *testing.T) {
	v := toT([][]float64{{0.3, 1.7, 9.0}, {-2.2, -9.0, 3.2}}) // 9,−9 saturate at s=1 (qp=7,qn=8)
	const s = 1.0
	tape := autograd.NewTape()
	out, err := nn.LSQQuantize(tape.Context(), v, scalar1x1(s), 8, 7, 1)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, backend.ReduceAttrs{Axes: []int{0, 1}})
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	gv := tape.Grad(v)
	for i := range 2 {
		for j := range 3 {
			x := v.AtF64(i, j) / s
			want := 0.0
			if x > -8 && x < 7 {
				want = 1
			}
			if math.Abs(gv.AtF64(i, j)-want) > 1e-9 {
				t.Errorf("∂/∂v[%d,%d]=%g want %g (x=%g)", i, j, gv.AtF64(i, j), want, x)
			}
		}
	}
}

// §V16 tier-1 (LSQ step-size gradient, Eq.3 + §2.4 scale): ∂L/∂s equals
// gradScale·Σ_i [round(v_i/s)−v_i/s in range | −Qn below | Qp above]. Verified against the
// closed form (finite diff invalid through the STE round).
func TestLSQStepSizeGradient(t *testing.T) {
	v := toT([][]float64{{0.3, 1.7, 9.0}, {-2.2, -9.0, 3.2}})
	const s = 1.0
	const qn, qp = 8, 7
	const g = 0.5 // non-trivial gradient scale
	tape := autograd.NewTape()
	sp := scalar1x1(s)
	out, err := nn.LSQQuantize(tape.Context(), v, sp, qn, qp, g)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{out}, backend.ReduceAttrs{Axes: []int{0, 1}})
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	var sum float64
	for i := range 2 {
		for j := range 3 {
			x := v.AtF64(i, j) / s
			switch {
			case x <= -qn:
				sum += -qn
			case x >= qp:
				sum += qp
			default:
				sum += math.Round(x) - x
			}
		}
	}
	want := g * sum
	if got := tape.Grad(sp).AtF64(0, 0); math.Abs(got-want) > 1e-9 {
		t.Errorf("∂L/∂s = %g, want gradScale·Σ = %g", got, want)
	}
}

func TestLSQErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.LSQQuantize(backend.NewContext(), r(3, 4, 1), scalar1x1(1), 8, 7, 1); err == nil {
		t.Error("rank-3 v: want error")
	}
	if _, err := nn.LSQQuantize(backend.NewContext(), r(3, 4), r(2, 2), 8, 7, 1); err == nil {
		t.Error("non-scalar s: want error")
	}
	if _, err := nn.LSQQuantize(backend.NewContext(), r(3, 4), scalar1x1(1), 8, 0, 1); err == nil {
		t.Error("qp=0: want error")
	}
}

func ExampleLSQQuantize() {
	// 4-bit signed fake-quant (Qn=8,Qp=7) of a small tensor with step size s=0.5: each value
	// snaps to the nearest multiple of s within [−4, 3.5].
	qn, qp := nn.LSQBounds(4, true)
	v := toT([][]float64{{0.4, 1.1, 9.0}})
	out, _ := nn.LSQQuantize(backend.NewContext(), v, scalar1x1(0.5), qn, qp, 1)
	fmt.Printf("%.1f %.1f %.1f\n", out.AtF64(0, 0), out.AtF64(0, 1), out.AtF64(0, 2))
	// Output:
	// 0.5 1.0 3.5
}
