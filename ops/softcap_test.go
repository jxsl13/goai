package ops_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/ops"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// §V16 tier-1: y = cap·tanh(x/cap) elementwise, matching the Gemma-2 formula.
func TestSoftCapFormula(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0, 1.5, -3, 100})
	const cap = 30.0
	y, err := ops.SoftCap(x, cap)
	if err != nil {
		t.Fatalf("SoftCap: %v", err)
	}
	for i := range 2 {
		for j := range 2 {
			v := x.AtF64(i, j)
			want := cap * math.Tanh(v/cap)
			if got := y.AtF64(i, j); math.Abs(got-want) > 1e-12 {
				t.Errorf("y[%d,%d] = %.15g, want %.15g", i, j, got, want)
			}
		}
	}
}

// §V16 tier-1: limiting behavior — near-identity for |x| ≪ cap, saturates to ±cap.
func TestSoftCapLimits(t *testing.T) {
	const cap = 50.0
	// small x ⇒ ≈ x
	small, _ := ops.SoftCap(tensor.FromFloat64(tensor.Shape{1}, []float64{0.5}), cap)
	if math.Abs(small.AtF64(0)-0.5) > 1e-3 {
		t.Errorf("small: %.6g, want ≈0.5", small.AtF64(0))
	}
	// large x ⇒ saturates to ±cap (tanh → ±1, exactly at f64 precision for huge x)
	big, _ := ops.SoftCap(tensor.FromFloat64(tensor.Shape{2}, []float64{1e6, -1e6}), cap)
	if v := big.AtF64(0); math.Abs(v-cap) > 1e-6 {
		t.Errorf("large +: %.10g, want ≈%g", v, cap)
	}
	if v := big.AtF64(1); math.Abs(v+cap) > 1e-6 {
		t.Errorf("large −: %.10g, want ≈%g", v, -cap)
	}
	// a moderately large value stays strictly inside the interval
	mid, _ := ops.SoftCap(tensor.FromFloat64(tensor.Shape{1}, []float64{200}), cap)
	if v := mid.AtF64(0); v >= cap {
		t.Errorf("mid: %.10g, want < %g", v, cap)
	}
}

func TestSoftCapErrors(t *testing.T) {
	if _, err := ops.SoftCap(tensor.FromFloat64(tensor.Shape{1}, []float64{1}), 0); err == nil {
		t.Error("cap=0: want error")
	}
	if _, err := ops.SoftCap(tensor.FromFloat64(tensor.Shape{1}, []float64{1}), -5); err == nil {
		t.Error("cap<0: want error")
	}
}

// §V2: the VJP g·(1−(y/cap)²) matches central finite differences.
func TestSoftCapGradcheck(t *testing.T) {
	const cap = 4.0
	xv := []float64{-6, -1.2, 0, 0.3, 2.5, 8}
	x := tensor.FromFloat64(tensor.Shape{len(xv)}, xv)
	w := tensor.FromFloat64(tensor.Shape{len(xv)}, []float64{0.7, -1.1, 0.4, 1.3, -0.6, 0.9})

	tape := autograd.NewTape()
	y, _ := backend.Execute(tape.Context(), backend.OpSoftCap, []*tensor.Tensor{x}, backend.SoftCapAttrs{Cap: cap})
	p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{p[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	const h = 1e-6
	lossOf := func() float64 {
		yy, _ := backend.Execute(backend.NewContext(), backend.OpSoftCap, []*tensor.Tensor{x}, backend.SoftCapAttrs{Cap: cap})
		var s float64
		for i := range xv {
			s += w.AtF64(i) * yy[0].AtF64(i)
		}
		return s
	}
	for i := range xv {
		orig := x.AtF64(i)
		x.SetF64(orig+h, i)
		lp := lossOf()
		x.SetF64(orig-h, i)
		lm := lossOf()
		x.SetF64(orig, i)
		num := (lp - lm) / (2 * h)
		if ana := g.AtF64(i); math.Abs(num-ana) > 1e-5*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", i, num, ana)
		}
	}
}

func ExampleSoftCap() {
	// Gemma-2's final-logit cap is 30: a huge logit saturates just under 30.
	y, _ := ops.SoftCap(tensor.FromFloat64(tensor.Shape{2}, []float64{1000, 2}), 30)
	fmt.Printf("%.4f %.4f\n", y.AtF64(0), y.AtF64(1))
	// Output:
	// 30.0000 1.9970
}
