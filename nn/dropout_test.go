package nn_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// In eval mode (and at rate 0) dropout is the identity — the forward pass needs
// no rescaling (§R55, inverted dropout).
func TestDropoutEvalIsIdentity(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, -2, 3, 4, -5, 6})
	d := nn.NewDropout(0.5, 1)
	d.Eval()
	out, err := d.Forward(nil, x)
	if err != nil {
		t.Fatal(err)
	}
	if out != x {
		t.Error("eval-mode dropout must return the input unchanged")
	}
	d.Train()
	d.Rate = 0
	if out, _ := d.Forward(nil, x); out != x {
		t.Error("rate-0 dropout must be identity")
	}
}

// §R55 expectation preservation: with inverted dropout E[y_i] = x_i (kept units
// scaled by 1/(1−p) exactly cancel the drop probability). Monte-Carlo averaged.
func TestDropoutExpectationPreserved(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{4}, []float64{2.0, -1.0, 5.0, 0.5})
	const rate = 0.4
	d := nn.NewDropout(rate, 99)
	const N = 20000
	sum := make([]float64, 4)
	for range N {
		out, err := d.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 4 {
			sum[i] += out.AtF64(i)
		}
	}
	for i := range 4 {
		if mean := sum[i] / N; math.Abs(mean-x.AtF64(i)) > 0.05*math.Max(1, math.Abs(x.AtF64(i))) {
			t.Errorf("E[out[%d]] = %.4f, want x = %.4f", i, mean, x.AtF64(i))
		}
	}
}

// In training each output is either 0 (dropped) or x/(1−p) (kept, up-scaled), and
// the dropped fraction matches the rate.
func TestDropoutMaskValuesAndRate(t *testing.T) {
	const rate, n = 0.3, 4000
	x := tensor.FromFloat64(tensor.Shape{n}, filled(n, 3.0))
	scale := 1.0 / (1.0 - rate)
	d := nn.NewDropout(rate, 7)
	out, err := d.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	dropped := 0
	for i := range n {
		v := out.AtF64(i)
		switch {
		case v == 0:
			dropped++
		case math.Abs(v-3.0*scale) > 1e-9:
			t.Fatalf("kept unit %d = %.6f, want 0 or %.6f", i, v, 3.0*scale)
		}
	}
	if frac := float64(dropped) / n; math.Abs(frac-rate) > 0.03 {
		t.Errorf("dropped fraction %.3f, want ≈ rate %.2f", frac, rate)
	}
}

// §R55 backward: the gradient flows only through the kept units, scaled by the
// same 1/(1−p) mask — dropout is an elementwise multiply, so its VJP multiplies
// the upstream gradient by that mask.
func TestDropoutGradientThroughMask(t *testing.T) {
	const rate = 0.5
	x := tensor.FromFloat64(tensor.Shape{6}, []float64{1, 2, 3, 4, 5, 6})
	d := nn.NewDropout(rate, 3)
	tape := autograd.NewTape()
	out, err := d.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	// Backward seeds the output cotangent to 1, so grad(x)[i] = mask[i] (dropout is
	// an elementwise multiply by the mask).
	if err := tape.Backward(out); err != nil {
		t.Fatal(err)
	}
	gx := tape.Grad(x)
	scale := 1.0 / (1.0 - rate)
	for i := range 6 {
		g := gx.AtF64(i)
		outi := out.AtF64(i)
		if outi == 0 { // dropped → zero gradient
			if g != 0 {
				t.Errorf("dropped unit %d has grad %g, want 0", i, g)
			}
		} else if math.Abs(g-scale) > 1e-9 { // kept → gradient scaled by 1/(1−p)
			t.Errorf("kept unit %d grad %g, want %g", i, g, scale)
		}
	}
}

func filled(n int, v float64) []float64 {
	d := make([]float64, n)
	for i := range d {
		d[i] = v
	}
	return d
}
