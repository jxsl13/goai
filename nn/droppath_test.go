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

func onesMat(r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := range t.Numel() {
		t.SetF64(1, tensor.Unravel(i, t.Shape())...)
	}
	return t
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1603.09382 + timm): the drop decision is
// PER-SAMPLE — within a row every element shares one mask value (0 or 1/keep), never
// a partial row (which would be element-wise dropout).
func TestDropPathPerSampleMask(t *testing.T) {
	dp := nn.NewDropPath(0.5, 7)
	x := onesMat(20, 5)
	out, err := dp.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	scale := 1 / 0.5
	sawDrop, sawKeep := false, false
	for b := 0; b < 20; b++ {
		v := out.AtF64(b, 0)
		for j := 0; j < 5; j++ { // whole row shares the value
			if out.AtF64(b, j) != v {
				t.Fatalf("row %d not uniform: %g vs %g (per-sample violated)", b, out.AtF64(b, j), v)
			}
		}
		switch {
		case v == 0:
			sawDrop = true
		case math.Abs(v-scale) < 1e-12:
			sawKeep = true
		default:
			t.Errorf("row %d value %g, want 0 or %g", b, v, scale)
		}
	}
	if !sawDrop || !sawKeep {
		t.Errorf("expected both dropped and kept rows over 20 samples (drop=%v keep=%v)", sawDrop, sawKeep)
	}
}

// §V16 tier-1: at eval (or Rate 0) DropPath is the identity.
func TestDropPathEvalIdentity(t *testing.T) {
	dp := nn.NewDropPath(0.4, 1)
	dp.Eval()
	x := onesMat(4, 3)
	out, _ := dp.Forward(backend.NewContext(), x)
	if out != x {
		t.Error("eval mode should return x unchanged")
	}
	dp.Train()
	if z := nn.NewDropPath(0, 1); true {
		o, _ := z.Forward(backend.NewContext(), x)
		if o != x {
			t.Error("rate 0 should return x unchanged")
		}
	}
}

// §V16 tier-1 (inverted dropout ⇒ E[out]=x): averaging many masks recovers the input.
func TestDropPathExpectationPreserved(t *testing.T) {
	dp := nn.NewDropPath(0.3, 99)
	x := onesMat(50, 1)
	const trials = 400
	var sum float64
	for r := 0; r < trials; r++ {
		out, _ := dp.Forward(backend.NewContext(), x)
		for b := 0; b < 50; b++ {
			sum += out.AtF64(b, 0)
		}
	}
	mean := sum / (trials * 50)
	if math.Abs(mean-1) > 0.03 { // E[out] = x = 1
		t.Errorf("mean output %g, want ≈1 (inverted dropout preserves expectation)", mean)
	}
}

// §V2: gradient flows per-sample — a kept row's gradient is 1/keep everywhere, a
// dropped row's is 0.
func TestDropPathGradient(t *testing.T) {
	dp := nn.NewDropPath(0.5, 3)
	x := onesMat(10, 4)
	tape := autograd.NewTape()
	out, err := dp.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	loss, _ := sumAll(tape.Context(), out)
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(x)
	scale := 1 / 0.5
	for b := 0; b < 10; b++ {
		gv := g.AtF64(b, 0)
		if gv != 0 && math.Abs(gv-scale) > 1e-12 {
			t.Errorf("row %d grad %g, want 0 or %g", b, gv, scale)
		}
		for j := 0; j < 4; j++ { // uniform within the row, matching the forward mask
			if g.AtF64(b, j) != gv {
				t.Errorf("row %d gradient not uniform", b)
			}
		}
	}
}

// §V16 tier-1 (linear decay, Eq.4): p_l = 1 − (l/L)(1 − pLast), pLast at the last
// block, monotone decreasing with depth.
func TestStochasticDepthSurvival(t *testing.T) {
	const L = 12
	if p := nn.StochasticDepthSurvival(L, L, 0.5); math.Abs(p-0.5) > 1e-12 {
		t.Errorf("survival(L,L)=%g want pLast 0.5", p)
	}
	if p := nn.StochasticDepthSurvival(0, L, 0.5); math.Abs(p-1) > 1e-12 {
		t.Errorf("survival(0,L)=%g want 1", p)
	}
	prev := 1.1
	for l := 1; l <= L; l++ {
		p := nn.StochasticDepthSurvival(l, L, 0.5)
		if p >= prev {
			t.Errorf("survival not decreasing at l=%d: %g ≥ %g", l, p, prev)
		}
		prev = p
	}
	if p := nn.StochasticDepthSurvival(6, L, 0.5); math.Abs(p-(1-0.5*6.0/12)) > 1e-12 { // 0.75
		t.Errorf("survival(6,12,0.5)=%g want 0.75", p)
	}
}

func TestDropPathErrors(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("rate ≥ 1: want panic")
		}
	}()
	nn.NewDropPath(1.0, 1)
}

func ExampleDropPath() {
	// two samples, drop rate 0.5 (keep-scale 2); with this seed sample 0 is kept and
	// up-scaled while sample 1 is dropped.
	dp := nn.NewDropPath(0.5, 2)
	x := onesMat(2, 3)
	out, _ := dp.Forward(backend.NewContext(), x)
	fmt.Printf("row0=%.0f row1=%.0f\n", out.AtF64(0, 0), out.AtF64(1, 0))
	// Output:
	// row0=2 row1=0
}
