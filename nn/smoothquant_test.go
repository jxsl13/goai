package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func matmul(t *testing.T, a, b *tensor.Tensor) *tensor.Tensor {
	t.Helper()
	o, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{a, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return o[0]
}

func maxAbs(x *tensor.Tensor) float64 {
	var m float64
	for i := range x.Numel() {
		if a := math.Abs(x.AtF64(tensor.Unravel(i, x.Shape())...)); a > m {
			m = a
		}
	}
	return m
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2211.10438 §4 Eq.3): the smoothing transform
// is mathematically equivalent — X̂·Ŵ = X·W exactly — for every α.
func TestSmoothQuantEquivalence(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 24))
	const tokens, cin, cout = 6, 4, 3
	x, w := randMat(rng, tokens, cin), randMat(rng, cin, cout)
	// inject a per-channel activation outlier (the case SmoothQuant targets).
	for tk := 0; tk < tokens; tk++ {
		x.SetF64(x.AtF64(tk, 0)*40, tk, 0)
	}
	want := matmul(t, x, w)
	for _, alpha := range []float64{0, 0.5, 1} {
		xHat, wHat, _, err := nn.SmoothQuant(x, w, alpha)
		if err != nil {
			t.Fatalf("alpha=%g: %v", alpha, err)
		}
		got := matmul(t, xHat, wHat)
		for i := 0; i < tokens; i++ {
			for j := 0; j < cout; j++ {
				if math.Abs(got.AtF64(i, j)-want.AtF64(i, j)) > 1e-9*math.Max(1, math.Abs(want.AtF64(i, j))) {
					t.Errorf("alpha=%g X̂Ŵ[%d,%d]=%.10g != XW %.10g", alpha, i, j, got.AtF64(i, j), want.AtF64(i, j))
				}
			}
		}
	}
}

// §V16 tier-1 (Eq.4): s_j = max(|X_j|)^α / max(|W_j|)^(1−α).
func TestSmoothQuantScaleFormula(t *testing.T) {
	// actAbsMax=[4,1], weightAbsMax=[2,8], α=0.5 ⇒ s=[√4/√2, √1/√8]=[√2, 1/(2√2)].
	s := nn.SmoothQuantScale([]float64{4, 1}, []float64{2, 8}, 0.5)
	want := []float64{math.Sqrt(4) / math.Sqrt(2), math.Sqrt(1) / math.Sqrt(8)}
	for j := range want {
		if math.Abs(s[j]-want[j]) > 1e-12 {
			t.Errorf("s[%d]=%g want %g", j, s[j], want[j])
		}
	}
	// degenerate all-zero channel ⇒ s=1 (no-op, no divide-by-zero).
	if z := nn.SmoothQuantScale([]float64{0}, []float64{5}, 0.5); z[0] != 1 {
		t.Errorf("zero-activation channel s=%g want 1", z[0])
	}
}

// §V2 (defining goal): a per-channel activation outlier is migrated into the
// weights, so the smoothed activation's dynamic range shrinks.
func TestSmoothQuantMigratesOutliers(t *testing.T) {
	// channel 0 is a ×100 outlier; weights balanced ⇒ s_0=10, s_1=1 (α=0.5).
	x := toT([][]float64{{100, 1}, {80, -0.5}})
	w := toT([][]float64{{1, -1}, {0.5, 1}})
	xHat, wHat, scale, err := nn.SmoothQuant(x, w, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(scale[0]-10) > 1e-9 || math.Abs(scale[1]-1) > 1e-9 {
		t.Errorf("scale=%v want [10 1]", scale)
	}
	// the outlier activation channel is divided down: global |X̂| ≪ |X|.
	if maxAbs(xHat) >= maxAbs(x) {
		t.Errorf("smoothed act range %g not reduced from %g", maxAbs(xHat), maxAbs(x))
	}
	if math.Abs(maxAbs(xHat)-10) > 1e-9 { // 100/10
		t.Errorf("smoothed act max=%g want 10", maxAbs(xHat))
	}
	// the difficulty moved into the weights (row 0 scaled up ×10).
	if maxAbs(wHat) <= maxAbs(w) {
		t.Errorf("weight range %g not increased from %g", maxAbs(wHat), maxAbs(w))
	}
}

func TestSmoothQuantErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, _, _, err := nn.SmoothQuant(r(2, 3, 1), r(3, 2), 0.5); err == nil {
		t.Error("rank-3: want error")
	}
	if _, _, _, err := nn.SmoothQuant(r(2, 4), r(3, 2), 0.5); err == nil {
		t.Error("C_in mismatch: want error")
	}
	if _, _, _, err := nn.SmoothQuant(r(2, 3), r(3, 2), 1.5); err == nil {
		t.Error("alpha>1: want error")
	}
}

func ExampleSmoothQuant() {
	// one activation outlier channel (×100); smoothing migrates it into the weights
	// while keeping the product X·W unchanged.
	x := toT([][]float64{{100, 1}})
	w := toT([][]float64{{1, 0}, {0, 1}})
	xHat, _, scale, _ := nn.SmoothQuant(x, w, 0.5)
	fmt.Printf("s=[%.0f %.0f] xHat=[%.0f %.0f]\n", scale[0], scale[1], xHat.AtF64(0, 0), xHat.AtF64(0, 1))
	// Output:
	// s=[10 1] xHat=[10 1]
}
