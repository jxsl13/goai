package nn_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func loadLabelSmooth(t *testing.T) (eps float64, shape []int, logits, targets []float64, loss float64) {
	t.Helper()
	raw, err := os.ReadFile("testdata/labelsmooth.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g struct {
		Epsilon float64   `json:"epsilon"`
		Shape   []int     `json:"shape"`
		Logits  []float64 `json:"logits"`
		Targets []float64 `json:"targets"`
		Loss    float64   `json:"loss"`
	}
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g.Epsilon, g.Shape, g.Logits, g.Targets, g.Loss
}

// §V1 + §V16: label-smoothed CE matches an INDEPENDENT numpy −Σ q'·log-softmax
// with q'=(1−ε)·onehot+ε/K (Szegedy 2016 / Vaswani 2017) at f64 rtol 1e-12.
func TestLabelSmoothParity(t *testing.T) {
	eps, shape, logits, targets, want := loadLabelSmooth(t)
	z := tensor.FromFloat64(tensor.Shape{shape[0], shape[1]}, logits)
	tg := tensor.FromFloat64(tensor.Shape{shape[0]}, targets)
	got, err := nn.CrossEntropySmooth(backend.NewContext(), z, tg, eps)
	if err != nil {
		t.Fatal(err)
	}
	if v := got.AtF64(); math.Abs(v-want) > 1e-12*math.Max(1, math.Abs(want)) {
		t.Errorf("label-smooth CE = %.17g, want %.17g", v, want)
	}
}

// ε=0 recovers ordinary hard-label cross-entropy exactly (byte-identical loss).
func TestLabelSmoothEps0EqualsCE(t *testing.T) {
	z := tensor.FromFloat64(tensor.Shape{3, 3}, []float64{2, 1, 0.1, 0.5, 2.5, -1, -3, 0, 4})
	tg := tensor.FromFloat64(tensor.Shape{3}, []float64{0, 1, 2})
	hard, _ := nn.CrossEntropy(backend.NewContext(), z, tg)
	smooth0, _ := nn.CrossEntropySmooth(backend.NewContext(), z, tg, 0)
	if hard.AtF64() != smooth0.AtF64() {
		t.Errorf("ε=0 must equal hard CE: %.17g vs %.17g", smooth0.AtF64(), hard.AtF64())
	}
}

// §V2: central finite differences of the label-smoothed loss w.r.t. the logits
// match the analytic VJP (gradient = softmax − q').
func TestLabelSmoothGradCheck(t *testing.T) {
	eps := 0.1
	data := []float64{2, 1, 0.1, 0.5, 2.5, -1, -3, 0, 4}
	m, c := 3, 3
	tg := tensor.FromFloat64(tensor.Shape{m}, []float64{0, 1, 2})

	lossAt := func(d []float64) float64 {
		out, err := nn.CrossEntropySmooth(backend.NewContext(), tensor.FromFloat64(tensor.Shape{m, c}, d), tg, eps)
		if err != nil {
			t.Fatal(err)
		}
		return out.AtF64()
	}

	zT := tensor.FromFloat64(tensor.Shape{m, c}, data)
	tape := autograd.NewTape()
	loss, err := nn.CrossEntropySmooth(tape.Context(), zT, tg, eps)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gz := tape.Grad(zT)
	const h = 1e-6
	work := append([]float64(nil), data...)
	for k := range data {
		o := work[k]
		work[k] = o + h
		lp := lossAt(work)
		work[k] = o - h
		lm := lossAt(work)
		work[k] = o
		num := (lp - lm) / (2 * h)
		ana := gz.AtF64(k/c, k%c)
		if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad[%d]: numeric %.8g vs analytic %.8g", k, num, ana)
		}
	}
}

// The soft-target CE gradient sums to zero per row: Σ_k (softmax_k − q'_k) = 1−1 = 0
// (both distributions are normalized). A logit-shift invariant of the loss.
func TestLabelSmoothGradSumsZero(t *testing.T) {
	z := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 2, -1, 0.5, 0.3, -0.7, 2.0, 1.1})
	tg := tensor.FromFloat64(tensor.Shape{2}, []float64{2, 0})
	tape := autograd.NewTape()
	loss, err := nn.CrossEntropySmooth(tape.Context(), z, tg, 0.2)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gz := tape.Grad(z)
	for i := range 2 {
		var s float64
		for j := range 4 {
			s += gz.AtF64(i, j)
		}
		if math.Abs(s) > 1e-12 {
			t.Errorf("row %d gradient sum = %g, want 0 (normalized soft target)", i, s)
		}
	}
}
