package nn_test

import (
	"encoding/json"
	"fmt"
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

type doraGolden struct {
	In    int       `json:"in"`
	Out   int       `json:"out"`
	R     int       `json:"r"`
	Alpha float64   `json:"alpha"`
	W     []float64 `json:"W"`
	A     []float64 `json:"A"`
	B     []float64 `json:"B"`
	M     []float64 `json:"m"`
	X     []float64 `json:"x"`
	V     []float64 `json:"V"`
	Wp    []float64 `json:"Wp"`
	Y     []float64 `json:"y"`
	MInit []float64 `json:"m_init"`
}

func loadDoRA(t *testing.T) doraGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/dora.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g doraGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func closeVec(t *testing.T, name string, got *tensor.Tensor, want []float64, tol float64) {
	t.Helper()
	for k := range got.Numel() {
		idx := tensor.Unravel(k, got.Shape())
		if g, w := got.AtF64(idx...), want[k]; math.Abs(g-w) > tol*math.Max(1, math.Abs(w)) {
			t.Errorf("%s[%d]: got %.12g want %.12g", name, k, g, w)
		}
	}
}

// §V16 tier-1: the fused DoRA weight decomposition W' = m·V/‖V‖_col matches the
// independent numpy reference at f64 1e-12.
func TestDoRAWeightParity(t *testing.T) {
	g := loadDoRA(t)
	v := tensor.FromFloat64(tensor.Shape{g.In, g.Out}, g.V)
	m := tensor.FromFloat64(tensor.Shape{g.Out}, g.M)
	out, err := backend.Execute(backend.NewContext(), backend.OpDoRAWeight, []*tensor.Tensor{v, m}, nil)
	if err != nil {
		t.Fatal(err)
	}
	closeVec(t, "Wp", out[0], g.Wp, 1e-12)
}

// §V2: central finite differences of Σ(W') w.r.t. V and the magnitude m match the
// analytic (through-norm) VJP.
func TestDoRAWeightGradCheck(t *testing.T) {
	g := loadDoRA(t)
	vData := append([]float64(nil), g.V...)
	mData := append([]float64(nil), g.M...)

	sumWp := func(vd, md []float64) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpDoRAWeight,
			[]*tensor.Tensor{tensor.FromFloat64(tensor.Shape{g.In, g.Out}, vd), tensor.FromFloat64(tensor.Shape{g.Out}, md)}, nil)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for k := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(k, out[0].Shape())...)
		}
		return s
	}

	vT := tensor.FromFloat64(tensor.Shape{g.In, g.Out}, vData)
	mT := tensor.FromFloat64(tensor.Shape{g.Out}, mData)
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), backend.OpDoRAWeight, []*tensor.Tensor{vT, mT}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	check := func(name string, data []float64, grad *tensor.Tensor, shape tensor.Shape, vd, md []float64, isV bool) {
		for k := range len(data) {
			o := data[k]
			data[k] = o + h
			lp := sumWp(vd, md)
			data[k] = o - h
			lm := sumWp(vd, md)
			data[k] = o
			idx := tensor.Unravel(k, shape)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad[%d]: numeric %.8g vs analytic %.8g", name, k, num, ana)
			}
		}
	}
	check("V", vData, tape.Grad(vT), tensor.Shape{g.In, g.Out}, vData, mData, true)
	check("m", mData, tape.Grad(mT), tensor.Shape{g.Out}, vData, mData, false)
}

// §V16 tier-1: the full DoRALinear forward (V = W+(α/r)AB → W' = m·V/‖V‖ → x·W')
// matches the independent numpy reference at f64 1e-12.
func TestDoRALinearForward(t *testing.T) {
	g := loadDoRA(t)
	d := &nn.DoRALinear{
		W:     tensor.FromFloat64(tensor.Shape{g.In, g.Out}, g.W),
		A:     tensor.FromFloat64(tensor.Shape{g.In, g.R}, g.A),
		B:     tensor.FromFloat64(tensor.Shape{g.R, g.Out}, g.B),
		M:     tensor.FromFloat64(tensor.Shape{g.Out}, g.M),
		R:     g.R,
		Alpha: g.Alpha,
	}
	y, err := d.Forward(backend.NewContext(), tensor.FromFloat64(tensor.Shape{2, g.In}, g.X))
	if err != nil {
		t.Fatal(err)
	}
	closeVec(t, "y", y, g.Y, 1e-12)
}

// At init the adapter is an exact no-op: B=0 and m=‖W‖_col ⇒ W'=W, so the DoRA
// layer reproduces the frozen base x·W bit-for-bit — training starts from the
// pretrained behaviour.
func TestDoRAInitIsBaseNoOp(t *testing.T) {
	g := loadDoRA(t)
	w := tensor.FromFloat64(tensor.Shape{g.In, g.Out}, g.W)
	d, err := nn.NewDoRA(w, g.R, g.Alpha, 7)
	if err != nil {
		t.Fatal(err)
	}
	closeVec(t, "m_init", d.M, g.MInit, 1e-12) // magnitude initialized to ‖W‖_col

	x := tensor.FromFloat64(tensor.Shape{2, g.In}, g.X)
	got, err := d.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	base, err := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range got.Numel() {
		idx := tensor.Unravel(k, got.Shape())
		if a, b := got.AtF64(idx...), base[0].AtF64(idx...); math.Abs(a-b) > 1e-12*math.Max(1, math.Abs(b)) {
			t.Errorf("init DoRA[%d] = %.12g, want base x·W %.12g", k, a, b)
		}
	}
}

// DoRA (Weight-Decomposed LoRA) fine-tunes a frozen weight by learning a per-neuron
// magnitude and a low-rank direction separately. At initialization it reproduces
// the base layer exactly, so the first forward pass equals the pretrained output —
// here shown by the adapter output matching x·W before any training.
func ExampleDoRALinear() {
	w := tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6})
	d, _ := nn.NewDoRA(w, 2 /*rank*/, 4 /*alpha*/, 1)
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{1, 0, -1})

	y, _ := d.Forward(backend.NewContext(), x)
	fmt.Printf("%.1f %.1f\n", y.AtF64(0, 0), y.AtF64(0, 1)) // == x·W row: [1-5, 2-6]
	// Output: -4.0 -4.0
}
