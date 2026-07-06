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

type loraGolden struct {
	M, In, Out, R int
	Alpha         float64   `json:"alpha"`
	X             []float64 `json:"x"`
	W             []float64 `json:"w"`
	A             []float64 `json:"a"`
	B             []float64 `json:"b"`
	Y             []float64 `json:"y"`
}

func loadLoRA(t *testing.T) loraGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/lora.json")
	if err != nil {
		t.Fatalf("read golden (run make golden): %v", err)
	}
	var g loraGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V1: LoRA forward y = x·W + (α/r)(x·A)·B matches torch at f64 rtol 1e-12.
func TestLoRAParity(t *testing.T) {
	g := loadLoRA(t)
	x := tensor.FromFloat64(tensor.Shape{g.M, g.In}, g.X)
	l := &nn.LoRALinear{
		W:     tensor.FromFloat64(tensor.Shape{g.In, g.Out}, g.W),
		A:     tensor.FromFloat64(tensor.Shape{g.In, g.R}, g.A),
		B:     tensor.FromFloat64(tensor.Shape{g.R, g.Out}, g.B),
		R:     g.R,
		Alpha: g.Alpha,
	}
	y, err := l.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range g.Y {
		idx := tensor.Unravel(i, y.Shape())
		if v := y.AtF64(idx...); math.Abs(v-g.Y[i]) > 1e-12*math.Max(1, math.Abs(g.Y[i])) {
			t.Errorf("y[%d]: got %.17g want %.17g", i, v, g.Y[i])
		}
	}
}

// §R27: with B=0 (fresh adapter) LoRA is a no-op — output equals the frozen base.
func TestLoRAZeroInit(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	l, err := nn.NewLoRA(w, 2, 8, 1)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 0, -1, 2, 0.5, 1, -2, 3})
	base, _ := backend.Execute(backend.NewContext(), backend.OpMatMul, []*tensor.Tensor{x, w}, nil)
	y, err := l.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for i := range 6 {
		idx := tensor.Unravel(i, y.Shape())
		if math.Abs(y.AtF64(idx...)-base[0].AtF64(idx...)) > 1e-14 {
			t.Errorf("B=0 LoRA must equal base at [%d]", i)
		}
	}
	if len(l.Params()) != 2 { // A, B only — W frozen
		t.Errorf("Params = %d, want 2 (A,B)", len(l.Params()))
	}
}

// LoRA trains its adapters while the base W stays frozen.
func TestLoRATrainsFrozenBase(t *testing.T) {
	w := tensor.FromFloat64(tensor.Shape{4, 3}, []float64{0.5, -1, 2, 0.3, 1.1, -0.7, 0.9, 0.2, -1.5, 0.6, 1.3, -0.4})
	wSnapshot := make([]float64, 12)
	for i := range wSnapshot {
		wSnapshot[i] = w.AtF64(tensor.Unravel(i, w.Shape())...)
	}
	l, _ := nn.NewLoRA(w, 2, 8, 3)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, -1, 0.5, 2, 0.3, 1, -2, 0.7})
	target := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 0, -1, 0.5, 2, -0.5})
	opt := nn.NewAdam(l.Params(), 0.05)

	var first, last float64
	for step := range 100 {
		tape := autograd.NewTape()
		ctx := tape.Context()
		y, err := l.Forward(ctx, x)
		if err != nil {
			t.Fatal(err)
		}
		loss, err := nn.MSE(ctx, y, target)
		if err != nil {
			t.Fatal(err)
		}
		if step == 0 {
			first = loss.AtF64()
		}
		last = loss.AtF64()
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		// only adapters get gradients
		if tape.Grad(l.A) == nil || tape.Grad(l.B) == nil {
			t.Fatal("adapters must receive gradients")
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	if last >= first {
		t.Fatalf("LoRA did not train: loss %.4f → %.4f", first, last)
	}
	// base W must be byte-for-byte unchanged (frozen)
	for i := range wSnapshot {
		if w.AtF64(tensor.Unravel(i, w.Shape())...) != wSnapshot[i] {
			t.Fatalf("frozen base W changed at [%d]", i)
		}
	}
	t.Logf("LoRA fine-tuned: MSE %.4f → %.4f, base frozen", first, last)
}
