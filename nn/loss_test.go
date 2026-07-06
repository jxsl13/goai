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

type lossGolden struct {
	Losses struct {
		MSE struct {
			Pred   []float64 `json:"pred"`
			Target []float64 `json:"target"`
			Shape  []int     `json:"shape"`
			F64    float64   `json:"f64"`
		} `json:"mse"`
		CE struct {
			Logits  []float64 `json:"logits"`
			Shape   []int     `json:"shape"`
			Targets []float64 `json:"targets"`
			F64     float64   `json:"f64"`
		} `json:"ce"`
	} `json:"losses"`
}

func loadLossGolden(t *testing.T) lossGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/losses.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g lossGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V1: MSE and CE match the Python golden at f64 rtol 1e-12.
func TestLossGoldenParity(t *testing.T) {
	g := loadLossGolden(t)
	ctx := backend.NewContext()

	m := g.Losses.MSE
	mse, err := nn.MSE(ctx,
		tensor.FromFloat64(tensor.Shape(m.Shape), m.Pred),
		tensor.FromFloat64(tensor.Shape(m.Shape), m.Target))
	if err != nil {
		t.Fatal(err)
	}
	if got := mse.AtF64(); math.Abs(got-m.F64) > 1e-12*math.Max(1, math.Abs(m.F64)) {
		t.Errorf("mse: got %.17g want %.17g", got, m.F64)
	}

	c := g.Losses.CE
	ce, err := nn.CrossEntropy(ctx,
		tensor.FromFloat64(tensor.Shape(c.Shape), c.Logits),
		tensor.FromFloat64(tensor.Shape{len(c.Targets)}, c.Targets))
	if err != nil {
		t.Fatal(err)
	}
	if got := ce.AtF64(); math.Abs(got-c.F64) > 1e-12*math.Max(1, math.Abs(c.F64)) {
		t.Errorf("ce: got %.17g want %.17g", got, c.F64)
	}
}

// §V12: the max-shift keeps CE finite and shift-invariant for huge logits
// (softmax(z) == softmax(z+K)).
func TestCrossEntropyStability(t *testing.T) {
	ctx := backend.NewContext()
	base := []float64{2, 1, 0.1, 0.5, 2.5, -1}
	shifted := make([]float64, len(base))
	for i, v := range base {
		shifted[i] = v + 1000
	}
	tg := tensor.FromFloat64(tensor.Shape{2}, []float64{0, 1})

	l1, err := nn.CrossEntropy(ctx, tensor.FromFloat64(tensor.Shape{2, 3}, base), tg)
	if err != nil {
		t.Fatal(err)
	}
	l2, err := nn.CrossEntropy(ctx, tensor.FromFloat64(tensor.Shape{2, 3}, shifted), tg)
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(l2.AtF64()) || math.IsInf(l2.AtF64(), 0) {
		t.Fatalf("CE overflowed on shifted logits: %v", l2.AtF64())
	}
	if math.Abs(l1.AtF64()-l2.AtF64()) > 1e-9 {
		t.Errorf("CE not shift-invariant: %v vs %v", l1.AtF64(), l2.AtF64())
	}
}

// §V2: CE gradient (softmax−onehot)/b matches finite differences; targets get
// nil grad. MSE gradient flows through its composition.
func TestLossGradCheck(t *testing.T) {
	const h, rtol = 1e-6, 1e-5
	zd := []float64{1.2, -0.4, 0.7, 0.1, 2.2, -1.3}
	ty := []float64{2, 0}

	ceLoss := func(z []float64) float64 {
		var total float64
		for i := range 2 {
			row := z[i*3 : (i+1)*3]
			m := math.Inf(-1)
			for _, v := range row {
				m = math.Max(m, v)
			}
			var s float64
			for _, v := range row {
				s += math.Exp(v - m)
			}
			total += m + math.Log(s) - row[int(ty[i])]
		}
		return total / 2
	}

	tape := autograd.NewTape()
	ctx := tape.Context()
	z := tensor.FromFloat64(tensor.Shape{2, 3}, zd)
	tg := tensor.FromFloat64(tensor.Shape{2}, ty)
	loss, err := nn.CrossEntropy(ctx, z, tg)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	gz := tape.Grad(z)
	if gz == nil {
		t.Fatal("nil grad for logits")
	}
	if tape.Grad(tg) != nil {
		t.Error("targets must be non-differentiable (nil grad)")
	}
	for j := range zd {
		p := append([]float64(nil), zd...)
		m := append([]float64(nil), zd...)
		p[j] += h
		m[j] -= h
		numeric := (ceLoss(p) - ceLoss(m)) / (2 * h)
		analytic := gz.AtF64(tensor.Unravel(j, tensor.Shape{2, 3})...)
		if math.Abs(numeric-analytic) > rtol*math.Max(1, math.Abs(numeric)) {
			t.Errorf("ce grad[%d]: numeric %.10g vs analytic %.10g", j, numeric, analytic)
		}
	}

	// MSE composition grad: d/dp mean((p-t)²) = 2(p-t)/n
	tape2 := autograd.NewTape()
	ctx2 := tape2.Context()
	pd := []float64{0.5, -1.5, 2.5, 0}
	td := []float64{1, -1, 2, 0.5}
	pr := tensor.FromFloat64(tensor.Shape{4}, pd)
	tr := tensor.FromFloat64(tensor.Shape{4}, td)
	mloss, err := nn.MSE(ctx2, pr, tr)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape2.Backward(mloss); err != nil {
		t.Fatal(err)
	}
	gp := tape2.Grad(pr)
	for i := range pd {
		want := 2 * (pd[i] - td[i]) / 4
		if math.Abs(gp.AtF64(i)-want) > 1e-12 {
			t.Errorf("mse grad[%d] = %v, want %v", i, gp.AtF64(i), want)
		}
	}
}

func TestCrossEntropyErrors(t *testing.T) {
	ctx := backend.NewContext()
	z := tensor.New(tensor.F64, tensor.Shape{2, 3})
	if _, err := nn.CrossEntropy(ctx, z, tensor.FromFloat64(tensor.Shape{2}, []float64{0, 7})); err == nil {
		t.Error("out-of-range target must error")
	}
	if _, err := nn.CrossEntropy(ctx, z, tensor.New(tensor.F64, tensor.Shape{3})); err == nil {
		t.Error("target/batch mismatch must error")
	}
}

// Activation layers apply their op; Sequential chains Linear+ReLU end to end.
func TestActivationAndSequential(t *testing.T) {
	ctx := backend.NewContext()
	x := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{-1, 0.5, 2})
	r, err := nn.ReLU().Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float64{0, 0.5, 2} {
		if r.AtF64(0, i) != want {
			t.Errorf("relu[%d] = %v, want %v", i, r.AtF64(0, i), want)
		}
	}

	seq := nn.NewSequential(nn.NewLinear(tensor.F64, 3, 4, 1), nn.ReLU(), nn.NewLinear(tensor.F64, 4, 2, 2))
	y, err := seq.Forward(ctx, x)
	if err != nil {
		t.Fatal(err)
	}
	if !y.Shape().Equal(tensor.Shape{1, 2}) {
		t.Errorf("sequential out shape %v, want (1,2)", y.Shape())
	}
	if got := len(seq.Params()); got != 4 {
		t.Errorf("sequential params = %d, want 4 (2×W + 2×b)", got)
	}
}
