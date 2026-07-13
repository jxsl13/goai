package autograd_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type conv1dGolden struct {
	L, D, K int
	X       []float64 `json:"x"`
	W       []float64 `json:"w"`
	B       []float64 `json:"b"`
	Y       []float64 `json:"y"`
}

func loadConv1D(t *testing.T) conv1dGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/conv1d.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g conv1dGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

// §V16 tier-1: causal depthwise conv1d matches the independent numpy reference at
// f64 1e-12.
func TestConv1DParity(t *testing.T) {
	g := loadConv1D(t)
	x := tensor.FromFloat64(tensor.Shape{g.L, g.D}, g.X)
	w := tensor.FromFloat64(tensor.Shape{g.D, g.K}, g.W)
	b := tensor.FromFloat64(tensor.Shape{g.D}, g.B)
	out, err := backend.Execute(backend.NewContext(), backend.OpConv1D, []*tensor.Tensor{x, w, b}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range out[0].Numel() {
		idx := tensor.Unravel(k, out[0].Shape())
		if got, want := out[0].AtF64(idx...), g.Y[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("conv1d y[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// §V2: finite differences of Σ(conv1d) w.r.t. x, w and bias match the analytic VJP.
func TestConv1DGradCheck(t *testing.T) {
	g := loadConv1D(t)
	ins := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{g.L, g.D}, g.X),
		tensor.FromFloat64(tensor.Shape{g.D, g.K}, g.W),
		tensor.FromFloat64(tensor.Shape{g.D}, g.B),
	}
	gradCheckOp(t, backend.OpConv1D, nil, ins)
}

// §V2: softplus matches its known value and its gradient is the sigmoid.
func TestSoftplus(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{3}, []float64{0, 2, -3})
	out, err := backend.Execute(backend.NewContext(), backend.OpSoftplus, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{math.Log(2), math.Log(1 + math.Exp(2)), math.Log(1 + math.Exp(-3))}
	for i, w := range want {
		if math.Abs(out[0].AtF64(i)-w) > 1e-12 {
			t.Errorf("softplus[%d] = %.12g, want %.12g", i, out[0].AtF64(i), w)
		}
	}
	gradCheckOp(t, backend.OpSoftplus, nil, []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{4}, []float64{0.5, -1, 2, -0.3}),
	})
}

// gradCheckOp finite-difference-checks Σ(op(ins)) against the analytic VJP for
// every input.
func gradCheckOp(t *testing.T, op backend.Op, attrs backend.Attrs, ins []*tensor.Tensor) {
	t.Helper()
	sum := func(xs []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), op, xs, attrs)
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for k := range out[0].Numel() {
			s += out[0].AtF64(tensor.Unravel(k, out[0].Shape())...)
		}
		return s
	}
	tape := autograd.NewTape()
	out, err := backend.Execute(tape.Context(), op, ins, attrs)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	for n, in := range ins {
		grad := tape.Grad(in)
		if grad == nil {
			t.Fatalf("input %d got no gradient", n)
		}
		for k := range in.Numel() {
			idx := tensor.Unravel(k, in.Shape())
			o := in.AtF64(idx...)
			in.SetF64(o+h, idx...)
			lp := sum(ins)
			in.SetF64(o-h, idx...)
			lm := sum(ins)
			in.SetF64(o, idx...)
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("op %v input %d grad[%d]: numeric %.8g vs analytic %.8g", op, n, k, num, ana)
			}
		}
	}
}
