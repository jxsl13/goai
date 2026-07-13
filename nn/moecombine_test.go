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

type moeCombineGolden struct {
	T, E, D int
	W       []float64   `json:"w"`
	Experts [][]float64 `json:"experts"`
	Out     []float64   `json:"out"`
}

func loadMoECombine(t *testing.T) moeCombineGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/moecombine.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g moeCombineGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func (g moeCombineGolden) inputs() []*tensor.Tensor {
	in := make([]*tensor.Tensor, 0, g.E+1)
	in = append(in, tensor.FromFloat64(tensor.Shape{g.T, g.E}, g.W))
	for i := range g.E {
		in = append(in, tensor.FromFloat64(tensor.Shape{g.T, g.D}, g.Experts[i]))
	}
	return in
}

// §V16 tier-1: the fused MoE combine matches the independent numpy reference
// (softmax gates → top-k mask → renormalize → mix) at f64 1e-12.
func TestMoECombineParity(t *testing.T) {
	g := loadMoECombine(t)
	out, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, g.inputs(), nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range out[0].Numel() {
		idx := tensor.Unravel(k, out[0].Shape())
		if got, want := out[0].AtF64(idx...), g.Out[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("moecombine[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// §V2: central finite differences of Σ(combine) w.r.t. the gate weights and every
// expert output match the analytic VJP (renorm gradient + per-expert weighting).
func TestMoECombineGradCheck(t *testing.T) {
	g := loadMoECombine(t)
	ins := g.inputs()

	sumOut := func(xs []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, xs, nil)
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
	out, err := backend.Execute(tape.Context(), backend.OpMoECombine, ins, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(out[0]); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	for n, in := range ins { // input 0 = weights, 1.. = experts
		grad := tape.Grad(in)
		if grad == nil {
			t.Fatalf("input %d received no gradient", n)
		}
		for k := range in.Numel() {
			idx := tensor.Unravel(k, in.Shape())
			o := in.AtF64(idx...)
			in.SetF64(o+h, idx...)
			lp := sumOut(ins)
			in.SetF64(o-h, idx...)
			lm := sumOut(ins)
			in.SetF64(o, idx...)
			num, ana := (lp-lm)/(2*h), grad.AtF64(idx...)
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("input %d grad[%d]: numeric %.8g vs analytic %.8g", n, k, num, ana)
			}
		}
	}
}

// The SparseMoE layer produces the right shapes and is differentiable end to end:
// a finite-difference check on a router weight and an expert weight matches the
// tape gradients (routing decision detached, everything else trainable).
func TestSparseMoEForwardAndGrad(t *testing.T) {
	moe := nn.NewSparseMoE(tensor.F64, 4 /*dim*/, 6 /*hidden*/, 3 /*experts*/, 2 /*top-k*/, 7)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{0.5, -1, 2, 0.3, -0.7, 1.1, 0.2, -1.5})

	loss := func() float64 {
		y, _, err := moe.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		if !y.Shape().Equal(tensor.Shape{2, 4}) {
			t.Fatalf("output shape %v, want (2, 4)", y.Shape())
		}
		var s float64
		for k := range y.Numel() {
			s += y.AtF64(tensor.Unravel(k, y.Shape())...)
		}
		return s
	}

	tape := autograd.NewTape()
	y, logits, err := moe.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !logits.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("gate logits shape %v, want (2, 3)", logits.Shape())
	}
	if err := tape.Backward(y); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	// one router weight and one gate weight of expert 0
	for _, p := range []*tensor.Tensor{moe.Router.W, moe.Experts[0].Wgate} {
		grad := tape.Grad(p)
		if grad == nil {
			t.Fatal("trainable parameter received no gradient")
		}
		idx := []int{0, 0}
		o := p.AtF64(idx...)
		p.SetF64(o+h, idx...)
		lp := loss()
		p.SetF64(o-h, idx...)
		lm := loss()
		p.SetF64(o, idx...)
		if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("param grad[0,0]: numeric %.8g vs analytic %.8g", num, ana)
		}
	}
}

// A SparseMoE layer routes each token through its top-k experts and mixes their
// SwiGLU outputs. Here 3 experts with top-2 routing map a [2,4] batch to a [2,4]
// output; the returned gate logits [2,3] feed the load-balancing auxiliary loss.
func ExampleSparseMoE() {
	moe := nn.NewSparseMoE(tensor.F64, 4 /*dim*/, 8 /*hidden*/, 3 /*experts*/, 2 /*top-k*/, 42)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 0, -1, 0.5, 0.2, 2, -0.3, 1})

	y, logits, _ := moe.Forward(backend.NewContext(), x)
	fmt.Println(y.Shape(), logits.Shape())
	// Output: (2, 4) (2, 3)
}
