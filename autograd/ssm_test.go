package autograd_test

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

type ssmGolden struct {
	L, D, N int
	U       []float64 `json:"u"`
	Delta   []float64 `json:"delta"`
	A       []float64 `json:"A"`
	B       []float64 `json:"B"`
	C       []float64 `json:"C"`
	Dskip   []float64 `json:"Dskip"`
	Y       []float64 `json:"y"`
}

func loadSSM(t *testing.T) ssmGolden {
	t.Helper()
	raw, err := os.ReadFile("testdata/ssm.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g ssmGolden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func (g ssmGolden) inputs(withSkip bool) []*tensor.Tensor {
	in := []*tensor.Tensor{
		tensor.FromFloat64(tensor.Shape{g.L, g.D}, g.U),
		tensor.FromFloat64(tensor.Shape{g.L, g.D}, g.Delta),
		tensor.FromFloat64(tensor.Shape{g.D, g.N}, g.A),
		tensor.FromFloat64(tensor.Shape{g.L, g.N}, g.B),
		tensor.FromFloat64(tensor.Shape{g.L, g.N}, g.C),
	}
	if withSkip {
		in = append(in, tensor.FromFloat64(tensor.Shape{g.D}, g.Dskip))
	}
	return in
}

// §V16 tier-1: the fused selective scan matches the independent numpy
// selective_scan_ref reference at f64 1e-12.
func TestSSMParity(t *testing.T) {
	g := loadSSM(t)
	out, err := backend.Execute(backend.NewContext(), backend.OpSSM, g.inputs(true), nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range out[0].Numel() {
		idx := tensor.Unravel(k, out[0].Shape())
		if got, want := out[0].AtF64(idx...), g.Y[k]; math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("ssm y[%d]: got %.12g want %.12g", k, got, want)
		}
	}
}

// The optional skip term is exactly D_skip⊙u: the 5-input scan plus D_skip⊙u
// equals the 6-input scan.
func TestSSMSkipDecomposition(t *testing.T) {
	g := loadSSM(t)
	noskip, err := backend.Execute(backend.NewContext(), backend.OpSSM, g.inputs(false), nil)
	if err != nil {
		t.Fatal(err)
	}
	for tt := range g.L {
		for d := range g.D {
			got := noskip[0].AtF64(tt, d) + g.Dskip[d]*g.U[tt*g.D+d]
			if math.Abs(got-g.Y[tt*g.D+d]) > 1e-12 {
				t.Errorf("skip decomposition mismatch at [%d,%d]", tt, d)
			}
		}
	}
}

// §V2: central finite differences of Σ(scan) w.r.t. all six inputs match the
// analytic reverse-scan VJP.
func TestSSMGradCheck(t *testing.T) {
	g := loadSSM(t)
	ins := g.inputs(true)

	sumOut := func(xs []*tensor.Tensor) float64 {
		out, err := backend.Execute(backend.NewContext(), backend.OpSSM, xs, nil)
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
	out, err := backend.Execute(tape.Context(), backend.OpSSM, ins, nil)
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
			if num, ana := (lp-lm)/(2*h), grad.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("input %d grad[%d]: numeric %.8g vs analytic %.8g", n, k, num, ana)
			}
		}
	}
}

// The Mamba selective scan is a linear-time alternative to attention: it carries a
// state forward through the sequence. A degenerate setting with Ā = exp(0) = 1 and
// Δ = B = C = 1 turns the recurrence h_t = Ā·h_{t-1} + Δ·B·u_t into a running sum,
// so the output is the cumulative sum of the input [1,2,3,4] → [1,3,6,10].
func Example_selectiveScan() {
	L, D, N := 4, 1, 1
	u := tensor.FromFloat64(tensor.Shape{L, D}, []float64{1, 2, 3, 4})
	delta := tensor.FromFloat64(tensor.Shape{L, D}, []float64{1, 1, 1, 1})
	a := tensor.FromFloat64(tensor.Shape{D, N}, []float64{0})
	b := tensor.FromFloat64(tensor.Shape{L, N}, []float64{1, 1, 1, 1})
	c := tensor.FromFloat64(tensor.Shape{L, N}, []float64{1, 1, 1, 1})

	y, _ := backend.Execute(backend.NewContext(), backend.OpSSM, []*tensor.Tensor{u, delta, a, b, c}, nil)
	fmt.Println(y[0].AtF64(0, 0), y[0].AtF64(1, 0), y[0].AtF64(2, 0), y[0].AtF64(3, 0))
	// Output: 1 3 6 10
}
