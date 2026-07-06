package ref

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type unaryGold struct {
	In  string    `json:"in"`
	F64 []float64 `json:"f64"`
	F32 []float64 `json:"f32"`
}
type binaryGold struct {
	F64 []float64 `json:"f64"`
	F32 []float64 `json:"f32"`
}
type golden struct {
	X      []float64             `json:"x"`
	XPos   []float64             `json:"xpos"`
	Y      []float64             `json:"y"`
	Unary  map[string]unaryGold  `json:"unary"`
	Binary map[string]binaryGold `json:"binary"`
}

func loadGolden(t *testing.T) golden {
	t.Helper()
	raw, err := os.ReadFile("testdata/elementwise.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g golden
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

var unaryOps = map[string]backend.Op{
	"neg": backend.OpNeg, "exp": backend.OpExp, "log": backend.OpLog,
	"tanh": backend.OpTanh, "relu": backend.OpReLU, "gelu": backend.OpGELU,
	"sigmoid": backend.OpSigmoid,
}
var binaryOps = map[string]backend.Op{
	"add": backend.OpAdd, "sub": backend.OpSub, "mul": backend.OpMul, "div": backend.OpDiv,
}

// close reports whether got matches want within rtol/atol, treating NaN==NaN and
// matching infinities by sign.
func almostEqual(got, want, rtol, atol float64) bool {
	if math.IsNaN(want) {
		return math.IsNaN(got)
	}
	if math.IsInf(want, 0) {
		return math.IsInf(got, int(math.Copysign(1, want)))
	}
	return math.Abs(got-want) <= atol+rtol*math.Abs(want)
}

func execF64(t *testing.T, op backend.Op, ins ...[]float64) []float64 {
	t.Helper()
	tin := make([]*tensor.Tensor, len(ins))
	for i, d := range ins {
		tin[i] = tensor.FromFloat64(tensor.Shape{len(d)}, d)
	}
	out, err := backend.Execute(backend.NewContext(), op, tin, nil)
	if err != nil {
		t.Fatalf("%v: %v", op, err)
	}
	res := make([]float64, out[0].Numel())
	for i := range res {
		res[i] = out[0].AtF64(i)
	}
	return res
}

func execF32(t *testing.T, op backend.Op, ins ...[]float64) []float64 {
	t.Helper()
	tin := make([]*tensor.Tensor, len(ins))
	for i, d := range ins {
		f := make([]float32, len(d))
		for j, v := range d {
			f[j] = float32(v)
		}
		tin[i] = tensor.FromFloat32(tensor.Shape{len(d)}, f)
	}
	out, err := backend.Execute(backend.NewContext(), op, tin, nil)
	if err != nil {
		t.Fatalf("%v: %v", op, err)
	}
	res := make([]float64, out[0].Numel())
	for i := range res {
		res[i] = out[0].AtF64(i)
	}
	return res
}

// §V1 PARITY: reference kernels match NumPy/libm golden within fixed tolerance
// (f64 rtol 1e-12, f32 rtol 1e-5, §R6).
func TestElementwiseParity(t *testing.T) {
	g := loadGolden(t)
	inputFor := map[string][]float64{"x": g.X, "xpos": g.XPos}

	for name, op := range unaryOps {
		gold := g.Unary[name]
		in := inputFor[gold.In]

		got64 := execF64(t, op, in)
		for i := range gold.F64 {
			if !almostEqual(got64[i], gold.F64[i], 1e-12, 1e-12) {
				t.Errorf("%s f64 [%d]: got %.17g want %.17g", name, i, got64[i], gold.F64[i])
			}
		}
		got32 := execF32(t, op, in)
		for i := range gold.F32 {
			if !almostEqual(got32[i], gold.F32[i], 1e-5, 1e-6) {
				t.Errorf("%s f32 [%d]: got %.9g want %.9g", name, i, got32[i], gold.F32[i])
			}
		}
	}

	for name, op := range binaryOps {
		gold := g.Binary[name]
		got64 := execF64(t, op, g.X, g.Y)
		for i := range gold.F64 {
			if !almostEqual(got64[i], gold.F64[i], 1e-12, 1e-12) {
				t.Errorf("%s f64 [%d]: got %.17g want %.17g", name, i, got64[i], gold.F64[i])
			}
		}
		got32 := execF32(t, op, g.X, g.Y)
		for i := range gold.F32 {
			if !almostEqual(got32[i], gold.F32[i], 1e-5, 1e-6) {
				t.Errorf("%s f32 [%d]: got %.9g want %.9g", name, i, got32[i], gold.F32[i])
			}
		}
	}
}

// §V12 EDGE: IEEE-754 special values propagate; no panics on degenerate input.
func TestElementwiseEdge(t *testing.T) {
	// div by zero → ±Inf, 0/0 → NaN
	num := []float64{1, -1, 0}
	den := []float64{0, 0, 0}
	got := execF64(t, backend.OpDiv, num, den)
	if !math.IsInf(got[0], 1) || !math.IsInf(got[1], -1) || !math.IsNaN(got[2]) {
		t.Errorf("div-by-zero IEEE: got %v", got)
	}
	// log(negative) → NaN, log(0) → -Inf
	lg := execF64(t, backend.OpLog, []float64{-1, 0})
	if !math.IsNaN(lg[0]) || !math.IsInf(lg[1], -1) {
		t.Errorf("log domain: got %v", lg)
	}
	// exp overflow → +Inf; sigmoid stays finite in [0,1] for large |x|
	if e := execF64(t, backend.OpExp, []float64{0, 1000}); e[0] != 1 || !math.IsInf(e[1], 1) {
		t.Errorf("exp: got %v", e)
	}
	sg := execF64(t, backend.OpSigmoid, []float64{-1000, 1000})
	if sg[0] < 0 || sg[0] > 1e-300 || sg[1] < 1-1e-9 || sg[1] > 1 {
		t.Errorf("sigmoid saturation: got %v", sg)
	}
	// empty tensor → empty output, no panic
	empty := tensor.New(tensor.F64, tensor.Shape{0})
	out, err := backend.Execute(backend.NewContext(), backend.OpNeg, []*tensor.Tensor{empty}, nil)
	if err != nil || out[0].Numel() != 0 {
		t.Errorf("empty op: err=%v numel=%d", err, out[0].Numel())
	}
}

// Reference kernels must honor arbitrary (non-contiguous) input layout.
func TestElementwiseNonContiguous(t *testing.T) {
	m := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	tr, _ := m.Transpose(0, 1) // logical [[1,3],[2,4]], non-contiguous
	out, err := backend.Execute(backend.NewContext(), backend.OpNeg, []*tensor.Tensor{tr}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := [][2]float64{{-1, -3}, {-2, -4}}
	for i := range 2 {
		for j := range 2 {
			if out[0].AtF64(i, j) != want[i][j] {
				t.Errorf("neg(transpose) (%d,%d)=%v want %v", i, j, out[0].AtF64(i, j), want[i][j])
			}
		}
	}
}

func TestBinaryMismatchErrors(t *testing.T) {
	a := tensor.New(tensor.F64, tensor.Shape{2, 3})
	b := tensor.New(tensor.F64, tensor.Shape{3, 2})
	if _, err := backend.Execute(backend.NewContext(), backend.OpAdd, []*tensor.Tensor{a, b}, nil); err == nil {
		t.Error("shape mismatch must error")
	}
}
