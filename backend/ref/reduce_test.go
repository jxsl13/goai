package ref

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

type redOp struct {
	F64   []float64 `json:"f64"`
	F32   []float64 `json:"f32"`
	Shape []int     `json:"shape"`
}
type argOp struct {
	Idx   []float64 `json:"idx"`
	Shape []int     `json:"shape"`
}
type reduceFile struct {
	Reduce struct {
		Input []float64 `json:"input"`
		Shape []int     `json:"shape"`
		Ops   struct {
			Sum    map[string]redOp `json:"sum"`
			Mean   map[string]redOp `json:"mean"`
			Max    map[string]redOp `json:"max"`
			Min    map[string]redOp `json:"min"`
			Argmax map[string]argOp `json:"argmax"`
		} `json:"ops"`
	} `json:"reduce"`
}

func loadReduce(t *testing.T) reduceFile {
	t.Helper()
	raw, err := os.ReadFile("testdata/elementwise.json")
	if err != nil {
		t.Fatalf("read golden (run `make golden`): %v", err)
	}
	var g reduceFile
	if err := json.Unmarshal(raw, &g); err != nil {
		t.Fatal(err)
	}
	return g
}

func shapeEqual(a []int, s tensor.Shape) bool {
	if len(a) != len(s) {
		return false
	}
	for i := range a {
		if a[i] != s[i] {
			return false
		}
	}
	return true
}

// runReduce executes op over axes on the input (as F64 or F32) and returns the
// flattened result plus the output shape.
func runReduce(t *testing.T, op backend.Op, in []float64, shape []int, axes []int, f32 bool) ([]float64, tensor.Shape) {
	t.Helper()
	var x *tensor.Tensor
	if f32 {
		f := make([]float32, len(in))
		for i, v := range in {
			f[i] = float32(v)
		}
		x = tensor.FromFloat32(tensor.Shape(shape), f)
	} else {
		x = tensor.FromFloat64(tensor.Shape(shape), in)
	}
	out, err := backend.Execute(backend.NewContext(), op, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		t.Fatalf("%v axes=%v: %v", op, axes, err)
	}
	res := make([]float64, out[0].Numel())
	for i := range res {
		res[i] = out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
	}
	return res, out[0].Shape()
}

var axisArg = map[string][]int{"axis0": {0}, "axis1": {1}, "all": nil}

// §V1 PARITY + §V10 ACCUM: reductions match golden (f64 accumulation) within
// tolerance, and produce the expected output shapes.
func TestReduceParity(t *testing.T) {
	g := loadReduce(t)
	in, sh := g.Reduce.Input, g.Reduce.Shape

	table := map[backend.Op]map[string]redOp{
		backend.OpSum:  g.Reduce.Ops.Sum,
		backend.OpMean: g.Reduce.Ops.Mean,
		backend.OpMax:  g.Reduce.Ops.Max,
		backend.OpMin:  g.Reduce.Ops.Min,
	}
	for op, byAxis := range table {
		for key, gold := range byAxis {
			axes := axisArg[key]
			got64, oshape := runReduce(t, op, in, sh, axes, false)
			if !shapeEqual(gold.Shape, oshape) {
				t.Errorf("%v %s: shape %v, want %v", op, key, oshape, gold.Shape)
			}
			for i := range gold.F64 {
				if !almostEqual(got64[i], gold.F64[i], 1e-12, 1e-12) {
					t.Errorf("%v %s f64[%d]: got %.17g want %.17g", op, key, i, got64[i], gold.F64[i])
				}
			}
			got32, _ := runReduce(t, op, in, sh, axes, true)
			for i := range gold.F32 {
				if !almostEqual(got32[i], gold.F32[i], 1e-5, 1e-6) {
					t.Errorf("%v %s f32[%d]: got %.9g want %.9g", op, key, i, got32[i], gold.F32[i])
				}
			}
		}
	}
}

func TestArgmaxParity(t *testing.T) {
	g := loadReduce(t)
	in, sh := g.Reduce.Input, g.Reduce.Shape
	x := tensor.FromFloat64(tensor.Shape(sh), in)

	check := func(attrs backend.Attrs, gold argOp, label string) {
		out, err := backend.Execute(backend.NewContext(), backend.OpArgMax, []*tensor.Tensor{x}, attrs)
		if err != nil {
			t.Fatalf("argmax %s: %v", label, err)
		}
		if !shapeEqual(gold.Shape, out[0].Shape()) {
			t.Errorf("argmax %s shape %v, want %v", label, out[0].Shape(), gold.Shape)
		}
		for i := range gold.Idx {
			got := out[0].AtF64(tensor.Unravel(i, out[0].Shape())...)
			if got != gold.Idx[i] {
				t.Errorf("argmax %s [%d]: got %v want %v", label, i, got, gold.Idx[i])
			}
		}
	}
	check(backend.ArgMaxAttrs{Axis: 0}, g.Reduce.Ops.Argmax["axis0"], "axis0")
	check(backend.ArgMaxAttrs{Axis: 1}, g.Reduce.Ops.Argmax["axis1"], "axis1")
	check(nil, g.Reduce.Ops.Argmax["flat"], "flat") // no axis → flat
}

// §V10 ACCUM: f32 reductions accumulate in f64 so large-K sums do not drift.
// Naive f32 accumulation of the same data drifts measurably; the kernel must not.
func TestReduceF64Accumulation(t *testing.T) {
	const n = 100_000
	f := make([]float32, n)
	for i := range f {
		f[i] = 0.1
	}
	x := tensor.FromFloat32(tensor.Shape{n}, f)
	out, err := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{x}, nil)
	if err != nil {
		t.Fatal(err)
	}
	got := out[0].AtF64()

	// true sum of n f32(0.1) values, accumulated exactly
	var exact float64
	for range n {
		exact += float64(float32(0.1))
	}
	// naive f32 accumulation, to show the drift the invariant prevents
	var naive float32
	for range n {
		naive += 0.1
	}
	if math.Abs(got-exact) > 1e-3 {
		t.Errorf("kernel sum %.6f drifted from f64-accumulated %.6f", got, exact)
	}
	if math.Abs(float64(naive)-exact) < 0.5 {
		t.Skip("naive f32 did not drift enough on this platform to contrast")
	}
	t.Logf("f64-accum kernel=%.6f exact=%.6f  vs naive-f32=%.6f (drift %.4f)",
		got, exact, naive, math.Abs(float64(naive)-exact))
}

func TestReduceKeepdimsAndEmpty(t *testing.T) {
	x := tensor.FromFloat64(tensor.Shape{3, 4}, make([]float64, 12))
	out, err := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{x},
		backend.ReduceAttrs{Axes: []int{1}, KeepDims: true})
	if err != nil {
		t.Fatal(err)
	}
	if !out[0].Shape().Equal(tensor.Shape{3, 1}) {
		t.Errorf("keepdims shape %v, want (3,1)", out[0].Shape())
	}

	// empty reduction identities: sum→0, mean→NaN
	e := tensor.New(tensor.F64, tensor.Shape{2, 0})
	s, _ := backend.Execute(backend.NewContext(), backend.OpSum, []*tensor.Tensor{e}, backend.ReduceAttrs{Axes: []int{1}})
	if s[0].AtF64(0) != 0 || s[0].AtF64(1) != 0 {
		t.Error("empty sum must be identity 0")
	}
	m, _ := backend.Execute(backend.NewContext(), backend.OpMean, []*tensor.Tensor{e}, backend.ReduceAttrs{Axes: []int{1}})
	if !math.IsNaN(m[0].AtF64(0)) {
		t.Error("empty mean must be NaN")
	}
}
