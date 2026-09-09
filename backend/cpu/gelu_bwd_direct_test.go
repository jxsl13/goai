package cpu_test

import (
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func TestGELUBackwardF64DirectShapes(t *testing.T) {
	for _, n := range []int{0, 1, 2, 3, 7, 17, 2048, 200003, 262144} {
		t.Run(fmt.Sprintf("n%d", n), func(t *testing.T) {
			x, g := bench.RandF64(tensor.Shape{n}, 1), bench.RandF64(tensor.Shape{n}, 2)
			for i := range x.Storage().F64() {
				x.Storage().F64()[i] *= 15
			}
			checkGELUBackwardF64Direct(t, x, g)
		})
	}
	for _, shape := range []tensor.Shape{nil, {}, {3, 0}, {2, 3, 5}} {
		t.Run(fmt.Sprintf("shape%v_nil%t", shape, shape == nil), func(t *testing.T) {
			checkGELUBackwardF64Direct(t, bench.RandF64(shape, 3), bench.RandF64(shape, 4))
		})
	}
}

func TestGELUBackwardF64DirectSpecialValues(t *testing.T) {
	if geluF64Tolerant {
		t.Skip("exact scalar GELU contract; amd64 SIMD retains its existing accuracy tests")
	}
	values := []float64{
		0, math.Copysign(0, -1), math.SmallestNonzeroFloat64, -math.SmallestNonzeroFloat64,
		1e-300, -1e-300, 1e-150, -1e-150, 0.5, -0.5, 1, -1, 8, -8, 40, -40,
		1e150, -1e150, math.MaxFloat64, -math.MaxFloat64,
		math.Inf(1), math.Inf(-1), math.NaN(), math.Float64frombits(0xfff8000000000042),
	}
	shape := tensor.Shape{len(values), len(values)}
	x, g := tensor.New(tensor.F64, shape), tensor.New(tensor.F64, shape)
	for i, xv := range values {
		for j, gv := range values {
			x.SetF64(xv, i, j)
			g.SetF64(gv, i, j)
		}
	}
	checkGELUBackwardF64Direct(t, x, g)
}

func TestGELUBackwardF64DirectViews(t *testing.T) {
	slice := func(x *tensor.Tensor, dim, start, stop int) *tensor.Tensor {
		t.Helper()
		v, err := x.Slice(dim, start, stop)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	transpose := func(x *tensor.Tensor) *tensor.Tensor {
		t.Helper()
		v, err := x.Transpose(0, 1)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	t.Run("offset", func(t *testing.T) {
		x := slice(bench.RandF64(tensor.Shape{29}, 1), 0, 3, 20)
		g := slice(bench.RandF64(tensor.Shape{31}, 2), 0, 5, 22)
		checkGELUBackwardF64Direct(t, x, g)
	})
	t.Run("transposed_and_offset", func(t *testing.T) {
		x := transpose(slice(bench.RandF64(tensor.Shape{11, 13}, 1), 1, 2, 11))
		g := slice(bench.RandF64(tensor.Shape{9, 13}, 2), 1, 1, 12)
		checkGELUBackwardF64Direct(t, x, g)
	})
	t.Run("gradient_transposed", func(t *testing.T) {
		x := bench.RandF64(tensor.Shape{9, 11}, 1)
		g := transpose(bench.RandF64(tensor.Shape{11, 9}, 2))
		checkGELUBackwardF64Direct(t, x, g)
	})
	t.Run("parallel_transposed", func(t *testing.T) {
		x := transpose(bench.RandF64(tensor.Shape{257, 1024}, 1))
		g := transpose(bench.RandF64(tensor.Shape{257, 1024}, 2))
		checkGELUBackwardF64Direct(t, x, g)
	})
}

// checkGELUBackwardF64Direct compares the public CPU dispatch with the independent
// reference, and checks the entire backing storage (including outside a view).
func checkGELUBackwardF64Direct(t *testing.T, x, g *tensor.Tensor) {
	t.Helper()
	inputs := []*tensor.Tensor{x, g}
	for _, input := range inputs {
		storage := input.Storage()
		before := slices.Clone(storage.F64())
		shape, strides, offset := slices.Clone(input.Shape()), slices.Clone(input.Strides()), input.Offset()
		defer func() {
			if input.Storage() != storage || input.Offset() != offset || !slices.Equal(input.Shape(), shape) || !slices.Equal(input.Strides(), strides) {
				t.Error("GELU backward changed input view metadata")
			}
			for i, v := range before {
				if math.Float64bits(storage.F64()[i]) != math.Float64bits(v) {
					t.Errorf("GELU backward changed input storage at %d", i)
					break
				}
			}
		}()
	}
	got := run(t, cpuBackend(t), backend.OpGELUBackward, inputs...)
	want := run(t, backend.Reference(), backend.OpGELUBackward, inputs...)
	if got.Dtype() != tensor.F64 || !got.Shape().Equal(x.Shape()) || !got.IsContiguous() || got.Offset() != 0 {
		t.Fatalf("unexpected output: dtype=%v shape=%v offset=%d", got.Dtype(), got.Shape(), got.Offset())
	}
	if got.Storage() == x.Storage() || got.Storage() == g.Storage() {
		t.Fatal("GELU backward output aliases an input")
	}
	for i, gv := range got.Storage().F64() {
		wv := want.Storage().F64()[i]
		switch {
		case math.IsNaN(wv):
			if !math.IsNaN(gv) {
				t.Fatalf("element %d: cpu %v, ref NaN", i, gv)
			}
		case math.IsNaN(gv), math.IsInf(wv, 0), math.IsInf(gv, 0), !geluF64Tolerant:
			if math.Float64bits(gv) != math.Float64bits(wv) {
				t.Fatalf("element %d: cpu %.17g (%016x), ref %.17g (%016x)", i, gv, math.Float64bits(gv), wv, math.Float64bits(wv))
			}
		default:
			// Preserve the existing amd64 SIMD TestCPUGeluBackwardCrossReference envelope.
			if math.Abs(gv-wv) > 1e-12*math.Max(1, math.Abs(wv)) {
				t.Fatalf("element %d: cpu %.17g, ref %.17g", i, gv, wv)
			}
		}
	}
}

type geluBackwardDirectRecorder struct{ calls int }

func (r *geluBackwardDirectRecorder) Record(backend.Op, []*tensor.Tensor, []*tensor.Tensor, backend.Attrs) {
	r.calls++
}

func TestGELUBackwardF64DirectFallback(t *testing.T) {
	cpuCtx := backend.NewContext().WithBackend(cpuBackend(t))
	refCtx := backend.NewContext().WithBackend(backend.Reference())
	x := bench.RandF64(tensor.Shape{17}, 1)
	g := bench.RandF32(tensor.Shape{17}, 2)
	t.Run("mixed_gradient_dtype", func(t *testing.T) {
		xBefore, gBefore := slices.Clone(x.Storage().F64()), slices.Clone(g.Storage().F32())
		recorder := &geluBackwardDirectRecorder{}
		inputs := []*tensor.Tensor{x, g}
		got, err := backend.Execute(cpuCtx.WithRecorder(recorder), backend.OpGELUBackward, inputs, nil)
		if err != nil {
			t.Fatal(err)
		}
		want, err := backend.Execute(refCtx, backend.OpGELUBackward, inputs, nil)
		if err != nil {
			t.Fatal(err)
		}
		if recorder.calls != 1 {
			t.Fatalf("recorded %d calls, want one", recorder.calls)
		}
		for i, v := range got[0].Storage().F64() {
			if math.Float64bits(v) != math.Float64bits(want[0].Storage().F64()[i]) {
				t.Fatalf("mixed dtype fallback differs at %d", i)
			}
		}
		if !slices.Equal(x.Storage().F64(), xBefore) || !slices.Equal(g.Storage().F32(), gBefore) {
			t.Fatal("mixed dtype fallback changed an input")
		}
	})
	for _, tc := range []struct {
		name   string
		inputs []*tensor.Tensor
		attrs  backend.Attrs
	}{
		{name: "missing_gradient", inputs: []*tensor.Tensor{x}},
		{name: "extra_input", inputs: []*tensor.Tensor{x, x, x}},
		{name: "shape_mismatch", inputs: []*tensor.Tensor{x, tensor.New(tensor.F64, tensor.Shape{1, 17})}},
		{name: "wrong_attrs", inputs: []*tensor.Tensor{x, x}, attrs: backend.ClipAttrs{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			recorder := &geluBackwardDirectRecorder{}
			_, got := backend.Execute(cpuCtx.WithRecorder(recorder), backend.OpGELUBackward, tc.inputs, tc.attrs)
			_, want := backend.Execute(refCtx, backend.OpGELUBackward, tc.inputs, tc.attrs)
			if got == nil || want == nil || got.Error() != want.Error() {
				t.Fatalf("cpu error %v, ref error %v", got, want)
			}
			if recorder.calls != 0 {
				t.Fatalf("recorded %d failed calls", recorder.calls)
			}
		})
	}
}

// BenchmarkGELUBackwardF64DirectDispatch is shared unchanged by the frozen control
// and candidate binaries. It includes Execute and output allocation, like the
// production 256K benchmark, with the same deterministic input seeds.
func BenchmarkGELUBackwardF64DirectDispatch(b *testing.B) {
	be, ok := backend.Get(backend.CPU)
	if !ok {
		b.Fatal("cpu backend not registered")
	}
	for _, n := range []int{2048, 262144} {
		b.Run(fmt.Sprintf("n%d", n), func(b *testing.B) {
			ctx := backend.NewContext().WithBackend(be)
			inputs := []*tensor.Tensor{bench.RandF64(tensor.Shape{n}, 1), bench.RandF64(tensor.Shape{n}, 2)}
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := backend.Execute(ctx, backend.OpGELUBackward, inputs, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
