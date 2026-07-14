package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestElementwiseF32Parity exercises the devirtualised F32 fast paths (§T646) of
// the elementwise kernels whose F32 branch is NOT already covered by
// TestElementwiseParity: the activation backward kernels (OpSiLUBackward,
// OpGELUBackward), plus a representative unary (OpSiLU) and binary (OpMaximum),
// and the select/clamp kernels (OpWhere, OpClip). It checks that the F32 result
// equals float32(F64 result) within 1e-3 — i.e. the typed fast path stays faithful
// to the numerical-truth F64 reference after narrowing (§V9).
func TestElementwiseF32Parity(t *testing.T) {
	x := []float64{-3.5, -1.25, -0.1, 0, 0.1, 0.75, 1.5, 4.2}
	g := []float64{0.3, -0.7, 1.1, -0.25, 0.5, -1.3, 0.9, 2.0}
	y := []float64{2.0, -1.5, 0.25, 3.3, -0.4, 1.1, -2.2, 0.6}

	execF64Local := func(op backend.Op, ins ...[]float64) []float64 {
		t.Helper()
		tin := make([]*tensor.Tensor, len(ins))
		for i, d := range ins {
			tin[i] = tensor.FromFloat64(tensor.Shape{len(d)}, d)
		}
		out, err := backend.Execute(backend.NewContext(), op, tin, nil)
		if err != nil {
			t.Fatalf("f64 %v: %v", op, err)
		}
		res := make([]float64, out[0].Numel())
		for i := range res {
			res[i] = out[0].AtF64(i)
		}
		return res
	}
	execF32Local := func(op backend.Op, ins ...[]float64) []float64 {
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
			t.Fatalf("f32 %v: %v", op, err)
		}
		if out[0].Dtype() != tensor.F32 {
			t.Fatalf("f32 %v: output dtype %v, want F32", op, out[0].Dtype())
		}
		res := make([]float64, out[0].Numel())
		for i := range res {
			res[i] = out[0].AtF64(i)
		}
		return res
	}

	// (op, inputs) cases that read/write per element and now have a typed F32 path.
	cases := []struct {
		name string
		op   backend.Op
		ins  [][]float64
	}{
		{"silu_backward", backend.OpSiLUBackward, [][]float64{x, g}},
		{"gelu_backward", backend.OpGELUBackward, [][]float64{x, g}},
		{"silu", backend.OpSiLU, [][]float64{x}},          // representative unary
		{"maximum", backend.OpMaximum, [][]float64{x, y}}, // representative binary
	}
	for _, c := range cases {
		got64 := execF64Local(c.op, c.ins...)
		got32 := execF32Local(c.op, c.ins...)
		for i := range got64 {
			if math.Abs(got32[i]-float32Round(got64[i])) > 1e-3 {
				t.Errorf("%s [%d]: f32=%.7g float32(f64)=%.7g", c.name, i, got32[i], float32Round(got64[i]))
			}
		}
	}

	// OpWhere: (cond, a, b) — cond shares the a/b dtype so the typed F32 path runs.
	cond := []float64{1, 0, 1, 0, 0, 1, 1, 0}
	w64 := execF64Local(backend.OpWhere, cond, x, y)
	w32 := execF32Local(backend.OpWhere, cond, x, y)
	for i := range w64 {
		if math.Abs(w32[i]-float32Round(w64[i])) > 1e-3 {
			t.Errorf("where [%d]: f32=%.7g float32(f64)=%.7g", i, w32[i], float32Round(w64[i]))
		}
	}

	// OpClip: single input + ClipAttrs (attrs identical for both dtypes).
	clipAttrs := backend.ClipAttrs{Lo: -1, Hi: 1}
	clip := func(dtype tensor.Dtype) []float64 {
		var in *tensor.Tensor
		if dtype == tensor.F32 {
			f := make([]float32, len(x))
			for i, v := range x {
				f[i] = float32(v)
			}
			in = tensor.FromFloat32(tensor.Shape{len(x)}, f)
		} else {
			in = tensor.FromFloat64(tensor.Shape{len(x)}, x)
		}
		out, err := backend.Execute(backend.NewContext(), backend.OpClip, []*tensor.Tensor{in}, clipAttrs)
		if err != nil {
			t.Fatalf("clip %v: %v", dtype, err)
		}
		res := make([]float64, out[0].Numel())
		for i := range res {
			res[i] = out[0].AtF64(i)
		}
		return res
	}
	clip64, clip32 := clip(tensor.F64), clip(tensor.F32)
	for i := range clip64 {
		if math.Abs(clip32[i]-float32Round(clip64[i])) > 1e-3 {
			t.Errorf("clip [%d]: f32=%.7g float32(f64)=%.7g", i, clip32[i], float32Round(clip64[i]))
		}
	}
}

func float32Round(v float64) float64 { return float64(float32(v)) }
