package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mkF32Parity builds the same tensor as F64 or F32 (float32(F64) elementwise) so
// the two runs share bit-identical F64 inputs modulo the F32 rounding of storage.
func mkF32Parity(f32 bool, sh tensor.Shape, data []float64) *tensor.Tensor {
	if !f32 {
		return tensor.FromFloat64(sh, data)
	}
	d32 := make([]float32, len(data))
	for i, v := range data {
		d32[i] = float32(v)
	}
	return tensor.FromFloat32(sh, d32)
}

func genF32Parity(n, seed int) []float64 {
	out := make([]float64, n)
	s := float64(seed)
	for i := range out {
		s = math.Mod(s*1103515245+12345, 2147483648)
		out[i] = s/2147483648*2 - 1 // (-1,1)
	}
	return out
}

// TestSoftplusForwardF32Parity exercises the devirtualised F32 fast path of the
// OpSoftplus forward (§T646). The F64 forward is the numerical truth (§V9); the
// F32 path reads inputs as float64, computes softplus in float64 and only rounds
// the STORED result to float32. The autograd TestSoftplusF32Parity covers the VJP
// (backward), not the forward — this closes the forward F32 gap. Uses a wide input
// range including large |x| to exercise the stable max(x,0)+log1p(exp(-|x|)) form.
func TestSoftplusForwardF32Parity(t *testing.T) {
	x := genF32Parity(64, 7)
	for i := range x {
		x[i] *= 12 // widen to (-12,12) so both saturating tails are hit
	}
	sh := tensor.Shape{8, 8}

	out64, err := backend.Execute(backend.NewContext(), backend.OpSoftplus,
		[]*tensor.Tensor{mkF32Parity(false, sh, x)}, nil)
	if err != nil {
		t.Fatalf("f64: %v", err)
	}
	out32, err := backend.Execute(backend.NewContext(), backend.OpSoftplus,
		[]*tensor.Tensor{mkF32Parity(true, sh, x)}, nil)
	if err != nil {
		t.Fatalf("f32: %v", err)
	}
	if out32[0].Dtype() != tensor.F32 {
		t.Fatalf("expected F32 output, got %v", out32[0].Dtype())
	}
	for k := range out64[0].Numel() {
		idx := tensor.Unravel(k, out64[0].Shape())
		g64 := out64[0].AtF64(idx...)
		g32 := out32[0].AtF64(idx...)
		if math.Abs(g32-g64) > 1e-3*math.Max(1, math.Abs(g64)) {
			t.Fatalf("y[%d]: f32 %.7g vs f64 %.7g", k, g32, g64)
		}
	}
}

// TestConv1DForwardF32Parity exercises the devirtualised F32 fast path of the
// OpConv1D forward (§T646). Inputs: x[L,D], w[D,K], optional bias[D]; output [L,D]
// (causal depthwise conv, §R77). The F64 forward is the numerical truth (§V9); the
// F32 path keeps the conv accumulator in float64 and only rounds the single STORED
// sum to float32. The autograd TestConv1DF32Parity covers the VJP (backward), not
// the forward — this closes the forward F32 gap. Both bias and no-bias are checked.
func TestConv1DForwardF32Parity(t *testing.T) {
	L, D, K := 7, 5, 4
	x := genF32Parity(L*D, 11)
	w := genF32Parity(D*K, 13)
	b := genF32Parity(D, 17)

	for _, hasBias := range []bool{false, true} {
		build := func(f32 bool) []*tensor.Tensor {
			in := []*tensor.Tensor{
				mkF32Parity(f32, tensor.Shape{L, D}, x),
				mkF32Parity(f32, tensor.Shape{D, K}, w),
			}
			if hasBias {
				in = append(in, mkF32Parity(f32, tensor.Shape{D}, b))
			}
			return in
		}

		out64, err := backend.Execute(backend.NewContext(), backend.OpConv1D, build(false), nil)
		if err != nil {
			t.Fatalf("hasBias=%v f64: %v", hasBias, err)
		}
		out32, err := backend.Execute(backend.NewContext(), backend.OpConv1D, build(true), nil)
		if err != nil {
			t.Fatalf("hasBias=%v f32: %v", hasBias, err)
		}
		if out32[0].Dtype() != tensor.F32 {
			t.Fatalf("hasBias=%v: expected F32 output, got %v", hasBias, out32[0].Dtype())
		}
		for k := range out64[0].Numel() {
			idx := tensor.Unravel(k, out64[0].Shape())
			g64 := out64[0].AtF64(idx...)
			g32 := out32[0].AtF64(idx...)
			if math.Abs(g32-g64) > 1e-3*math.Max(1, math.Abs(g64)) {
				t.Fatalf("hasBias=%v y[%d]: f32 %.7g vs f64 %.7g", hasBias, k, g32, g64)
			}
		}
	}
}
