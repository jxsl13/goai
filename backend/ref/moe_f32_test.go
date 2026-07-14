package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// moeGenF32 produces deterministic pseudo-random values in a chosen range for the
// F32 forward parity tests. positive=true maps into (0.2,1.0) so router-gate sums
// (denominators) stay well away from zero.
func moeGenF32(n, seed int, positive bool) []float64 {
	out := make([]float64, n)
	s := float64(seed)
	for i := range out {
		s = math.Mod(s*1103515245+12345, 2147483648)
		v := s/2147483648*2 - 1 // (-1,1)
		if positive {
			v = 0.2 + 0.8*(v*0.5+0.5) // (0.2,1.0)
		}
		out[i] = v
	}
	return out
}

func moeMkF32(sh tensor.Shape, data []float64, f32 bool) *tensor.Tensor {
	if !f32 {
		return tensor.FromFloat64(sh, data)
	}
	d32 := make([]float32, len(data))
	for i, v := range data {
		d32[i] = float32(v)
	}
	return tensor.FromFloat32(sh, d32)
}

// TestMoEBalanceForwardF32Parity exercises the devirtualised F32 fast path of the
// OpMoEBalance forward (§T645). The F64 forward is the numerical truth (§V9); the
// F32 path widens each logit to float64, keeps the softmax accumulators in float64,
// and only narrows the STORED scalar loss to float32. So the F32 loss must equal
// float32(F64 loss) within f32 rounding tolerance. The nn TestMoEBalanceParity
// covers F64 only — this closes the forward F32 gap.
func TestMoEBalanceForwardF32Parity(t *testing.T) {
	T, N := 7, 4
	logits := moeGenF32(T*N, 11, false)
	assign := make([]float64, T)
	for tt := range assign {
		assign[tt] = float64(tt % N) // valid top-1 index in [0,N)
	}
	attrs := backend.MoEBalanceAttrs{Alpha: 0.013}

	build := func(f32 bool) []*tensor.Tensor {
		return []*tensor.Tensor{
			moeMkF32(tensor.Shape{T, N}, logits, f32),
			moeMkF32(tensor.Shape{T}, assign, f32),
		}
	}

	out64, err := backend.Execute(backend.NewContext(), backend.OpMoEBalance, build(false), attrs)
	if err != nil {
		t.Fatalf("f64: %v", err)
	}
	out32, err := backend.Execute(backend.NewContext(), backend.OpMoEBalance, build(true), attrs)
	if err != nil {
		t.Fatalf("f32: %v", err)
	}
	if out32[0].Dtype() != tensor.F32 {
		t.Fatalf("expected F32 output, got %v", out32[0].Dtype())
	}
	l64, l32 := out64[0].AtF64(), out32[0].AtF64()
	if math.Abs(l32-l64) > 1e-3*math.Max(1, math.Abs(l64)) {
		t.Fatalf("loss: f32 %.7g vs f64 %.7g", l32, l64)
	}
}

// TestMoECombineForwardF32Parity exercises the devirtualised F32 fast path of the
// OpMoECombine forward (§T645). The F64 forward is the numerical truth (§V9); the
// F32 path widens the gate weights and expert outputs to float64, accumulates the
// per-token denominator and the mixture in float64, and only narrows each STORED
// output element to float32. So the F32 output must equal float32(F64 output)
// within f32 rounding tolerance. Router weights are positive so denominators are
// nonzero. The nn TestMoECombineParity covers F64 only — this closes the F32 gap.
func TestMoECombineForwardF32Parity(t *testing.T) {
	T, E, D := 5, 3, 4
	w := moeGenF32(T*E, 21, true) // positive gates ⇒ denom>0
	experts := make([][]float64, E)
	for i := range experts {
		experts[i] = moeGenF32(T*D, 31+i, false)
	}

	build := func(f32 bool) []*tensor.Tensor {
		in := []*tensor.Tensor{moeMkF32(tensor.Shape{T, E}, w, f32)}
		for i := range experts {
			in = append(in, moeMkF32(tensor.Shape{T, D}, experts[i], f32))
		}
		return in
	}

	out64, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, build(false), nil)
	if err != nil {
		t.Fatalf("f64: %v", err)
	}
	out32, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, build(true), nil)
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
			t.Fatalf("out[%d]: f32 %.7g vs f64 %.7g", k, g32, g64)
		}
	}
}
