package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSSMForwardF32Parity exercises the devirtualised F32 fast path of the OpSSM
// forward (§T645). The generic F64 forward is the numerical truth (§V9); the F32
// path reads inputs as float64, keeps the scan state in float64, and only rounds
// the STORED output to float32. So F32 forward y must equal float32(F64 forward y)
// within f32 rounding tolerance. TestSSMParity covers F64 only, and the autograd
// TestSSMF32Parity exercises the VJP (not the forward) — this closes the forward
// F32 gap. Shapes (Gu & Dao 2023): u,delta [L,D]; A [D,N]; B,C [L,N]; D_skip [D].
func TestSSMForwardF32Parity(t *testing.T) {
	L, D, N := 6, 4, 3

	// Deterministic, well-conditioned inputs: delta>0 and A<0 keep Ā=exp(Δ·A) stable.
	gen := func(n, seed int, positive, negative bool) []float64 {
		out := make([]float64, n)
		s := float64(seed)
		for i := range out {
			s = math.Mod(s*1103515245+12345, 2147483648)
			v := s/2147483648*2 - 1 // (-1,1)
			switch {
			case positive:
				v = 0.2 + 0.8*(v*0.5+0.5) // (0.2,1.0)
			case negative:
				v = -(0.2 + 0.8*(v*0.5+0.5)) // (-1.0,-0.2)
			}
			out[i] = v
		}
		return out
	}

	for _, hasSkip := range []bool{false, true} {
		u := gen(L*D, 1, false, false)
		delta := gen(L*D, 2, true, false)
		A := gen(D*N, 3, false, true)
		B := gen(L*N, 4, false, false)
		C := gen(L*N, 5, false, false)
		dskip := gen(D, 6, false, false)

		build := func(f32 bool) []*tensor.Tensor {
			mk2 := func(sh tensor.Shape, data []float64) *tensor.Tensor {
				if !f32 {
					return tensor.FromFloat64(sh, data)
				}
				d32 := make([]float32, len(data))
				for i, v := range data {
					d32[i] = float32(v)
				}
				return tensor.FromFloat32(sh, d32)
			}
			in := []*tensor.Tensor{
				mk2(tensor.Shape{L, D}, u),
				mk2(tensor.Shape{L, D}, delta),
				mk2(tensor.Shape{D, N}, A),
				mk2(tensor.Shape{L, N}, B),
				mk2(tensor.Shape{L, N}, C),
			}
			if hasSkip {
				in = append(in, mk2(tensor.Shape{D}, dskip))
			}
			return in
		}

		out64, err := backend.Execute(backend.NewContext(), backend.OpSSM, build(false), nil)
		if err != nil {
			t.Fatalf("hasSkip=%v f64: %v", hasSkip, err)
		}
		out32, err := backend.Execute(backend.NewContext(), backend.OpSSM, build(true), nil)
		if err != nil {
			t.Fatalf("hasSkip=%v f32: %v", hasSkip, err)
		}
		if out32[0].Dtype() != tensor.F32 {
			t.Fatalf("hasSkip=%v: expected F32 output, got %v", hasSkip, out32[0].Dtype())
		}
		for k := range out64[0].Numel() {
			idx := tensor.Unravel(k, out64[0].Shape())
			g64 := out64[0].AtF64(idx...)
			g32 := out32[0].AtF64(idx...)
			if math.Abs(g32-g64) > 1e-3*math.Max(1, math.Abs(g64)) {
				t.Fatalf("hasSkip=%v y[%d]: f32 %.7g vs f64 %.7g", hasSkip, k, g32, g64)
			}
		}
	}
}
