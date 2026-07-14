package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// mlaGenF32 produces deterministic pseudo-random values in (-1,1) for the MLA F32
// forward parity test (well-conditioned inputs so the softmax is stable).
func mlaGenF32(n, seed int) []float64 {
	out := make([]float64, n)
	s := float64(seed)
	for i := range out {
		s = math.Mod(s*1103515245+12345, 2147483648)
		out[i] = s/2147483648*2 - 1 // (-1,1)
	}
	return out
}

func mlaMkF32(sh tensor.Shape, data []float64, f32 bool) *tensor.Tensor {
	if !f32 {
		return tensor.FromFloat64(sh, data)
	}
	d32 := make([]float32, len(data))
	for i, v := range data {
		d32[i] = float32(v)
	}
	return tensor.FromFloat32(sh, d32)
}

// TestMLAForwardF32Parity exercises the devirtualised F32 fast path of the OpMLA
// forward (mlaRoPE + mlaKernel, §T645). The F64 forward is the numerical truth
// (§V9); the F32 path widens each loaded input to float64, keeps every attention
// accumulator (score, softmax denominator, output mix) in float64, and only narrows
// the STORED output element to float32. So the F32 output must equal float32(F64
// output) within f32 rounding tolerance. The nn TestMLAOpParity / TestMLALayerForward
// and autograd TestMLAF32Parity cover F64 forward and the F32 VJP respectively — this
// closes the F32 forward gap. Both causal on and off are covered.
func TestMLAForwardF32Parity(t *testing.T) {
	seq, heads, dh, dR := 5, 2, 4, 4
	hdh := heads * dh
	qC := mlaGenF32(seq*hdh, 11)
	kC := mlaGenF32(seq*hdh, 23)
	vC := mlaGenF32(seq*hdh, 37)
	qRpre := mlaGenF32(seq*heads*dR, 51)
	kRpre := mlaGenF32(seq*dR, 67)

	build := func(f32 bool) []*tensor.Tensor {
		return []*tensor.Tensor{
			mlaMkF32(tensor.Shape{seq, hdh}, qC, f32),
			mlaMkF32(tensor.Shape{seq, hdh}, kC, f32),
			mlaMkF32(tensor.Shape{seq, hdh}, vC, f32),
			mlaMkF32(tensor.Shape{seq, heads * dR}, qRpre, f32),
			mlaMkF32(tensor.Shape{seq, dR}, kRpre, f32),
		}
	}

	for _, causal := range []bool{false, true} {
		attrs := backend.MLAAttrs{Heads: heads, Causal: causal, RoPEBase: 10000}
		out64, err := backend.Execute(backend.NewContext(), backend.OpMLA, build(false), attrs)
		if err != nil {
			t.Fatalf("causal=%v f64: %v", causal, err)
		}
		out32, err := backend.Execute(backend.NewContext(), backend.OpMLA, build(true), attrs)
		if err != nil {
			t.Fatalf("causal=%v f32: %v", causal, err)
		}
		if out32[0].Dtype() != tensor.F32 {
			t.Fatalf("causal=%v: expected F32 output, got %v", causal, out32[0].Dtype())
		}
		for k := range out64[0].Numel() {
			idx := tensor.Unravel(k, out64[0].Shape())
			g64 := out64[0].AtF64(idx...)
			g32 := out32[0].AtF64(idx...)
			if math.Abs(g32-g64) > 1e-3*math.Max(1, math.Abs(g64)) {
				t.Fatalf("causal=%v out[%d]: f32 %.7g vs f64 %.7g", causal, k, g32, g64)
			}
		}
	}
}
