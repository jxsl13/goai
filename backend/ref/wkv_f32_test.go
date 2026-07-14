package ref

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// wkvGenF32 produces deterministic pseudo-random values in (-1,1) for the WKV F32
// forward parity test (well-conditioned inputs so the exp/scan is stable).
func wkvGenF32(n, seed int) []float64 {
	out := make([]float64, n)
	s := float64(seed)
	for i := range out {
		s = math.Mod(s*1103515245+12345, 2147483648)
		out[i] = s/2147483648*2 - 1 // (-1,1)
	}
	return out
}

func wkvMkF32(sh tensor.Shape, data []float64, f32 bool) *tensor.Tensor {
	if !f32 {
		return tensor.FromFloat64(sh, data)
	}
	d32 := make([]float32, len(data))
	for i, v := range data {
		d32[i] = float32(v)
	}
	return tensor.FromFloat32(sh, d32)
}

// TestWKVForwardF32Parity exercises the devirtualised F32 fast path of the OpWKV
// forward (wkvKernel, §T645). The F64 forward is the numerical truth (§V9); the F32
// path widens each loaded input to float64, keeps the per-channel running
// numerator/denominator/max recurrence state in float64, and only narrows the STORED
// output element to float32. So the F32 output must equal float32(F64 output) within
// f32 rounding tolerance. The autograd TestWKVOpMatchesHostWKV covers the F64 forward
// and TestWKVVJPF32Parity covers the F32 VJP — this closes the F32 forward gap.
func TestWKVForwardF32Parity(t *testing.T) {
	const seq, d = 6, 4
	kData := wkvGenF32(seq*d, 11)
	vData := wkvGenF32(seq*d, 23)
	uData := wkvGenF32(d, 37)
	// decay w in a moderate range [0.3,1.1] so the exp/scan stays stable.
	wData := make([]float64, d)
	for c := range d {
		wData[c] = 0.3 + 0.8*float64(c)/float64(d-1)
	}

	build := func(f32 bool) []*tensor.Tensor {
		return []*tensor.Tensor{
			wkvMkF32(tensor.Shape{seq, d}, kData, f32),
			wkvMkF32(tensor.Shape{seq, d}, vData, f32),
			wkvMkF32(tensor.Shape{d}, wData, f32),
			wkvMkF32(tensor.Shape{d}, uData, f32),
		}
	}

	out64, err := backend.Execute(backend.NewContext(), backend.OpWKV, build(false), nil)
	if err != nil {
		t.Fatalf("f64: %v", err)
	}
	out32, err := backend.Execute(backend.NewContext(), backend.OpWKV, build(true), nil)
	if err != nil {
		t.Fatalf("f32: %v", err)
	}
	if out32[0].Dtype() != tensor.F32 {
		t.Fatalf("expected F32 output, got %v", out32[0].Dtype())
	}
	for i := range out64[0].Numel() {
		idx := tensor.Unravel(i, out64[0].Shape())
		g64 := out64[0].AtF64(idx...)
		g32 := out32[0].AtF64(idx...)
		if math.Abs(g32-g64) > 1e-3*math.Max(1, math.Abs(g64)) {
			t.Fatalf("out[%d]: f32 %.7g vs f64 %.7g", i, g32, g64)
		}
	}
}
