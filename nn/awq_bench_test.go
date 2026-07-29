package nn

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// BenchmarkReconErrMat covers AWQ's reconstruction-error kernel ‖(W−Ŵ)·X‖_F, called once
// per candidate scale during AWQ calibration. GEMM-shaped: out=512, in=512, samples=128.
func BenchmarkReconErrMat(b *testing.B) {
	const out, in, samples = 512, 512, 128
	w := make([][]float64, out)
	for r := range w {
		w[r] = make([]float64, in)
		for i := range w[r] {
			w[r][i] = math.Sin(float64(r*in+i) * 0.001)
		}
	}
	wq := tensor.New(tensor.F64, tensor.Shape{out, in})
	wqs := wq.Storage().F64()
	for i := range wqs {
		wqs[i] = math.Round(math.Sin(float64(i)*0.001)*8) / 8
	}
	x := tensor.New(tensor.F64, tensor.Shape{in, samples})
	xs := x.Storage().F64()
	for i := range xs {
		xs[i] = math.Cos(float64(i) * 0.002)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = reconErrMat(w, wq, x, out, in, samples)
	}
}

// BenchmarkQuantizeScaled covers AWQ's per-row symmetric quantizer, the sibling of
// reconErrMat inside the alpha grid-search. out=512, in=512, levels=16.
func BenchmarkQuantizeScaled(b *testing.B) {
	const out, in, levels = 512, 512, 16
	w := make([][]float64, out)
	for r := range w {
		w[r] = make([]float64, in)
		for i := range w[r] {
			w[r][i] = math.Sin(float64(r*in+i) * 0.001)
		}
	}
	scale := make([]float64, in)
	for i := range scale {
		scale[i] = 0.5 + 0.5*math.Cos(float64(i)*0.01)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		_ = quantizeScaled(w, scale, out, in, levels, tensor.F64)
	}
}
