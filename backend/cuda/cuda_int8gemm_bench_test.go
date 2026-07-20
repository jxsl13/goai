//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// Measure-first: is INT8 MMQ (W8A8, Ampere int8 tensor cores) ~2× f16 cuBLAS at decode GEMM dims?
// The batched decode step is ~70% cuBLAS-f16 GEMM at ~88% of f16 peak; int8 tensor cores are ~2×
// f16 throughput on the RTX 3060 (Ampere), so W8A8 decode GEMMs could nearly halve the GEMM path.
func benchGemmF16vsMMQ(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(1))
	wt := tensor.New(tensor.F32, tensor.Shape{k, n})
	wf := wt.Storage().F32()
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.02
	}
	af := make([]float32, m*k)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.1
	}
	a, _ := cuda.NewDeviceF32(m, k)
	a.UploadF32(af)
	defer a.Free()

	wf16, err := cuda.NewResidentBF16(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer wf16.Free()
	wmmq, err := cuda.NewResidentMMQ(wt)
	if err != nil {
		b.Fatal(err)
	}
	defer wmmq.Free()

	b.Run("f16", func(b *testing.B) {
		o, _ := wf16.MatMulDevice(a)
		o.Free()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			o, _ := wf16.MatMulDevice(a)
			o.Free()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("mmq_int8", func(b *testing.B) {
		o, err := wmmq.MatMulDevice(a)
		if err != nil {
			b.Skipf("mmq: %v", err)
		}
		o.Free()
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			o, _ := wmmq.MatMulDevice(a)
			o.Free()
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
}

func BenchmarkGemm_512x2048x2048(b *testing.B) { benchGemmF16vsMMQ(b, 512, 2048, 2048) }
func BenchmarkGemm_512x2048x5632(b *testing.B) { benchGemmF16vsMMQ(b, 512, 2048, 5632) }

func benchInt8Gemm(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	p, err := cuda.NewInt8GemmProbe(m, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer p.Free()
	if err := p.Run(); err != nil {
		b.Skipf("int8 gemm: %v", err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		p.Run()
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TOP/s")
}

func BenchmarkGemmInt8_512x2048x2048(b *testing.B) { benchInt8Gemm(b, 512, 2048, 2048) }
func BenchmarkGemmInt8_512x2048x5632(b *testing.B) { benchInt8Gemm(b, 512, 2048, 5632) }
