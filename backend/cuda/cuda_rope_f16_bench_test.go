//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// cu_rope_f16 (f16 prefill RoPE) has the same per-head-redundant sincosf/FP64 as cu_rope_f32.
func benchRoPEF16(b *testing.B, seq, heads, hd int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{seq, heads * hd}, 1)
	dx32, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx32.Free()
	dx, err := cuda.F16FromF32(dx32)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	rq := backend.RoPEAttrs{Heads: heads, Base: 10000}
	invD, _ := backend.RoPEFreqs(hd, rq)
	inv32 := make([]float32, len(invD))
	for i := range invD {
		inv32[i] = float32(invD[i])
	}
	invDev, _ := cuda.NewDeviceF32(1, len(inv32))
	defer invDev.Free()
	invDev.UploadF32(inv32)
	if err := dx.RoPE(invDev, rq); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RoPE(invDev, rq)
	}
	cuda.GraphSync()
	b.StopTimer()
	gbps := 2 * 2 * float64(seq) * float64(heads) * float64(hd) / (b.Elapsed().Seconds() / float64(b.N)) / 1e9 // f16: 2 bytes
	b.ReportMetric(gbps, "GB/s")
}

func BenchmarkRoPEF16_2048x32x128(b *testing.B) { benchRoPEF16(b, 2048, 32, 128) }
func BenchmarkRoPEF16_4096x32x128(b *testing.B) { benchRoPEF16(b, 4096, 32, 128) }
