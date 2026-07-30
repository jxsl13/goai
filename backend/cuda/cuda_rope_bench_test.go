//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/tensor"
)

// cu_rope_f32 computes sincosf per (pos,head,dim-pair) though the angle depends only on (pos,i)
// — redundant across heads. Bench at a prefill shape to measure whether it is sincosf-bound
// (caching helps) or bandwidth-bound (it won't). Reports GB/s (2 f32 R+W per rotated element).
func benchRoPE(b *testing.B, seq, heads, hd int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	x := bench.RandF32(tensor.Shape{seq, heads * hd}, 1)
	dx, err := cuda.UploadF32(x)
	if err != nil {
		b.Fatal(err)
	}
	defer dx.Free()
	attrs := backend.RoPEAttrs{Heads: heads, Base: 10000}
	if err := dx.RoPE(attrs); err != nil {
		b.Fatal(err)
	}
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = dx.RoPE(attrs)
	}
	cuda.GraphSync()
	b.StopTimer()
	// bytes = seq*heads*hd elements * 4B * 2 (read+write)
	gbps := 2 * 4 * float64(seq) * float64(heads) * float64(hd) / (b.Elapsed().Seconds() / float64(b.N)) / 1e9
	b.ReportMetric(gbps, "GB/s")
}

func BenchmarkRoPE_2048x32x128(b *testing.B) { benchRoPE(b, 2048, 32, 128) }
func BenchmarkRoPE_4096x32x128(b *testing.B) { benchRoPE(b, 4096, 32, 128) }
