//go:build cuda && cgo && linux

package cuda_test

import (
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/internal/bench"
	"github.com/jxsl13/goai/tensor"
)

// GroupedQueryAttention at hd=128 bypasses the WMMA fast path (hd==64 only), so it runs the
// materialize-scores triple cu_gqa_scores → cu_attn_softmax → cu_gqa_out. The fused softmax
// (double exp per score) is a large share since it is compute-bound on GA106's FP64.
func benchGQAMaterialize(b *testing.B, seq, heads, hd int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	mk := func() *cuda.DeviceF32 {
		t := bench.RandF32(tensor.Shape{seq, heads * hd}, 1)
		d, err := cuda.UploadF32(t)
		if err != nil {
			b.Fatal(err)
		}
		return d
	}
	dq, dk, dv := mk(), mk(), mk()
	defer dq.Free()
	defer dk.Free()
	defer dv.Free()
	out, err := cuda.GroupedQueryAttention(dq, dk, dv, heads, heads, true)
	if err != nil {
		b.Fatal(err)
	}
	out.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := cuda.GroupedQueryAttention(dq, dk, dv, heads, heads, true)
		if err != nil {
			b.Fatal(err)
		}
		o.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
}

func BenchmarkGQAMaterialize_1024x8x128(b *testing.B) { benchGQAMaterialize(b, 1024, 8, 128) }
func BenchmarkGQAMaterialize_2048x8x128(b *testing.B) { benchGQAMaterialize(b, 2048, 8, 128) }
