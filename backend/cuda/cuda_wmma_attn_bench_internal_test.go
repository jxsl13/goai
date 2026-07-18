//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// A/B: the fused nvcc WMMA attention kernel vs the current cuBLAS-batched GroupedQueryAttention
// (materialized global scores) at a TinyLlama MHA prefill shape (heads=32, seq=128, hd=64),
// KERNEL-only (resident buffers, devSync-bounded). Measure-first check on whether fused
// attention is the win the vLLM prefill gap predicts, before the e2e wiring.

func benchWMMAAttnResident(b *testing.B, heads, seq, hd int) {
	n := heads * seq * hd
	f := make([]float32, n)
	rng := rand.New(rand.NewSource(5))
	for i := range f {
		f[i] = float32(rng.NormFloat64()) * 0.25
	}
	dq, dk, dv := f16Upload(f), f16Upload(f), f16Upload(f)
	do := devAllocBytes(n * 4)
	if dq == nil || dk == nil || dv == nil || do == nil {
		b.Fatal("alloc failed")
	}
	defer devFree(dq)
	defer devFree(dk)
	defer devFree(dv)
	defer devFree(do)
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	wmmaAttnPtr(dq, dk, dv, do, heads, seq, hd, scale)
	devSync()
	b.ResetTimer()
	for range b.N {
		wmmaAttnPtr(dq, dk, dv, do, heads, seq, hd, scale)
	}
	devSync()
	b.StopTimer()
}

func benchGQAResident(b *testing.B, heads, seq, hd int) {
	width := heads * hd
	f := make([]float32, seq*width)
	rng := rand.New(rand.NewSource(5))
	for i := range f {
		f[i] = float32(rng.NormFloat64()) * 0.25
	}
	mk := func() *DeviceF32 {
		d, _ := NewDeviceF32(seq, width)
		d.UploadF32(f)
		return d
	}
	q, k, v := mk(), mk(), mk()
	defer q.Free()
	defer k.Free()
	defer v.Free()
	if o, err := GroupedQueryAttentionTF32(q, k, v, heads, heads, true); err != nil {
		b.Fatal(err)
	} else {
		o.Free()
	}
	devSync()
	b.ResetTimer()
	for range b.N {
		out, err := GroupedQueryAttentionTF32(q, k, v, heads, heads, true)
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
	devSync()
	b.StopTimer()
}

func BenchmarkAttnFused_32x128x64(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	benchWMMAAttnResident(b, 32, 128, 64)
}
func BenchmarkAttnGQA_32x128x64(b *testing.B) {
	if !Available() {
		b.Skip("no gpu")
	}
	benchGQAResident(b, 32, 128, 64)
}
