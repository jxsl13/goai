//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// The fused GQA WMMA attention (device buffers, [seq,heads·hd] layout) must match the existing
// cuBLAS-batched GroupedQueryAttentionTF32 within the f16-input envelope — the parity gate for
// the drop-in that replaces it in the prefill forward. TinyLlama GQA: 32 q-heads, 4 kv-heads,
// hd=64, seq=128.
func TestCUDAGQAWMMAParity(t *testing.T) {
	skipNoGPU(t)
	const qHeads, kvHeads, hd, seq = 32, 4, 64, 128
	rng := rand.New(rand.NewSource(31))
	mk := func(cols int) (*cuda.DeviceF32, []float32) {
		f := make([]float32, seq*cols)
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * 0.25
		}
		d, err := cuda.NewDeviceF32(seq, cols)
		must(t, err)
		must(t, d.UploadF32(f))
		return d, f
	}
	q, _ := mk(qHeads * hd)
	k, _ := mk(kvHeads * hd)
	v, _ := mk(kvHeads * hd)
	defer q.Free()
	defer k.Free()
	defer v.Free()

	ref, err := cuda.GroupedQueryAttentionTF32(q, k, v, qHeads, kvHeads, true)
	must(t, err)
	defer ref.Free()
	got, err := cuda.GroupedQueryAttentionWMMA(q, k, v, qHeads, kvHeads)
	must(t, err)
	defer got.Free()

	hr, err := ref.ToHost()
	must(t, err)
	hg, err := got.ToHost()
	must(t, err)
	var num, den float64
	for r := 0; r < seq; r++ {
		for c := 0; c < qHeads*hd; c++ {
			d := hg.AtF64(r, c) - hr.AtF64(r, c)
			num += d * d
			den += hr.AtF64(r, c) * hr.AtF64(r, c)
		}
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("fused GQA WMMA vs cuBLAS-batched GQA: norm-rel-RMS %.3e", relRMS)
	if relRMS > 2e-2 {
		t.Fatalf("fused GQA WMMA diverges: norm-rel-RMS %.3e", relRMS)
	}
}

func benchGQA(b *testing.B, fused bool, qHeads, kvHeads, hd, seq int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(5))
	mk := func(cols int) *cuda.DeviceF32 {
		f := make([]float32, seq*cols)
		for i := range f {
			f[i] = float32(rng.NormFloat64()) * 0.25
		}
		d, _ := cuda.NewDeviceF32(seq, cols)
		d.UploadF32(f)
		return d
	}
	q, k, v := mk(qHeads*hd), mk(kvHeads*hd), mk(kvHeads*hd)
	defer q.Free()
	defer k.Free()
	defer v.Free()
	run := func() (*cuda.DeviceF32, error) {
		if fused {
			return cuda.GroupedQueryAttentionWMMA(q, k, v, qHeads, kvHeads)
		}
		return cuda.GroupedQueryAttentionTF32(q, k, v, qHeads, kvHeads, true)
	}
	o, err := run()
	if err != nil {
		b.Fatal(err)
	}
	o.Free()
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		out, err := run()
		if err != nil {
			b.Fatal(err)
		}
		out.Free()
	}
	cuda.GraphSync()
	b.StopTimer()
}

func BenchmarkGQAWMMA_32_4_128(b *testing.B)   { benchGQA(b, true, 32, 4, 64, 128) }
func BenchmarkGQACublas_32_4_128(b *testing.B) { benchGQA(b, false, 32, 4, 64, 128) }
