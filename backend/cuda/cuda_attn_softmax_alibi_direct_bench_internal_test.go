//go:build cuda && cgo && linux

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// benchAttnSoftmaxAlibiDirect measures cu_attn_softmax_alibi (ALiBi position-bias attention
// softmax, BLOOM/MPT/Jina-BERT) in isolation at a causal prefill shape [heads*seq, seq].
// offset=0 → row r (query qi=r%seq) attends keys 0..qi (causal). The base kernel does per-element
// FP64 and recomputes the biased score x*scale+slope·(j-qabs) twice; the _c twin caches it once
// in shared memory and runs FP32 per-element (FP64 only in the reductions).
func benchAttnSoftmaxAlibiDirect(b *testing.B, heads, seq int) {
	if !Available() {
		b.Skip("no gpu")
	}
	rows, cols := heads*seq, seq
	d, err := NewDeviceF32(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	slopes, err := NewDeviceF32(1, heads)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { d.Free(); slopes.Free() }()
	f := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(7))
	for i := range f {
		f[i] = float32(rng.NormFloat64()) * 0.25
	}
	d.UploadF32(f)
	sl := make([]float32, heads)
	for h := range sl {
		sl[h] = float32(math.Pow(2, -8.0*float64(h+1)/float64(heads))) // standard ALiBi slopes
	}
	slopes.UploadF32(sl)
	scale := float32(1.0 / math.Sqrt(float64(cols)))
	attnSoftmaxAlibiPtr(d.ptr, slopes.ptr, rows, cols, scale, 0, seq)
	devSync()
	b.ResetTimer()
	for range b.N {
		attnSoftmaxAlibiPtr(d.ptr, slopes.ptr, rows, cols, scale, 0, seq)
	}
	devSync()
	b.StopTimer()
	b.ReportMetric(2*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkAttnSMAlibi_8x1024(b *testing.B) { benchAttnSoftmaxAlibiDirect(b, 8, 1024) }
func BenchmarkAttnSMAlibi_8x2048(b *testing.B) { benchAttnSoftmaxAlibiDirect(b, 8, 2048) }
