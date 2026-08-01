//go:build cuda && cgo && linux

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// benchAttnSoftmaxBiasDirect measures cu_attn_softmax_bias (T5 relative-position-bias
// attention softmax) in isolation at a bidirectional prefill shape [heads*seq, seq].
// offset=cols makes every row attend all keys (full seq², T5 is non-causal). The base
// kernel does per-element FP64 and recomputes the biased score x*scale+bias[j] twice.
func benchAttnSoftmaxBiasDirect(b *testing.B, heads, seq int) {
	if !Available() {
		b.Skip("no gpu")
	}
	rows, cols := heads*seq, seq
	d, err := NewDeviceF32(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	bias, err := NewDeviceF32(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer func() { d.Free(); bias.Free() }()
	f := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(7))
	for i := range f {
		f[i] = float32(rng.NormFloat64()) * 0.25
	}
	d.UploadF32(f)
	bf := make([]float32, rows*cols)
	for i := range bf {
		bf[i] = float32(rng.NormFloat64()) * 0.3
	}
	bias.UploadF32(bf)
	scale := float32(1.0 / math.Sqrt(float64(cols)))
	attnSoftmaxBiasPtr(d.ptr, bias.ptr, rows, cols, scale, cols, seq) // offset=cols → full bidirectional rows
	devSync()
	b.ResetTimer()
	for range b.N {
		attnSoftmaxBiasPtr(d.ptr, bias.ptr, rows, cols, scale, cols, seq)
	}
	devSync()
	b.StopTimer()
	// read x + read bias + write x ≈ 3×4 bytes/element min traffic
	b.ReportMetric(3*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkAttnSMBias_8x1024(b *testing.B) { benchAttnSoftmaxBiasDirect(b, 8, 1024) }
func BenchmarkAttnSMBias_8x2048(b *testing.B) { benchAttnSoftmaxBiasDirect(b, 8, 2048) }
