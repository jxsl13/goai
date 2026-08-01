//go:build cuda && cgo && linux

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// benchAttnSoftmaxDirect measures cu_attn_softmax in isolation (no QKᵀ/PV) at a
// prefill scores shape [heads*seq, seq]. The base kernel does per-element FP64
// (max*scale, exp arg, sum) — FP64 runs at 1/64 the FP32 rate on GA106. GB/s
// far below the ~360 peak flags the FP64 cap and the shared-cache opportunity.
func benchAttnSoftmaxDirect(b *testing.B, heads, seq int) {
	if !Available() {
		b.Skip("no gpu")
	}
	rows, cols := heads*seq, seq
	d, err := NewDeviceF32(rows, cols)
	if err != nil {
		b.Fatal(err)
	}
	defer d.Free()
	f := make([]float32, rows*cols)
	rng := rand.New(rand.NewSource(7))
	for i := range f {
		f[i] = float32(rng.NormFloat64()) * 0.25
	}
	d.UploadF32(f)
	scale := float32(1.0 / math.Sqrt(float64(cols)))
	attnSoftmaxPtr(d.ptr, rows, cols, scale, 0, seq)
	devSync()
	b.ResetTimer()
	for range b.N {
		attnSoftmaxPtr(d.ptr, rows, cols, scale, 0, seq)
	}
	devSync()
	b.StopTimer()
	// causal softmax touches ~half the [rows,cols] triangle; report read+write of the full matrix as a floor
	b.ReportMetric(2*4*float64(rows)*float64(cols)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GB/s")
}

func BenchmarkAttnSM_8x1024(b *testing.B) { benchAttnSoftmaxDirect(b, 8, 1024) }
func BenchmarkAttnSM_8x2048(b *testing.B) { benchAttnSoftmaxDirect(b, 8, 2048) }
