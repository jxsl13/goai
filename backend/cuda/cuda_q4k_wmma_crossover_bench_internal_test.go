//go:build cuda && cgo && linux

package cuda

import "testing"

// benchQ4KWMMACrossover compares the Q4_K M-tiled decode GEMV (cu_qmatmul_q4k_mt, the current
// recorder prefill kernel) vs the tensor-core WMMA path (rec.q4kPrefillWMMA) at a range of m — to
// pin q4kWMMAThreshold, the row count where WMMA starts beating MT. Timing only; the weight bytes
// are uninitialized (the kernels decode garbage, runtime unaffected).
func benchQ4KWMMACrossover(b *testing.B, m, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*144) // Q4_K = 144 bytes / 256 weights
	rq, err := NewResidentBQ4KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	rec, err := NewRecorder()
	if err != nil {
		b.Fatal(err)
	}
	a, _ := NewDeviceF32(m, k)
	out, _ := NewDeviceF32(m, n)
	defer func() { rq.Free(); a.Free(); out.Free(); rec.Free() }()
	tflops := func(b *testing.B) {
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	}
	b.Run("mt", func(b *testing.B) {
		rq.qmatmulMTForBench(a, out)
		GraphSync()
		b.ResetTimer()
		for range b.N {
			rq.qmatmulMTForBench(a, out)
		}
		GraphSync()
		b.StopTimer()
		tflops(b)
	})
	b.Run("wmma", func(b *testing.B) {
		if err := rec.q4kPrefillWMMA(a, rq, out, m); err != nil {
			b.Skipf("wmma: %v", err)
		}
		GraphSync()
		b.ResetTimer()
		for range b.N {
			rec.q4kPrefillWMMA(a, rq, out, m)
		}
		GraphSync()
		b.StopTimer()
		tflops(b)
	})
}

func BenchmarkQ4KWMMACrossover_16x4096x4096(b *testing.B)  { benchQ4KWMMACrossover(b, 16, 4096, 4096) }
func BenchmarkQ4KWMMACrossover_32x4096x4096(b *testing.B)  { benchQ4KWMMACrossover(b, 32, 4096, 4096) }
func BenchmarkQ4KWMMACrossover_48x4096x4096(b *testing.B)  { benchQ4KWMMACrossover(b, 48, 4096, 4096) }
func BenchmarkQ4KWMMACrossover_64x4096x4096(b *testing.B)  { benchQ4KWMMACrossover(b, 64, 4096, 4096) }
func BenchmarkQ4KWMMACrossover_128x4096x4096(b *testing.B) { benchQ4KWMMACrossover(b, 128, 4096, 4096) }
func BenchmarkQ4KWMMACrossover_256x4096x4096(b *testing.B) { benchQ4KWMMACrossover(b, 256, 4096, 4096) }
