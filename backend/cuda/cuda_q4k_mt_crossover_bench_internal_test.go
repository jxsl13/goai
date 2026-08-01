//go:build cuda && cgo && linux

package cuda

import "testing"

// benchQ4KCrossover compares the Q4_K decode GEMV (cu_qmatmul_q4k) vs the weight-read-once
// M-tiled GEMM (cu_qmatmul_q4k_mt) at small m — to find the recorder's MT routing threshold
// (QMatMulResidentQ4K gates m>=8, while the resident qmatmul method already uses m>=2). Q4_K MT
// is bit-identical to the GEMV, so lowering the gate is golden-safe. Timing only — the weight
// bytes are uninitialized (kernel decodes garbage; the runtime is unaffected).
func benchQ4KCrossover(b *testing.B, m, k, n int) {
	if !Available() {
		b.Skip("no gpu")
	}
	raw := make([]byte, (k*n/256)*144) // Q4_K = 144 bytes / 256 weights
	rq, err := NewResidentBQ4KFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	a, _ := NewDeviceF32(m, k)
	out, _ := NewDeviceF32(m, n)
	defer func() { rq.Free(); a.Free(); out.Free() }()
	tflops := func(b *testing.B) {
		b.ReportMetric(2*float64(m)*float64(k)*float64(n)*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	}
	b.Run("gemv", func(b *testing.B) {
		rq.qmatmulGEMVForBench(a, out)
		GraphSync()
		b.ResetTimer()
		for range b.N {
			rq.qmatmulGEMVForBench(a, out)
		}
		GraphSync()
		b.StopTimer()
		tflops(b)
	})
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
}

func BenchmarkQ4KCrossover_2x2048x2048(b *testing.B) { benchQ4KCrossover(b, 2, 2048, 2048) }
func BenchmarkQ4KCrossover_4x2048x2048(b *testing.B) { benchQ4KCrossover(b, 4, 2048, 2048) }
func BenchmarkQ4KCrossover_2x2048x5632(b *testing.B) { benchQ4KCrossover(b, 2, 2048, 5632) }
func BenchmarkQ4KCrossover_4x2048x5632(b *testing.B) { benchQ4KCrossover(b, 4, 2048, 5632) }
func BenchmarkQ4KCrossover_8x2048x2048(b *testing.B) { benchQ4KCrossover(b, 8, 2048, 2048) }
