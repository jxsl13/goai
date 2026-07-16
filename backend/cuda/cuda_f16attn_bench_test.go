//go:build cuda && cgo && (linux || windows)

package cuda

import "testing"

// Tw65 measure-first probe: does f16 cublasGemmEx (tensor cores) beat f32 cublasSgemm at the
// ATTENTION GEMM shapes, before converting cu_gqa_scores/cu_gqa_out? The Tw61 prefill win was
// on large-K FFN GEMMs (K=2048-5632); attention's tiny K matters:
//   QKᵀ  per head: M=seqQ, N=seqKV, K=hd  (K=64 — only 4 MMA tiles, may not amortize)
//   PV   per head: M=seqQ, N=hd,   K=seqKV (K large, N=64 — the friendlier one)
// benchAttnF32 = f32 device Sgemm (matmulF32ddd); benchAttnF16 = f16 tensor-core (matmulF16w).
// Compare GFLOP/s across the pair for each shape.

func benchAttnF32(b *testing.B, m, k, n int) {
	dc := devAllocBytes(m * n * 4)
	if dc == nil {
		b.Fatal("alloc C failed")
	}
	defer devFree(dc)
	aF := make([]float32, m*k)
	for i := range aF {
		aF[i] = 0.01 * float32(i%17)
	}
	bF := make([]float32, k*n)
	for i := range bF {
		bF[i] = 0.01 * float32(i%19)
	}
	da, db := f32Upload(aF), f32Upload(bF)
	if da == nil || db == nil {
		b.Fatal("upload failed")
	}
	defer devFree(da)
	defer devFree(db)
	run := func() {
		if rc := matmulF32ddd(da, db, dc, m, k, n); rc != 0 {
			b.Fatalf("rc=%d", rc)
		}
	}
	run()
	devSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	devSync()
	b.StopTimer()
	b.ReportMetric(2.0*float64(m)*float64(n)*float64(k)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func benchAttnF16(b *testing.B, m, k, n int) {
	dc := devAllocBytes(m * n * 4)
	if dc == nil {
		b.Fatal("alloc C failed")
	}
	defer devFree(dc)
	aF := make([]float32, m*k)
	for i := range aF {
		aF[i] = 0.01 * float32(i%17)
	}
	wF := make([]float32, n*k)
	for i := range wF {
		wF[i] = 0.01 * float32(i%19)
	}
	da, dw := f32Upload(aF), f16Upload(wF)
	if da == nil || dw == nil {
		b.Fatal("upload failed")
	}
	defer devFree(da)
	defer devFree(dw)
	run := func() {
		if rc := matmulF16w(da, dw, dc, m, k, n); rc != 0 {
			b.Fatalf("rc=%d", rc)
		}
	}
	run()
	devSync()
	b.ResetTimer()
	for range b.N {
		run()
	}
	devSync()
	b.StopTimer()
	b.ReportMetric(2.0*float64(m)*float64(n)*float64(k)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

// QKᵀ shape (K=hd=64) at growing context — the amortization question.
func BenchmarkAttnQKt_512_f32(b *testing.B)  { benchAttnF32(b, 512, 64, 512) }
func BenchmarkAttnQKt_512_f16(b *testing.B)  { benchAttnF16(b, 512, 64, 512) }
func BenchmarkAttnQKt_1024_f32(b *testing.B) { benchAttnF32(b, 1024, 64, 1024) }
func BenchmarkAttnQKt_1024_f16(b *testing.B) { benchAttnF16(b, 1024, 64, 1024) }

// PV shape (K=seqKV large, N=hd=64).
func BenchmarkAttnPV_512_f32(b *testing.B)  { benchAttnF32(b, 512, 512, 64) }
func BenchmarkAttnPV_512_f16(b *testing.B)  { benchAttnF16(b, 512, 512, 64) }
func BenchmarkAttnPV_1024_f32(b *testing.B) { benchAttnF32(b, 1024, 1024, 64) }
func BenchmarkAttnPV_1024_f16(b *testing.B) { benchAttnF16(b, 1024, 1024, 64) }
