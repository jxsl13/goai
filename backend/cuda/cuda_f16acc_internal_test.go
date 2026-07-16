//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"testing"
)

// f16-accumulate accuracy vs f32-accumulate (the current prefill quality baseline). Both use
// f16 weights + f16 activation; only the accumulate precision differs. norm-rel-RMS over the
// f32 output sizes whether the ~1.5-2× f16-accumulate speedup is usable (real-model gate next).
func TestCUDAF16AccAccuracy(t *testing.T) {
	for _, s := range []struct{ m, k, n int }{{8, 2048, 2048}, {8, 5632, 2048}, {8, 2048, 5632}} {
		aF := make([]float32, s.m*s.k)
		for i := range aF {
			aF[i] = float32(math.Sin(float64(i) * 0.017))
		}
		wF := make([]float32, s.n*s.k)
		for i := range wF {
			wF[i] = float32(math.Cos(float64(i) * 0.013))
		}
		da, dw := f32Upload(aF), f16Upload(wF)
		c32, c16 := devAllocBytes(s.m*s.n*4), devAllocBytes(s.m*s.n*2)
		if matmulF16w(da, dw, c32, s.m, s.k, s.n) != 0 || matmulF16acc16(da, dw, c16, s.m, s.k, s.n) != 0 {
			t.Fatal("gemm failed")
		}
		devSync()
		ref := make([]float32, s.m*s.n)
		gotH := make([]uint16, s.m*s.n)
		downloadF32(c32, ref)
		downloadU16(c16, gotH)
		var num, den float64
		for i := range ref {
			g := f16ToF32(gotH[i])
			num += float64(g-ref[i]) * float64(g-ref[i])
			den += float64(ref[i]) * float64(ref[i])
		}
		devFree(da)
		devFree(dw)
		devFree(c32)
		devFree(c16)
		t.Logf("K=%d: f16-acc vs f32-acc norm-rel-RMS %.3e", s.k, math.Sqrt(num/den))
	}
}

func f16ToF32(h uint16) float32 {
	s := uint32(h&0x8000) << 16
	e := uint32(h>>10) & 0x1f
	m := uint32(h & 0x3ff)
	if e == 0 {
		if m == 0 {
			return math.Float32frombits(s)
		}
		for m&0x400 == 0 {
			m <<= 1
			e--
		}
		e++
		m &= 0x3ff
	} else if e == 0x1f {
		return math.Float32frombits(s | 0x7f800000 | m<<13)
	}
	return math.Float32frombits(s | (e+112)<<23 | m<<13)
}

// f16 accumulate (COMPUTE_16F) vs f32 accumulate (COMPUTE_32F) — the GeForce half-rate lever.
func benchF16acc(b *testing.B, m, k, n int, acc16 bool) {
	cbytes := m * n * 4
	if acc16 {
		cbytes = m * n * 2 // f16 output
	}
	dc := devAllocBytes(cbytes)
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
		var rc int
		if acc16 {
			rc = matmulF16acc16(da, dw, dc, m, k, n)
		} else {
			rc = matmulF16w(da, dw, dc, m, k, n)
		}
		if rc != 0 {
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

func BenchmarkF16acc_gateup_a32(b *testing.B) { benchF16acc(b, 128, 2048, 5632, false) }
func BenchmarkF16acc_gateup_a16(b *testing.B) { benchF16acc(b, 128, 2048, 5632, true) }
func BenchmarkF16acc_down_a32(b *testing.B)   { benchF16acc(b, 128, 5632, 2048, false) }
func BenchmarkF16acc_down_a16(b *testing.B)   { benchF16acc(b, 128, 5632, 2048, true) }
func BenchmarkF16acc_qkv_a32(b *testing.B)    { benchF16acc(b, 128, 2048, 2048, false) }
func BenchmarkF16acc_qkv_a16(b *testing.B)    { benchF16acc(b, 128, 2048, 2048, true) }
func BenchmarkF16acc_bigM_a32(b *testing.B)   { benchF16acc(b, 2048, 4096, 4096, false) }
func BenchmarkF16acc_bigM_a16(b *testing.B)   { benchF16acc(b, 2048, 4096, 4096, true) }
