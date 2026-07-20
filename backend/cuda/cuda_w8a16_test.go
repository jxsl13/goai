//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"

	"github.com/jxsl13/goai/backend/cuda"
)

// quantW8Col does per-column symmetric int8 quant of a row-major [k,n] f32 weight: scale[n] =
// maxabs(col n)/127, w8 = round(w/scale). Returns the int8 weights, the f32 scales, and the
// dequantized f32 (w8*scale) so a reference GEMM can use the EXACT values the kernel dequantizes to.
func quantW8Col(wf []float32, k, n int) (w8 []int8, scale []float32, wdq []float32) {
	w8 = make([]int8, k*n)
	scale = make([]float32, n)
	wdq = make([]float32, k*n)
	for c := 0; c < n; c++ {
		mx := float32(0)
		for r := 0; r < k; r++ {
			a := wf[r*n+c]
			if a < 0 {
				a = -a
			}
			if a > mx {
				mx = a
			}
		}
		s := mx / 127
		if s == 0 {
			s = 1
		}
		scale[c] = s
		for r := 0; r < k; r++ {
			q := int32(math.Round(float64(wf[r*n+c] / s)))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			w8[r*n+c] = int8(q)
			wdq[r*n+c] = float32(q) * s
		}
	}
	return
}

// TestGemmW8A16Parity: the dequant-in-tile W8A16 mma GEMM must match a cuBLAS f16 GEMM on the SAME
// dequantized weights, within f16-accumulate tolerance. Validates v1 correctness of the low-batch
// serving lever (int8 weight read → f16 mma). M=64 = a decode shape.
func TestGemmW8A16Parity(t *testing.T) {
	if !cuda.Available() {
		t.Skip("no gpu")
	}
	const M, K, N = 64, 2048, 2048
	rng := rand.New(rand.NewSource(7))
	af := make([]float32, M*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.1
	}
	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.02
	}
	w8, scale, wdq := quantW8Col(wf, K, N)

	// device f16 activation
	adf, _ := cuda.NewDeviceF32(M, K)
	adf.UploadF32(af)
	a16 := cuda.AllocU16(M * K)
	cuda.CvtF32ToF16(a16, adf.DevPtr(), M*K)
	adf.Free()
	defer cuda.FreeDev(a16)

	// device f16 weight = dequant(w8) — the reference operand (exact values the kernel produces)
	wdf, _ := cuda.NewDeviceF32(K, N)
	wdf.UploadF32(wdq)
	w16 := cuda.AllocU16(K * N)
	cuda.CvtF32ToF16(w16, wdf.DevPtr(), K*N)
	wdf.Free()
	defer cuda.FreeDev(w16)

	// device int8 weight + f32 scale for the W8A16 kernel
	w8dev := cuda.UploadI8(w8)
	defer cuda.FreeDev(w8dev)
	scf, _ := cuda.NewDeviceF32(1, N)
	scf.UploadF32(scale)
	defer scf.Free()

	// reference: cuBLAS f16 GEMM on the dequantized weights
	ref16 := cuda.AllocU16(M * N)
	if rc := cuda.GemmF16Pure(a16, w16, ref16, M, K, N); rc != 0 {
		t.Fatalf("ref GemmF16Pure rc=%d", rc)
	}
	defer cuda.FreeDev(ref16)

	// W8A16 kernel
	out16 := cuda.AllocU16(M * N)
	if rc := cuda.GemmW8A16(a16, w8dev, unsafe.Pointer(scf.DevPtr()), out16, M, K, N); rc != 0 {
		t.Fatalf("GemmW8A16 rc=%d", rc)
	}
	defer cuda.FreeDev(out16)

	dl := func(p unsafe.Pointer) []float32 {
		d, _ := cuda.NewDeviceF32(M, N)
		cuda.CvtF16ToF32(d.DevPtr(), p, M*N)
		out := make([]float32, M*N)
		d.DownloadF32(out)
		d.Free()
		return out
	}
	// tiled variant (shared-mem staging)
	outT := cuda.AllocU16(M * N)
	if rc := cuda.GemmW8A16T(a16, w8dev, unsafe.Pointer(scf.DevPtr()), outT, M, K, N); rc != 0 {
		t.Fatalf("GemmW8A16T rc=%d", rc)
	}
	defer cuda.FreeDev(outT)
	// BM-spanning variant
	outB := cuda.AllocU16(M * N)
	if rc := cuda.GemmW8A16B(a16, w8dev, unsafe.Pointer(scf.DevPtr()), outB, M, K, N); rc != 0 {
		t.Fatalf("GemmW8A16B rc=%d", rc)
	}
	defer cuda.FreeDev(outB)
	// double-buffered (cp.async) variant
	outD := cuda.AllocU16(M * N)
	if rc := cuda.GemmW8A16D(a16, w8dev, unsafe.Pointer(scf.DevPtr()), outD, M, K, N); rc != 0 {
		t.Fatalf("GemmW8A16D rc=%d", rc)
	}
	defer cuda.FreeDev(outD)

	ref := dl(ref16)
	relOf := func(p unsafe.Pointer) float64 {
		got := dl(p)
		var num, den float64
		for i := range ref {
			d := float64(ref[i] - got[i])
			num += d * d
			den += float64(ref[i]) * float64(ref[i])
		}
		return math.Sqrt(num / den)
	}
	rel := relOf(out16)
	relT := relOf(outT)
	relB := relOf(outB)
	relD := relOf(outD)
	if rel > 3e-2 || relT > 3e-2 || relB > 3e-2 || relD > 3e-2 {
		t.Fatalf("W8A16 kernel bug — v1 %.3e tiled %.3e bmspan %.3e dbuf %.3e", rel, relT, relB, relD)
	}
	t.Logf("W8A16 CORRECT — v1 %.3e, tiled %.3e, BM-span %.3e, dbuf %.3e vs cuBLAS-f16", rel, relT, relB, relD)
}

// benchmark v1 W8A16 vs cuBLAS f16 at a decode shape — establishes the optimization baseline. v1 is
// naive (uncoalesced per-thread int8 W load, no shared tiling), so expect it slower than cuBLAS
// initially; the target (roofline ~2x at M=64) needs the coalesced-staging follow-up.
func benchW8A16vsF16(b *testing.B, M, K, N int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(3))
	af := make([]float32, M*K)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.1
	}
	wf := make([]float32, K*N)
	for i := range wf {
		wf[i] = float32(rng.NormFloat64()) * 0.02
	}
	w8, scale, wdq := quantW8Col(wf, K, N)
	adf, _ := cuda.NewDeviceF32(M, K)
	adf.UploadF32(af)
	a16 := cuda.AllocU16(M * K)
	cuda.CvtF32ToF16(a16, adf.DevPtr(), M*K)
	adf.Free()
	defer cuda.FreeDev(a16)
	wdf, _ := cuda.NewDeviceF32(K, N)
	wdf.UploadF32(wdq)
	w16 := cuda.AllocU16(K * N)
	cuda.CvtF32ToF16(w16, wdf.DevPtr(), K*N)
	wdf.Free()
	defer cuda.FreeDev(w16)
	w8dev := cuda.UploadI8(w8)
	defer cuda.FreeDev(w8dev)
	scf, _ := cuda.NewDeviceF32(1, N)
	scf.UploadF32(scale)
	defer scf.Free()
	c16 := cuda.AllocU16(M * N)
	defer cuda.FreeDev(c16)
	gf := 2 * float64(M) * float64(K) * float64(N)

	b.Run("f16", func(b *testing.B) {
		cuda.GemmF16Pure(a16, w16, c16, M, K, N)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmF16Pure(a16, w16, c16, M, K, N)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(gf*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("w8a16", func(b *testing.B) {
		cuda.GemmW8A16(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmW8A16(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(gf*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("w8a16t", func(b *testing.B) {
		cuda.GemmW8A16T(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmW8A16T(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(gf*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("w8a16b", func(b *testing.B) {
		cuda.GemmW8A16B(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmW8A16B(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(gf*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
	b.Run("w8a16d", func(b *testing.B) {
		cuda.GemmW8A16D(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		cuda.GraphSync()
		b.ResetTimer()
		for range b.N {
			cuda.GemmW8A16D(a16, w8dev, unsafe.Pointer(scf.DevPtr()), c16, M, K, N)
		}
		cuda.GraphSync()
		b.StopTimer()
		b.ReportMetric(gf*float64(b.N)/b.Elapsed().Seconds()/1e12, "TFLOP/s")
	})
}

func BenchmarkW8A16_64x2048x2048(b *testing.B)  { benchW8A16vsF16(b, 64, 2048, 2048) }
func BenchmarkW8A16_128x2048x2048(b *testing.B) { benchW8A16vsF16(b, 128, 2048, 2048) }
