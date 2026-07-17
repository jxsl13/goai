//go:build vulkan && cgo

package vulkan

import (
	"math"
	"math/rand"
	"testing"
)

// f32ToHalf converts to IEEE binary16 (round-to-nearest-even not needed for test data —
// truncation-level accuracy is inside the f16 tolerance the checks use).
func f32ToHalf(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16(bits>>16) & 0x8000
	exp := int32(bits>>23&0xff) - 127 + 15
	man := bits >> 13 & 0x3ff
	if exp <= 0 {
		return sign // flush tiny to zero (test data avoids subnormals)
	}
	if exp >= 0x1f {
		return sign | 0x7c00
	}
	return sign | uint16(exp)<<10 | uint16(man)
}

// Correctness: the coopmat GEMM must track an f64 host reference within the f16-input
// rounding envelope (inputs round to f16 ⇒ ~1e-3 rel per element; accumulate is f32).
func TestVulkanCoopmatGemmF16(t *testing.T) {
	if !Available() {
		t.Skip("no vulkan device")
	}
	if !HasCoopMat() {
		t.Skip("device lacks VK_KHR_cooperative_matrix")
	}
	const M, K, N = 32, 64, 128
	rng := rand.New(rand.NewSource(7))
	af := make([]float32, M*K)
	bf := make([]float32, K*N)
	ah := make([]uint16, M*K)
	bh := make([]uint16, K*N)
	for i := range af {
		af[i] = float32(rng.NormFloat64()) * 0.25
		ah[i] = f32ToHalf(af[i])
	}
	for i := range bf {
		bf[i] = float32(rng.NormFloat64()) * 0.25
		bh[i] = f32ToHalf(bf[i])
	}
	c := make([]float32, M*N)
	if err := CoopmatGemmF16(ah, bh, c, M, K, N); err != nil {
		t.Fatal(err)
	}
	var num, den float64
	var maxRel float64
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			var ref float64
			for k := 0; k < K; k++ {
				ref += float64(af[m*K+k]) * float64(bf[k*N+n])
			}
			d := float64(c[m*N+n]) - ref
			num += d * d
			den += ref * ref
			if r := math.Abs(d) / math.Max(math.Abs(ref), 1e-3); r > maxRel {
				maxRel = r
			}
		}
	}
	relRMS := math.Sqrt(num / den)
	t.Logf("coopmat GEMM vs f64 ref: norm-rel-RMS %.3e, maxRel %.3e", relRMS, maxRel)
	if relRMS > 5e-3 {
		t.Fatalf("coopmat GEMM off reference: norm-rel-RMS %.3e (f16-input envelope is ~1e-3)", relRMS)
	}
}

// benchCoopmat reports GFLOP/s for the probe kernel at prefill shapes; compare against the
// CUDA cuBLAS-f16 numbers (BenchmarkF16acc_*: gate/up 26513, qkv 20059 GFLOP/s) and this
// backend's own f32 tiled matmul.
func benchCoopmat(b *testing.B, m, k, n int) {
	if !Available() {
		b.Skip("no vulkan device")
	}
	if !HasCoopMat() {
		b.Skip("device lacks VK_KHR_cooperative_matrix")
	}
	rng := rand.New(rand.NewSource(11))
	ah := make([]uint16, m*k)
	bh := make([]uint16, k*n)
	for i := range ah {
		ah[i] = f32ToHalf(float32(rng.NormFloat64()) * 0.25)
	}
	for i := range bh {
		bh[i] = f32ToHalf(float32(rng.NormFloat64()) * 0.25)
	}
	c := make([]float32, m*n)
	if err := CoopmatGemmF16(ah, bh, c, m, k, n); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if err := CoopmatGemmF16(ah, bh, c, m, k, n); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(2.0*float64(m)*float64(n)*float64(k)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

// benchCoopmatRes measures the KERNEL rate: A/B/C device-resident, no per-call host transfer.
func benchCoopmatRes(b *testing.B, m, k, n int) {
	if !Available() {
		b.Skip("no vulkan device")
	}
	if !HasCoopMat() {
		b.Skip("device lacks VK_KHR_cooperative_matrix")
	}
	rng := rand.New(rand.NewSource(11))
	ah := make([]uint16, m*k)
	bh := make([]uint16, k*n)
	for i := range ah {
		ah[i] = f32ToHalf(float32(rng.NormFloat64()) * 0.25)
	}
	for i := range bh {
		bh[i] = f32ToHalf(float32(rng.NormFloat64()) * 0.25)
	}
	r, err := NewCoopmatResident(ah, bh, m, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer r.Free()
	if err := r.Gemm(); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if err := r.Gemm(); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
	b.ReportMetric(2.0*float64(m)*float64(n)*float64(k)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkCoopmatRes_qkv(b *testing.B)    { benchCoopmatRes(b, 128, 2048, 2048) }
func BenchmarkCoopmatRes_gateup(b *testing.B) { benchCoopmatRes(b, 128, 2048, 5632) }
func BenchmarkCoopmatRes_down(b *testing.B)   { benchCoopmatRes(b, 128, 5632, 2048) }

func BenchmarkCoopmat_qkv(b *testing.B)    { benchCoopmat(b, 128, 2048, 2048) }
func BenchmarkCoopmat_gateup(b *testing.B) { benchCoopmat(b, 128, 2048, 5632) }
func BenchmarkCoopmat_down(b *testing.B)   { benchCoopmat(b, 128, 5632, 2048) }
