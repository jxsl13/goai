//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// rawIQ3S builds N rows of sbs=K/256 valid IQ3_S super-blocks (110 B: f16 d + 64 qs + 8 qh +
// 32 sign bytes + 4 sub-scale bytes). Any bytes are valid; the kernel and GEMV decode the same
// bytes through the same 512×4 grid.
func rawIQ3S(K, N int, rng *rand.Rand) []byte {
	sbs := K / 256
	raw := make([]byte, N*sbs*110)
	for i := 0; i < N*sbs; i++ {
		off := i * 110
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1))
		for b := 0; b < 64; b++ {
			raw[off+2+b] = byte(rng.Intn(256))
		}
		for b := 0; b < 8; b++ {
			raw[off+66+b] = byte(rng.Intn(256))
		}
		for b := 0; b < 32; b++ {
			raw[off+74+b] = byte(rng.Intn(256))
		}
		for b := 0; b < 4; b++ {
			raw[off+106+b] = byte(rng.Intn(256))
		}
	}
	return raw
}

// The IQ3_S M>1 path routes through the weight-read-once M-tiled kernel (cu_qmatmul_iq3s_mt).
// Invariant: reproduces the per-(m,n) GEMV EXACTLY (grid-decode hoisted, arithmetic verbatim) →
// max abs diff ~0. M=13 = full MT=8 tile + ragged tail.
func TestCUDAIQ3SMatMulMTParity(t *testing.T) {
	skipNoGPU(t)
	const K, N, M = 512, 48, 13
	rng := rand.New(rand.NewSource(71))
	raw := rawIQ3S(K, N, rng)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := cuda.NewResidentBIQ3SFromBlocks(raw, K, N)
	must(t, err)
	defer rq.Free()
	da, err := cuda.NewDeviceF32(M, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dout, err := cuda.NewDeviceF32(M, N)
	must(t, err)
	defer dout.Free()
	must(t, rq.QMatMulInto(da, dout)) // M=13 >= 8 → MT kernel
	got, err := dout.ToHost()
	must(t, err)

	gd, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer gd.Free()
	gout, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer gout.Free()
	var maxAbs float64
	for m := 0; m < M; m++ {
		must(t, gd.UploadF32(a[m*K:(m+1)*K]))
		must(t, rq.QMatMulInto(gd, gout)) // M=1 → GEMV
		gv, err := gout.ToHost()
		must(t, err)
		for n := 0; n < N; n++ {
			d := math.Abs(got.AtF64(m, n) - gv.AtF64(0, n))
			if d > maxAbs {
				maxAbs = d
			}
		}
	}
	t.Logf("IQ3_S M-tiled (M=%d) vs per-row GEMV: max abs diff %.3g", M, maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("IQ3_S M-tiled kernel diverges from the GEMV: max abs %.3g", maxAbs)
	}

	// beta=1 residual fuse across the batch
	must(t, rq.QMatMulAccInto(da, dout))
	got2, err := dout.ToHost()
	must(t, err)
	for m := 0; m < M; m++ {
		for n := 0; n < N; n++ {
			want := 2 * got.AtF64(m, n)
			if math.Abs(got2.AtF64(m, n)-want) > 1e-3*math.Max(math.Abs(want), 1) {
				t.Fatalf("MT QMatMulAccInto beta=1 wrong at [%d,%d]: %g want %g", m, n, got2.AtF64(m, n), want)
			}
		}
	}
}

func benchIQ3SM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(73))
	raw := rawIQ3S(k, n, rng)
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ3SFromBlocks(raw, k, n)
	if err != nil {
		b.Fatal(err)
	}
	defer rq.Free()
	da, _ := cuda.NewDeviceF32(m, k)
	defer da.Free()
	da.UploadF32(a)
	out, _ := cuda.NewDeviceF32(m, n)
	defer out.Free()
	rq.QMatMulInto(da, out)
	cuda.GraphSync()
	b.ResetTimer()
	for range b.N {
		rq.QMatMulInto(da, out)
	}
	cuda.GraphSync()
	b.StopTimer()
	b.ReportMetric(2*float64(m)*float64(k)*float64(n)/(b.Elapsed().Seconds()/float64(b.N))/1e9, "GFLOP/s")
}

func BenchmarkIQ3SM16_2048(b *testing.B)      { benchIQ3SM(b, 16, 2048, 2048) }
func BenchmarkIQ3SM32_2048(b *testing.B)      { benchIQ3SM(b, 32, 2048, 2048) }
func BenchmarkIQ3SM64_2048x5632(b *testing.B) { benchIQ3SM(b, 64, 2048, 5632) }
func BenchmarkIQ3SM64_5632x2048(b *testing.B) { benchIQ3SM(b, 64, 5632, 2048) } // ffn_down (deep-K), M=64
