//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// rawIQ3XXS builds N rows of sbs=K/256 valid IQ3_XXS super-blocks (98 B: f16 d + 64 grid-index
// bytes + 8 ksigns/scale words). Any bytes are valid; the kernel and GEMV decode the same bytes
// through the same 256×4 grid.
func rawIQ3XXS(K, N int, rng *rand.Rand) []byte {
	sbs := K / 256
	raw := make([]byte, N*sbs*98)
	for i := 0; i < N*sbs; i++ {
		off := i * 98
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1))
		for pos := 0; pos < 64; pos++ {
			raw[off+2+pos] = byte(rng.Intn(256))
		}
		for g := 0; g < 8; g++ {
			binary.LittleEndian.PutUint32(raw[off+66+g*4:], rng.Uint32())
		}
	}
	return raw
}

// The IQ3_XXS M>1 path routes through the weight-read-once M-tiled kernel (cu_qmatmul_iq3xxs_mt).
// Invariant: reproduces the per-(m,n) GEMV EXACTLY (grid-decode hoisted, arithmetic verbatim) →
// max abs diff ~0. M=13 = full MT=8 tile + ragged tail.
func TestCUDAIQ3XXSMatMulMTParity(t *testing.T) {
	skipNoGPU(t)
	const K, N, M = 512, 48, 7 // M in [2,8): validates the newly-routed MT path (partial tile)
	rng := rand.New(rand.NewSource(91))
	raw := rawIQ3XXS(K, N, rng)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := cuda.NewResidentBIQ3XXSFromBlocks(raw, K, N)
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
	t.Logf("IQ3_XXS M-tiled (M=%d) vs per-row GEMV: max abs diff %.3g", M, maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("IQ3_XXS M-tiled kernel diverges from the GEMV: max abs %.3g", maxAbs)
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

func benchIQ3XXSM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(97))
	raw := rawIQ3XXS(k, n, rng)
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ3XXSFromBlocks(raw, k, n)
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

func BenchmarkIQ3XXSM16_2048(b *testing.B)      { benchIQ3XXSM(b, 16, 2048, 2048) }
func BenchmarkIQ3XXSM32_2048(b *testing.B)      { benchIQ3XXSM(b, 32, 2048, 2048) }
func BenchmarkIQ3XXSM64_2048x5632(b *testing.B) { benchIQ3XXSM(b, 64, 2048, 5632) }

func BenchmarkIQ3XXSM2_5632(b *testing.B) { benchIQ3XXSM(b, 2, 2048, 5632) }
func BenchmarkIQ3XXSM4_5632(b *testing.B) { benchIQ3XXSM(b, 4, 2048, 5632) }
