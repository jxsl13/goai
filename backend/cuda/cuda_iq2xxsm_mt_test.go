//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// rawIQ2XXS builds N rows of sbs=K/256 valid IQ2_XXS super-blocks (66 B: f16 d + 8 pairs of
// qs0 grid-index / qs1 ksigns+scale). Any 32-bit qs words are valid; the kernel and the GEMV
// decode the same bytes through the same 256×8 grid.
func rawIQ2XXS(K, N int, rng *rand.Rand) []byte {
	sbs := K / 256
	raw := make([]byte, N*sbs*66)
	for i := 0; i < N*sbs; i++ {
		off := i * 66
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1))
		for pair := 0; pair < 8; pair++ {
			binary.LittleEndian.PutUint32(raw[off+2+pair*8:], rng.Uint32())
			binary.LittleEndian.PutUint32(raw[off+2+pair*8+4:], rng.Uint32())
		}
	}
	return raw
}

// The IQ2_XXS M>1 path routes through the weight-read-once M-tiled kernel (cu_qmatmul_iq2xxs_mt).
// Invariant: reproduces the per-(m,n) GEMV EXACTLY (grid-decode hoisted, arithmetic verbatim) →
// max abs diff ~0. M=13 = full MT=8 tile + ragged tail.
func TestCUDAIQ2XXSMatMulMTParity(t *testing.T) {
	skipNoGPU(t)
	const K, N, M = 512, 48, 13
	rng := rand.New(rand.NewSource(61))
	raw := rawIQ2XXS(K, N, rng)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := cuda.NewResidentBIQ2XXSFromBlocks(raw, K, N)
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
	t.Logf("IQ2_XXS M-tiled (M=%d) vs per-row GEMV: max abs diff %.3g", M, maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("IQ2_XXS M-tiled kernel diverges from the GEMV: max abs %.3g", maxAbs)
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

func benchIQ2XXSM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(67))
	raw := rawIQ2XXS(k, n, rng)
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ2XXSFromBlocks(raw, k, n)
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

func BenchmarkIQ2XXSM16_2048(b *testing.B)      { benchIQ2XXSM(b, 16, 2048, 2048) }
func BenchmarkIQ2XXSM32_2048(b *testing.B)      { benchIQ2XXSM(b, 32, 2048, 2048) }
func BenchmarkIQ2XXSM64_2048x5632(b *testing.B) { benchIQ2XXSM(b, 64, 2048, 5632) }
