//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
)

// rawIQ4XS builds N rows of sbs=K/256 valid IQ4_XS super-blocks (any nibble is a valid
// codebook index; any finite d/scales work — the kernel and the GEMV decode the same bytes).
func rawIQ4XS(K, N int, rng *rand.Rand) []byte {
	sbs := K / 256
	raw := make([]byte, N*sbs*136)
	for i := 0; i < N*sbs; i++ {
		off := i * 136
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		binary.LittleEndian.PutUint16(raw[off+2:], uint16(rng.Intn(0x10000)))             // scales_h
		for j := 0; j < 4; j++ {
			raw[off+4+j] = byte(rng.Intn(256)) // scales_l
		}
		for j := 0; j < 128; j++ {
			raw[off+8+j] = byte(rng.Intn(256)) // qs nibbles
		}
	}
	return raw
}

// The IQ4_XS M>1 path routes through the weight-read-once M-tiled kernel (cu_qmatmul_iq4xs_mt).
// Invariant: reproduces the per-(m,n) GEMV EXACTLY (codebook decode hoisted, arithmetic verbatim)
// → max abs diff ~0. M=13 = full MT=8 tile + ragged tail.
func TestCUDAIQ4XSMatMulMTParity(t *testing.T) {
	skipNoGPU(t)
	const K, N, M = 512, 48, 7 // M in [2,8): validates the newly-routed MT path (partial tile)
	rng := rand.New(rand.NewSource(53))
	raw := rawIQ4XS(K, N, rng)
	a := make([]float32, M*K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}

	rq, err := cuda.NewResidentBIQ4XSFromBlocks(raw, K, N)
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
	t.Logf("IQ4_XS M-tiled (M=%d) vs per-row GEMV: max abs diff %.3g", M, maxAbs)
	if maxAbs > 1e-3 {
		t.Fatalf("IQ4_XS M-tiled kernel diverges from the GEMV: max abs %.3g", maxAbs)
	}

	// The MT entry shape-routes (K>N → shared-staged kernel, N>=K → unstaged): the shape above
	// exercises the staged branch; also pin the unstaged branch (N>K) to the per-row GEMV.
	t.Run("wideN", func(t *testing.T) {
		const K2, N2, M2 = 256, 300, 13
		raw2 := rawIQ4XS(K2, N2, rng)
		a2 := make([]float32, M2*K2)
		for i := range a2 {
			a2[i] = float32(rng.NormFloat64())
		}
		rq2, err := cuda.NewResidentBIQ4XSFromBlocks(raw2, K2, N2)
		must(t, err)
		defer rq2.Free()
		da2, err := cuda.NewDeviceF32(M2, K2)
		must(t, err)
		defer da2.Free()
		must(t, da2.UploadF32(a2))
		dout2, err := cuda.NewDeviceF32(M2, N2)
		must(t, err)
		defer dout2.Free()
		must(t, rq2.QMatMulInto(da2, dout2))
		got2, err := dout2.ToHost()
		must(t, err)
		gd2, err := cuda.NewDeviceF32(1, K2)
		must(t, err)
		defer gd2.Free()
		gout2, err := cuda.NewDeviceF32(1, N2)
		must(t, err)
		defer gout2.Free()
		var maxAbs2 float64
		for m := 0; m < M2; m++ {
			must(t, gd2.UploadF32(a2[m*K2:(m+1)*K2]))
			must(t, rq2.QMatMulInto(gd2, gout2))
			gv, err := gout2.ToHost()
			must(t, err)
			for n := 0; n < N2; n++ {
				if d := math.Abs(got2.AtF64(m, n) - gv.AtF64(0, n)); d > maxAbs2 {
					maxAbs2 = d
				}
			}
		}
		t.Logf("IQ4_XS M-tiled wide-N (N>K) vs per-row GEMV: max abs diff %.3g", maxAbs2)
		if maxAbs2 > 1e-3 {
			t.Fatalf("wide-N MT branch diverges from the GEMV: max abs %.3g", maxAbs2)
		}
	})

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

func benchIQ4XSM(b *testing.B, m, k, n int) {
	if !cuda.Available() {
		b.Skip("no gpu")
	}
	rng := rand.New(rand.NewSource(59))
	raw := rawIQ4XS(k, n, rng)
	a := make([]float32, m*k)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ4XSFromBlocks(raw, k, n)
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

func BenchmarkIQ4XSM16_2048(b *testing.B)      { benchIQ4XSM(b, 16, 2048, 2048) }
func BenchmarkIQ4XSM32_2048(b *testing.B)      { benchIQ4XSM(b, 32, 2048, 2048) }
func BenchmarkIQ4XSM64_2048x5632(b *testing.B) { benchIQ4XSM(b, 64, 2048, 5632) }
func BenchmarkIQ4XSM64_5632x2048(b *testing.B) { benchIQ4XSM(b, 64, 5632, 2048) } // ffn_down (deep-K), M=64
