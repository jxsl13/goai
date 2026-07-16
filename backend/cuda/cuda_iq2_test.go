//go:build cuda && cgo && (linux || windows)

package cuda_test

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend/cuda"
	"github.com/jxsl13/goai/format/gguf"
)

// IQ2_XXS is the first GRID-codebook i-quant and has no gguf encoder (dequant/READ-only), so
// the test builds valid blocks directly (any grid index / ksigns index / scale is valid) and
// compares the kernel to gguf.Dequantize of the same bytes. The kernel reconstructs the 256×8
// grid host-side via the public gguf.Dequantize (no gguf internals). Parity is asserted relative
// to the output magnitude (random scales inflate weights, so f32-accumulation abs error tracks
// that while a real bug lands ~O(1)).
func TestCUDAIQ2XXSMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/256 = 2 super-blocks per row
	rng := rand.New(rand.NewSource(47))
	sbs := K / 256
	raw := make([]byte, N*sbs*66)
	for i := 0; i < N*sbs; i++ {
		off := i * 66
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		for pair := 0; pair < 8; pair++ {
			binary.LittleEndian.PutUint32(raw[off+2+pair*8:], rng.Uint32())   // qs0 (grid indices)
			binary.LittleEndian.PutUint32(raw[off+2+pair*8+4:], rng.Uint32()) // qs1 (ksigns + scale)
		}
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ2XXSFromBlocks(raw, K, N)
	must(t, err)
	defer rq.Free()
	da, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dout, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer dout.Free()
	must(t, rq.QMatMulInto(da, dout))
	got, err := dout.ToHost()
	must(t, err)

	deq, err := gguf.Dequantize(raw, gguf.IQ2_XXS, N*K)
	must(t, err)
	df := deq.Storage().F32()
	ref := make([]float64, N)
	var maxAbs, maxMag float64
	for n := 0; n < N; n++ {
		for k := 0; k < K; k++ {
			ref[n] += float64(a[k]) * float64(df[n*K+k])
		}
		if abs := math.Abs(got.AtF64(0, n) - ref[n]); abs > maxAbs {
			maxAbs = abs
		}
		if math.Abs(ref[n]) > maxMag {
			maxMag = math.Abs(ref[n])
		}
	}
	rel := maxAbs / math.Max(maxMag, 1e-9)
	t.Logf("IQ2_XXS GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ2_XXS GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}

	// beta=1
	init := make([]float32, N)
	for i := range init {
		init[i] = float32(rng.NormFloat64())
	}
	must(t, dout.UploadF32(init))
	must(t, rq.QMatMulAccInto(da, dout))
	got2, err := dout.ToHost()
	must(t, err)
	maxAbs = 0
	for n := 0; n < N; n++ {
		if abs := math.Abs(got2.AtF64(0, n) - (float64(init[n]) + ref[n])); abs > maxAbs {
			maxAbs = abs
		}
	}
	if r2 := maxAbs / math.Max(maxMag, 1e-9); r2 > 1e-5 {
		t.Fatalf("IQ2_XXS GEMV beta=1 deviates: rel %.3e", r2)
	}
}

// IQ2_XS: 74-byte super-block (f16 d + 32 u16 qs + 8 scale bytes), 512×8 grid. Same self-built
// parity approach as IQ2_XXS.
func TestCUDAIQ2XSMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(53))
	sbs := K / 256
	raw := make([]byte, N*sbs*74)
	for i := 0; i < N*sbs; i++ {
		off := i * 74
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		for l := 0; l < 32; l++ {
			binary.LittleEndian.PutUint16(raw[off+2+l*2:], uint16(rng.Intn(0x10000))) // qs (9-bit grid + 7-bit ksigns)
		}
		for j := 0; j < 8; j++ {
			raw[off+66+j] = byte(rng.Intn(256)) // 16 four-bit scales
		}
	}
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ2XSFromBlocks(raw, K, N)
	must(t, err)
	defer rq.Free()
	da, err := cuda.NewDeviceF32(1, K)
	must(t, err)
	defer da.Free()
	must(t, da.UploadF32(a))
	dout, err := cuda.NewDeviceF32(1, N)
	must(t, err)
	defer dout.Free()
	must(t, rq.QMatMulInto(da, dout))
	got, err := dout.ToHost()
	must(t, err)

	deq, err := gguf.Dequantize(raw, gguf.IQ2_XS, N*K)
	must(t, err)
	df := deq.Storage().F32()
	var maxAbs, maxMag float64
	for n := 0; n < N; n++ {
		var ref float64
		for k := 0; k < K; k++ {
			ref += float64(a[k]) * float64(df[n*K+k])
		}
		if abs := math.Abs(got.AtF64(0, n) - ref); abs > maxAbs {
			maxAbs = abs
		}
		if math.Abs(ref) > maxMag {
			maxMag = math.Abs(ref)
		}
	}
	rel := maxAbs / math.Max(maxMag, 1e-9)
	t.Logf("IQ2_XS GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ2_XS GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}
