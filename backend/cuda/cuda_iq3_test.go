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

// IQ3_XXS is a READ-only grid-codebook i-quant (no gguf encoder), so the test builds valid blocks
// directly — any grid index (0..255), ksigns index (0..127) and 4-bit scale is valid, so random
// bytes/words are all legal — and compares the CUDA GEMV to gguf.Dequantize of the same bytes plus
// a host matmul. The kernel reconstructs the 256×4 grid host-side via the public gguf.Dequantize
// (no gguf internals). Same parity discipline as the IQ2_XXS test.
func TestCUDAIQ3XXSMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/256 = 2 super-blocks per row
	rng := rand.New(rand.NewSource(53))
	sbs := K / 256
	raw := make([]byte, N*sbs*98)
	for i := 0; i < N*sbs; i++ {
		off := i * 98
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		for pos := 0; pos < 64; pos++ {
			raw[off+2+pos] = byte(rng.Intn(256)) // grid index
		}
		for g := 0; g < 8; g++ {
			binary.LittleEndian.PutUint32(raw[off+66+g*4:], rng.Uint32()) // 4 ksigns idx + 4-bit scale
		}
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ3XXSFromBlocks(raw, K, N)
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

	deq, err := gguf.Dequantize(raw, gguf.IQ3_XXS, N*K)
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
	t.Logf("IQ3_XXS GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ3_XXS GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}

// TestCUDAIQ3SMatMulParity: the IQ3_S sibling (3.44-bit, 512×4 grid, 9-bit indices via qh, direct
// sign bytes, explicit per-32 4-bit sub-scales). READ-only i-quant → build valid blocks directly
// (any qs/qh/signs bytes and any 4-bit scale are legal) and compare the CUDA GEMV to gguf.Dequantize
// + host matmul.
func TestCUDAIQ3SMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/256 = 2 super-blocks per row
	rng := rand.New(rand.NewSource(59))
	sbs := K / 256
	raw := make([]byte, N*sbs*110)
	for i := 0; i < N*sbs; i++ {
		off := i * 110
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		for b := 0; b < 64; b++ {
			raw[off+2+b] = byte(rng.Intn(256)) // qs (grid low 8 bits)
		}
		for b := 0; b < 8; b++ {
			raw[off+66+b] = byte(rng.Intn(256)) // qh (grid high bits)
		}
		for b := 0; b < 32; b++ {
			raw[off+74+b] = byte(rng.Intn(256)) // direct sign bytes
		}
		for b := 0; b < 4; b++ {
			raw[off+106+b] = byte(rng.Intn(256)) // 8 four-bit sub-scales
		}
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ3SFromBlocks(raw, K, N)
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

	deq, err := gguf.Dequantize(raw, gguf.IQ3_S, N*K)
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
	t.Logf("IQ3_S GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ3_S GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}
