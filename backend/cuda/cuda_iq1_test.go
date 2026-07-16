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

// IQ1_S is the 1.56-bit ternary grid-codebook i-quant (READ-only, no gguf encoder), so the test
// builds valid blocks directly — any qs/qh bytes are legal (grid index 0..2047, multiplier s 0..7,
// delta sign) — and compares the CUDA GEMV to gguf.Dequantize of the same bytes + a host matmul.
// The kernel reconstructs the 2048×8 ternary grid host-side via the public gguf.Dequantize. Closes
// the i-quant family (IQ1/IQ2/IQ3/IQ4 all native on CUDA).
func TestCUDAIQ1SMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/256 = 2 super-blocks per row
	rng := rand.New(rand.NewSource(71))
	sbs := K / 256
	raw := make([]byte, N*sbs*50)
	for i := 0; i < N*sbs; i++ {
		off := i * 50
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1)) // d
		for b := 0; b < 32; b++ {
			raw[off+2+b] = byte(rng.Intn(256)) // qs (grid low 8 bits)
		}
		for b := 0; b < 8; b++ {
			binary.LittleEndian.PutUint16(raw[off+34+b*2:], uint16(rng.Intn(65536))) // qh (grid-high + s + δ-sign)
		}
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ1SFromBlocks(raw, K, N)
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

	deq, err := gguf.Dequantize(raw, gguf.IQ1_S, N*K)
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
	t.Logf("IQ1_S GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ1_S GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}

// TestCUDAIQ1MMatMulParity: the IQ1_M sibling (1.75-bit, same 2048×8 grid, per-2 sub-scales, and the
// split-f16 super-scale quirk — d's nibbles live in the top nibbles of the 4 scale words). READ-only,
// so build valid blocks directly (encode a reasonable d split across the scale words; random
// sub-scales/qs/qh are all legal) and compare the CUDA GEMV to gguf.Dequantize + host matmul.
func TestCUDAIQ1MMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48
	rng := rand.New(rand.NewSource(73))
	sbs := K / 256
	raw := make([]byte, N*sbs*56)
	for i := 0; i < N*sbs; i++ {
		off := i * 56
		for b := 0; b < 32; b++ {
			raw[off+b] = byte(rng.Intn(256)) // qs
		}
		for b := 0; b < 16; b++ {
			raw[off+32+b] = byte(rng.Intn(256)) // qh (grid-high triple + delta sign per nibble)
		}
		d16 := uint32(f32to16(float32(rng.NormFloat64()) * 0.1)) // super-scale, split across the 4 scale words
		for w := 0; w < 4; w++ {
			low12 := uint16(rng.Intn(4096))          // per-2 3-bit sub-scales
			topnib := uint16((d16 >> (4 * w)) & 0xF) // this word's slice of d's f16 bits
			binary.LittleEndian.PutUint16(raw[off+48+w*2:], low12|(topnib<<12))
		}
	}

	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := cuda.NewResidentBIQ1MFromBlocks(raw, K, N)
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

	deq, err := gguf.Dequantize(raw, gguf.IQ1_M, N*K)
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
	t.Logf("IQ1_M GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("IQ1_M GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", rel, maxAbs)
	}
}
