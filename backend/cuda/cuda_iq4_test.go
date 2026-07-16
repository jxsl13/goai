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

// f32to16 encodes a normal-range float32 to IEEE half bits — used only to fill the
// f16 d of hand-built IQ4 blocks. Exactness doesn't matter for parity: the kernel and
// the gguf reference decode the SAME bytes, so any finite half works.
func f32to16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16((b >> 16) & 0x8000)
	e := int((b>>23)&0xFF) - 127
	if e < -14 {
		return sign // underflow → signed zero
	}
	if e > 15 {
		return sign | 0x7BFF // clamp to max normal (avoid Inf/NaN)
	}
	return sign | uint16((e+15)<<10) | uint16((b>>13)&0x3FF)
}

// IQ4 has no gguf ENCODER (dequant/READ-only), so build valid blocks directly: any
// nibbles are valid codebook indices. The reference is gguf.Dequantize of the same
// bytes → f64 dot. Asserts absolute error (coarse quant → near-zero outputs spike rel).

func TestCUDAIQ4NLMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/32 = 16 blocks per row
	rng := rand.New(rand.NewSource(41))
	nblk := K / 32
	raw := make([]byte, N*nblk*18)
	for i := 0; i < N*nblk; i++ {
		off := i * 18
		binary.LittleEndian.PutUint16(raw[off:], f32to16(float32(rng.NormFloat64())*0.1))
		for j := 0; j < 16; j++ {
			raw[off+2+j] = byte(rng.Intn(16)) | byte(rng.Intn(16))<<4
		}
	}
	iq4Parity(t, "IQ4_NL", raw, K, N, gguf.IQ4_NL,
		func(b []byte) (qProj, error) { return cuda.NewResidentBIQ4NLFromBlocks(b, K, N) })
}

func TestCUDAIQ4XSMatMulParity(t *testing.T) {
	skipNoGPU(t)
	const K, N = 512, 48 // K/256 = 2 super-blocks per row
	rng := rand.New(rand.NewSource(43))
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
	iq4Parity(t, "IQ4_XS", raw, K, N, gguf.IQ4_XS,
		func(b []byte) (qProj, error) { return cuda.NewResidentBIQ4XSFromBlocks(b, K, N) })
}

func iq4Parity(t *testing.T, name string, raw []byte, K, N int, qt gguf.QuantType, mk func([]byte) (qProj, error)) {
	rng := rand.New(rand.NewSource(7))
	a := make([]float32, K)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	rq, err := mk(raw)
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

	deq, err := gguf.Dequantize(raw, qt, N*K)
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
	// Relative to the output scale: f32-accumulation error tracks the (synthetic,
	// unrealistically large under random scales) weight magnitude, so a bit-exact kernel
	// keeps maxAbs/maxMag at the f32-epsilon floor while any real indexing bug lands ~O(1).
	rel := maxAbs / math.Max(maxMag, 1e-9)
	t.Logf("%s GEMV maxAbs %.3e (maxMag %.3e, rel %.2e)", name, maxAbs, maxMag, rel)
	if rel > 1e-5 {
		t.Fatalf("%s GEMV deviates from dequant reference: rel %.3e (maxAbs %.3e)", name, rel, maxAbs)
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
	relAcc := maxAbs / math.Max(maxMag, 1e-9)
	t.Logf("%s GEMV acc maxAbs %.3e (rel %.2e)", name, maxAbs, relAcc)
	if relAcc > 1e-5 {
		t.Fatalf("%s GEMV beta=1 deviates: rel %.3e (maxAbs %.3e)", name, relAcc, maxAbs)
	}
}
