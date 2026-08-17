//go:build arm64

package gguf

import (
	"math"
	"math/rand"
	"os"
	"testing"
)

func TestDequantQ6KNeonBitExact(t *testing.T) {
	const blocks = 19
	raw := make([]byte, blocks*q6kBlockSize)
	rng := rand.New(rand.NewSource(17))
	if _, err := rng.Read(raw); err != nil {
		t.Fatal(err)
	}
	// Exercise finite normal, subnormal, signed-zero, and maximum-finite scale
	// encodings. Quantized GGUF weights require finite scales; NaN arithmetic
	// payload propagation is not a model-format semantic.
	halves := [...]uint16{0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff}
	for block := range blocks {
		h := halves[block%len(halves)]
		raw[block*q6kBlockSize+208] = byte(h)
		raw[block*q6kBlockSize+209] = byte(h >> 8)
	}

	want := make([]float32, blocks*qkK)
	got := make([]float32, blocks*qkK)
	dequantQ6_KIntoScalar(want, raw)
	dequantQ6_KIntoArch(got, raw)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: got %08x (%g), want %08x (%g)", i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

func TestDequantQ4KNeonBitExact(t *testing.T) {
	const blocks = 19
	raw := make([]byte, blocks*q4kBlockSize)
	rng := rand.New(rand.NewSource(19))
	if _, err := rng.Read(raw); err != nil {
		t.Fatal(err)
	}
	halves := [...]uint16{0x0000, 0x8000, 0x0001, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff, 0xfbff}
	for block := range blocks {
		d := halves[block%len(halves)]
		dmin := halves[(block+3)%len(halves)]
		o := block * q4kBlockSize
		raw[o+0], raw[o+1] = byte(d), byte(d>>8)
		raw[o+2], raw[o+3] = byte(dmin), byte(dmin>>8)
	}

	want := make([]float32, blocks*qkK)
	got := make([]float32, blocks*qkK)
	dequantQ4_KIntoScalar(want, raw)
	dequantQ4_KIntoArch(got, raw)
	for i := range want {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("element %d: got %08x (%g), want %08x (%g)", i,
				math.Float32bits(got[i]), got[i], math.Float32bits(want[i]), want[i])
		}
	}
}

func BenchmarkDequantQ6KIntoPaths(b *testing.B) {
	raw := make([]byte, (benchN/qkK)*q6kBlockSize)
	rng := rand.New(rand.NewSource(23))
	if _, err := rng.Read(raw); err != nil {
		b.Fatal(err)
	}
	// Keep scale halves finite and representative rather than allowing random
	// NaNs to turn the benchmark into a payload-propagation artifact.
	for block := 0; block < benchN/qkK; block++ {
		raw[block*q6kBlockSize+208] = 0x00
		raw[block*q6kBlockSize+209] = 0x3c // 1.0
	}
	dst := make([]float32, benchN)
	type path struct {
		name string
		fn   func([]float32, []byte)
	}
	paths := []path{{"scalar", dequantQ6_KIntoScalar}, {"neon", dequantQ6_KIntoArch}}
	if os.Getenv("GOAI_GGUF_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for b.Loop() {
				path.fn(dst, raw)
			}
		})
	}
}

func BenchmarkDequantQ4KIntoPaths(b *testing.B) {
	raw := make([]byte, (benchN/qkK)*q4kBlockSize)
	rng := rand.New(rand.NewSource(29))
	if _, err := rng.Read(raw); err != nil {
		b.Fatal(err)
	}
	for block := 0; block < benchN/qkK; block++ {
		o := block * q4kBlockSize
		raw[o+0], raw[o+1] = 0x00, 0x3c
		raw[o+2], raw[o+3] = 0x00, 0x38
	}
	dst := make([]float32, benchN)
	type path struct {
		name string
		fn   func([]float32, []byte)
	}
	paths := []path{{"scalar", dequantQ4_KIntoScalar}, {"neon", dequantQ4_KIntoArch}}
	if os.Getenv("GOAI_GGUF_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(int64(len(raw)))
			b.ResetTimer()
			for b.Loop() {
				path.fn(dst, raw)
			}
		})
	}
}
