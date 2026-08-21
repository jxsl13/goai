//go:build arm64

package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
)

func TestDotIQ4NLRowNeonKnownValue(t *testing.T) {
	x := [blockElems]float32{}
	raw := [iq4nlBlockSize]byte{}
	for i := range x {
		x[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[:2], f32ToF16(1))
	for i := 2; i < len(raw); i++ {
		raw[i] = 0x10 // low=-127, high=-104
	}
	got := dotIQ4NLRowNeon(
		&x[0], &raw[0], &f16Table[0], &iq4KValuesI8[0], &qKByteToF32Indexes[0], 1,
	)
	const want = -3696
	if got != want {
		t.Fatalf("dotIQ4NLRowNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotIQ4NLAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{32, 64, 256, 4096}[trial%4]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/blockElems*iq4nlBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*blockElems < k; b++ {
			scale := [...]float32{0.0625, -0.125, 0.5, 2, 8}[(trial+b)%5]
			//perfscan:ignore PS4001 randomized gate writes one f16 scale per strided IQ4_NL block
			binary.LittleEndian.PutUint16(raw[b*iq4nlBlockSize:], f32ToF16(scale))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got := dotIQ4NLRowASM(x, raw, k)
		want := dotIQ4NLRow(x, raw, k)
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
		maxRel = max(maxRel, rel)
		if rel > 1e-4 {
			t.Fatalf("trial=%d k=%d: asm %v vs scalar %v (relative=%g)", trial, k, got, want, rel)
		}
		if !slices.Equal(x, xBefore) || !bytes.Equal(raw, rawBefore) {
			t.Fatalf("trial=%d k=%d: kernel mutated an input", trial, k)
		}
	}
	if maxRel > 1e-4 {
		t.Fatalf("maximum scalar-relative error %g exceeds 1e-4", maxRel)
	}
	t.Logf("maximum scalar-relative error across arbitrary raw rows: %g", maxRel)
}

func TestDotIQ4NLAsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeIQ4NLRaw(k)
	// Repeat the first half's weights in the second half. Activations cancel
	// term-for-term except for one controlled residual, so the result is tiny
	// relative to the absolute products without becoming an undefined zero-relative gate.
	copy(raw[k/blockElems/2*iq4nlBlockSize:], raw[:k/blockElems/2*iq4nlBlockSize])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got := dotIQ4NLRowASM(x, raw, k)
	want := dotIQ4NLRow(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotIQ4NLAsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeIQ4NLRaw(k)
	if got := testing.AllocsPerRun(1000, func() {
		iq4DotSink = dotIQ4NLRowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotIQ4NLRowASM allocations = %g, want 0", got)
	}
}

var iq4DotSink float64

func BenchmarkDotIQ4NLPaths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeIQ4NLRaw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ4NLRow}, {"neon", dotIQ4NLRowASM}}
	if os.Getenv("GOAI_GGUF_IQ4NL_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				iq4DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulIQ4NLPaths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ4NLRow}, {"neon", dotIQ4NLRowASM}}
	if os.Getenv("GOAI_GGUF_IQ4NL_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotIQ4NLRowFn
	defer func() { dotIQ4NLRowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotIQ4NLRowFn = path.dot
					benchQMatMulIQ4NLNK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
