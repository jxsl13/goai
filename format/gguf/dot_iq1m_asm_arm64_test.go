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

func TestDotIQ1MRowNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [iq1mBlockSize]byte{}
	for i := range x {
		x[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[52:54], 0xc000)
	binary.LittleEndian.PutUint16(raw[54:56], 0x3000)
	got := dotIQ1MRowNeon(&x[0], &raw[0], &f16Table[0], &iq1sDeltaGrid[0][0][0], &iq1sOddScales[0], &iq1mQHOffsets[0][0], 1)
	const want = -224
	if got != want {
		t.Fatalf("dotIQ1MRowNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotIQ1MAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/qkK*iq1mBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*qkK < k; b++ {
			putIQ1MScale(raw[b*iq1mBlockSize:(b+1)*iq1mBlockSize], [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5])
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotIQ1MRowASM(x, raw, k), dotIQ1MRow(x, raw, k)
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

func TestDotIQ1MAsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeIQ1MRaw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotIQ1MRowASM(x, raw, k), dotIQ1MRow(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotIQ1MAsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeIQ1MRaw(k)
	if got := testing.AllocsPerRun(1000, func() {
		iq1MDotSink = dotIQ1MRowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotIQ1MRowASM allocations = %g, want 0", got)
	}
}

var iq1MDotSink float64

func BenchmarkDotIQ1MPaths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeIQ1MRaw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ1MRow}, {"neon", dotIQ1MRowASM}}
	if os.Getenv("GOAI_GGUF_IQ1M_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				iq1MDotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulIQ1MPaths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ1MRow}, {"neon", dotIQ1MRowASM}}
	if os.Getenv("GOAI_GGUF_IQ1M_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotIQ1MRowFn
	defer func() { dotIQ1MRowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotIQ1MRowFn = path.dot
					benchQMatMulIQ1MNK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
