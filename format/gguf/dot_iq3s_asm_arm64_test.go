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

func TestDotIQ3SBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [iq3sBlockSize]byte{}
	coeff := [8]float32{}
	for i := range x {
		x[i] = 1
	}
	for i := range coeff {
		coeff[i] = 1
	}
	got := dotIQ3SBlockNeon(&x[0], &raw[2], &coeff[0], &iq3sGrid[0][0], &iq3sSignNibbles[0][0])
	const want = 256
	if got != want {
		t.Fatalf("dotIQ3SBlockNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotIQ3SAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/qkK*iq3sBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*qkK < k; b++ {
			scale := [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5]
			//perfscan:ignore PS4001 randomized gate writes one f16 scale per strided IQ3_S super-block
			binary.LittleEndian.PutUint16(raw[b*iq3sBlockSize:], f32ToF16(scale))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotIQ3SRowASM(x, raw, k), dotIQ3SRow(x, raw, k)
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

func TestDotIQ3SAsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeIQ3SRaw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotIQ3SRowASM(x, raw, k), dotIQ3SRow(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotIQ3SAsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeIQ3SRaw(k)
	if got := testing.AllocsPerRun(1000, func() {
		iq3SDotSink = dotIQ3SRowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotIQ3SRowASM allocations = %g, want 0", got)
	}
}

var iq3SDotSink float64

func BenchmarkDotIQ3SPaths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeIQ3SRaw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ3SRow}, {"neon", dotIQ3SRowASM}}
	if os.Getenv("GOAI_GGUF_IQ3S_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				iq3SDotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulIQ3SPaths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ3SRow}, {"neon", dotIQ3SRowASM}}
	if os.Getenv("GOAI_GGUF_IQ3S_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotIQ3SRowFn
	defer func() { dotIQ3SRowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotIQ3SRowFn = path.dot
					benchQMatMulIQ3SNK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
