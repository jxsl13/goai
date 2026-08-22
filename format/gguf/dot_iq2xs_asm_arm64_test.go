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

func TestDotIQ2XSBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [iq2xsBlockSize]byte{}
	coeff := [16]float32{}
	for i := range x {
		x[i] = 1
	}
	for i := range coeff {
		coeff[i] = 1
	}
	got := dotIQ2XSBlockNeon(&x[0], &raw[2], &coeff[0], &iq2xsGrid[0][0], &iqKSignMasks[0][0])
	const want = 2048
	if got != want {
		t.Fatalf("dotIQ2XSBlockNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotIQ2XSAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/qkK*iq2xsBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*qkK < k; b++ {
			scale := [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5]
			//perfscan:ignore PS4001 randomized gate writes one f16 scale per strided IQ2_XS super-block
			binary.LittleEndian.PutUint16(raw[b*iq2xsBlockSize:], f32ToF16(scale))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotIQ2XSRowASM(x, raw, k), dotIQ2XSRow(x, raw, k)
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

func TestDotIQ2XSAsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeIQ2XSRaw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotIQ2XSRowASM(x, raw, k), dotIQ2XSRow(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotIQ2XSAsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeIQ2XSRaw(k)
	if got := testing.AllocsPerRun(1000, func() {
		iq2XSDotSink = dotIQ2XSRowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotIQ2XSRowASM allocations = %g, want 0", got)
	}
}

var iq2XSDotSink float64

func BenchmarkDotIQ2XSPaths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeIQ2XSRaw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ2XSRow}, {"neon", dotIQ2XSRowASM}}
	if os.Getenv("GOAI_GGUF_IQ2XS_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				iq2XSDotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulIQ2XSPaths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotIQ2XSRow}, {"neon", dotIQ2XSRowASM}}
	if os.Getenv("GOAI_GGUF_IQ2XS_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotIQ2XSRowFn
	defer func() { dotIQ2XSRowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotIQ2XSRowFn = path.dot
					benchQMatMulIQ2XSNK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
