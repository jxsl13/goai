//go:build arm64

package gguf

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"slices"
	"testing"
)

func TestDotMXFP4RowNeonKnownValue(t *testing.T) {
	x := [blockElems]float32{}
	raw := [mxfp4BlockSize]byte{128}
	for i := range x {
		x[i] = 1
	}
	for i := 1; i < len(raw); i++ {
		raw[i] = 0x21
	}
	got := dotMXFP4RowNeon(
		&x[0], &raw[0], &e8m0HalfTable[0], &mxfp4KValuesI8[0], &qKByteToF32Indexes[0], 1,
	)
	const want = 48
	if got != want {
		t.Fatalf("dotMXFP4RowNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotMXFP4AsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	scales := [...]byte{0, 1, 2, 120, 127, 128, 135}
	maxRel := 0.0
	for trial := range 100 {
		k := []int{32, 64, 256, 4096}[trial%4]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/blockElems*mxfp4BlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*blockElems < k; b++ {
			raw[b*mxfp4BlockSize] = scales[(trial+b)%len(scales)]
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotMXFP4RowASM(x, raw, k), dotMXFP4Row(x, raw, k)
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
		maxRel = max(maxRel, rel)
		if rel > 1e-4 {
			t.Fatalf("trial=%d k=%d: asm %v vs scalar %v (relative=%g)", trial, k, got, want, rel)
		}
		if !slices.Equal(x, xBefore) || !bytes.Equal(raw, rawBefore) {
			t.Fatalf("trial=%d k=%d: kernel mutated an input", trial, k)
		}
	}
	t.Logf("maximum scalar-relative error across arbitrary raw rows: %g", maxRel)
}

func TestDotMXFP4AsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeMXFP4Raw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotMXFP4RowASM(x, raw, k), dotMXFP4Row(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotMXFP4AsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeMXFP4Raw(k)
	if got := testing.AllocsPerRun(1000, func() {
		mxfp4DotSink = dotMXFP4RowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotMXFP4RowASM allocations = %g, want 0", got)
	}
}

var mxfp4DotSink float64

func BenchmarkDotMXFP4Paths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeMXFP4Raw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotMXFP4Row}, {"neon", dotMXFP4RowASM}}
	if os.Getenv("GOAI_GGUF_MXFP4_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				mxfp4DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulMXFP4Paths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotMXFP4Row}, {"neon", dotMXFP4RowASM}}
	if os.Getenv("GOAI_GGUF_MXFP4_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotMXFP4RowFn
	defer func() { dotMXFP4RowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotMXFP4RowFn = path.dot
					benchQMatMulMXFP4NK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
