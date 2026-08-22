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

func TestDotTQ1RowNeonKnownTrits(t *testing.T) {
	x := [tq1BlockElems]float32{}
	for i := range x {
		x[i] = 1
	}
	for _, tc := range []struct {
		name   string
		packed byte
		tail   byte
		want   float64
	}{{"negative", 0, 0, -tq1BlockElems}, {"zero", 128, 127, 0}, {"positive", 255, 253, tq1BlockElems}} {
		t.Run(tc.name, func(t *testing.T) {
			raw := [tq1BlockSize]byte{}
			for i := 0; i < tq1PackedBytes; i++ {
				raw[i] = tc.packed
			}
			for i := tq1PackedBytes; i < tq1PackedBytes+tq1TailBytes; i++ {
				raw[i] = tc.tail
			}
			binary.LittleEndian.PutUint16(raw[52:], f32ToF16(1))
			got := dotTQ1RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], 1)
			if got != tc.want {
				t.Fatalf("dotTQ1RowNeon = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDotTQ1AsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/tq1BlockElems*tq1BlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*tq1BlockElems < k; b++ {
			d := [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5]
			//perfscan:ignore PS4001 The test plants distinct f16 values in strided 54-byte scale fields.
			binary.LittleEndian.PutUint16(raw[b*tq1BlockSize+52:], f32ToF16(d))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotTQ1RowASM(x, raw, k), dotTQ1Row(x, raw, k)
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

func TestDotTQ1AsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeTQ1Raw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotTQ1RowASM(x, raw, k), dotTQ1Row(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotTQ1AsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeTQ1Raw(k)
	if got := testing.AllocsPerRun(1000, func() {
		tq1DotSink = dotTQ1RowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotTQ1RowASM allocations = %g, want 0", got)
	}
}

var tq1DotSink float64

func BenchmarkDotTQ1Paths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeTQ1Raw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotTQ1Row}, {"neon", dotTQ1RowASM}}
	if os.Getenv("GOAI_GGUF_TQ1_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				tq1DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulTQ1Paths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotTQ1Row}, {"neon", dotTQ1RowASM}}
	if os.Getenv("GOAI_GGUF_TQ1_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotTQ1RowFn
	defer func() { dotTQ1RowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotTQ1RowFn = path.dot
					benchQMatMulTQ1NK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
