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

func TestDotTQ2RowNeonKnownCodes(t *testing.T) {
	x := [tq2BlockElems]float32{}
	for i := range x {
		x[i] = 1
	}
	for _, tc := range []struct {
		name   string
		packed byte
		want   float64
	}{{"negative", 0x00, -tq2BlockElems}, {"zero", 0x55, 0}, {"positive", 0xaa, tq2BlockElems}, {"raw_code_3", 0xff, 2 * tq2BlockElems}} {
		t.Run(tc.name, func(t *testing.T) {
			raw := [tq2BlockSize]byte{}
			for i := 0; i < tq2PackedBytes; i++ {
				raw[i] = tc.packed
			}
			binary.LittleEndian.PutUint16(raw[tq2PackedBytes:], f32ToF16(1))
			got := dotTQ2RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], 1)
			if got != tc.want {
				t.Fatalf("dotTQ2RowNeon = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDotTQ2AsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxRel := 0.0
	for trial := 0; trial < 100; trial++ {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/tq2BlockElems*tq2BlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*tq2BlockElems < k; b++ {
			d := [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5]
			//perfscan:ignore PS4001 The test plants distinct f16 values in strided 66-byte scale fields.
			binary.LittleEndian.PutUint16(raw[b*tq2BlockSize+tq2PackedBytes:], f32ToF16(d))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotTQ2RowASM(x, raw, k), dotTQ2Row(x, raw, k)
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

func TestDotTQ2AsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeTQ2Raw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotTQ2RowASM(x, raw, k), dotTQ2Row(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotTQ2AsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeTQ2Raw(k)
	if got := testing.AllocsPerRun(1000, func() {
		tq2DotSink = dotTQ2RowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotTQ2RowASM allocations = %g, want 0", got)
	}
}

var tq2DotSink float64

func BenchmarkDotTQ2Paths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeTQ2Raw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotTQ2Row}, {"neon", dotTQ2RowASM}}
	if os.Getenv("GOAI_GGUF_TQ2_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				tq2DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulTQ2Paths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotTQ2Row}, {"neon", dotTQ2RowASM}}
	if os.Getenv("GOAI_GGUF_TQ2_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotTQ2RowFn
	defer func() { dotTQ2RowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotTQ2RowFn = path.dot
					benchQMatMulTQ2NK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
