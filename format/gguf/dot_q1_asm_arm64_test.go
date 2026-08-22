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

func TestDotQ1RowNeonKnownSigns(t *testing.T) {
	x := [q1BlockElems]float32{}
	raw := [q1BlockSize]byte{}
	for i := range x {
		x[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[:2], f32ToF16(1))
	for _, tc := range []struct {
		name  string
		signs byte
		want  float64
	}{{"positive", 0xff, q1BlockElems}, {"negative", 0x00, -q1BlockElems}} {
		t.Run(tc.name, func(t *testing.T) {
			for i := 2; i < len(raw); i++ {
				raw[i] = tc.signs
			}
			got := dotQ1RowNeon(&x[0], &raw[0], &f16Table[0], &q1SignBytes[0][0], 1)
			if got != tc.want {
				t.Fatalf("dotQ1RowNeon = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestDotQ1AsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{128, 256, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/q1BlockElems*q1BlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*q1BlockElems < k; b++ {
			d := [...]float32{0.03125, -0.0625, 0.125, 0.5, 2}[(trial+b)%5]
			binary.LittleEndian.PutUint16(raw[b*q1BlockSize:], f32ToF16(d))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got, want := dotQ1RowASM(x, raw, k), dotQ1Row(x, raw, k)
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

func TestDotQ1AsmCancellationHeavy(t *testing.T) {
	const k = 4096
	raw := makeQ1Raw(k)
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	x := make([]float32, k)
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got, want := dotQ1RowASM(x, raw, k), dotQ1Row(x, raw, k)
	rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
	if rel > 1e-4 {
		t.Fatalf("cancellation-heavy asm %v vs scalar %v (relative=%g)", got, want, rel)
	}
}

func TestDotQ1AsmAllocs(t *testing.T) {
	const k = 4096
	x, raw := benchF32(k), makeQ1Raw(k)
	if got := testing.AllocsPerRun(1000, func() {
		q1DotSink = dotQ1RowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotQ1RowASM allocations = %g, want 0", got)
	}
}

var q1DotSink float64

func BenchmarkDotQ1Paths(b *testing.B) {
	const k = 4096
	x, raw := benchF32(k), makeQ1Raw(k)
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotQ1Row}, {"neon", dotQ1RowASM}}
	if os.Getenv("GOAI_GGUF_Q1_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				q1DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulQ1Paths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotQ1Row}, {"neon", dotQ1RowASM}}
	if os.Getenv("GOAI_GGUF_Q1_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotQ1RowFn
	defer func() { dotQ1RowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotQ1RowFn = path.dot
					benchQMatMulQ1NK(b, 1, shape.n, 1024)
				})
			}
		})
	}
}
