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

func TestDotQ3KBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [96]byte{} // high mask[32] followed by qs[64]
	coeff := [16]float32{}
	for i := range x {
		x[i] = 1
	}
	for i := range 16 {
		raw[i] = 0xff
	}
	for i := 32; i < len(raw); i++ {
		raw[i] = 0xe4 // two-bit streams 0, 1, 2, 3
	}
	for i := range coeff {
		coeff[i] = 1
	}
	got := dotQ3KBlockNeon(&x[0], &raw[0], &coeff[0], &qKByteToF32Indexes[0])
	// Each half has sixteen high-mask-set and sixteen high-mask-clear values
	// at each q=0..3: 2 * sum_j(16*j + 16*(j-4)) = -128.
	if got != -128 {
		t.Fatalf("dotQ3KBlockNeon = %v (%#08x), want -128", got, math.Float32bits(got))
	}
}

func TestDotQ3KAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		raw := make([]byte, k/qkK*q3kBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for sb := 0; sb*qkK < k; sb++ {
			d := []float32{0.25, 0.5, 1, 2}[(trial+sb)%4]
			//perfscan:ignore PS4001 randomized correctness gate writes one scalar f16 header per 256-weight block
			binary.LittleEndian.PutUint16(raw[sb*q3kBlockSize+108:], f32ToF16(d))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got := dotQ3_KRowASM(x, raw, k)
		want := dotQ3_KRow(x, raw, k)
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
		if rel > maxRel {
			maxRel = rel
		}
		if rel > 1e-4 {
			t.Fatalf("trial=%d k=%d: asm %v vs scalar %v (rel %g)", trial, k, got, want, rel)
		}
		if !slices.Equal(x, xBefore) || !bytes.Equal(raw, rawBefore) {
			t.Fatalf("trial=%d k=%d: kernel mutated an input", trial, k)
		}
	}
	if maxRel > 1e-4 {
		t.Fatalf("maximum scalar-relative error %g exceeds 1e-4", maxRel)
	}
	t.Logf("maximum scalar-relative error across random raw blocks: %g", maxRel)
}

var q3dsink float64

func BenchmarkDotQ3KAsm(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ3_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q3dsink = dotQ3_KRowASM(x, raw, k)
	}
}

func BenchmarkDotQ3KScalar(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ3_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q3dsink = dotQ3_KRow(x, raw, k)
	}
}

func BenchmarkQMatMulQ3KPaths(b *testing.B) {
	type path struct {
		name string
		dot  func([]float32, []byte, int) float64
	}
	paths := []path{{"scalar", dotQ3_KRow}, {"neon", dotQ3_KRowASM}}
	if os.Getenv("GOAI_GGUF_Q3K_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					old := dotQ3KRowFn
					dotQ3KRowFn = path.dot
					defer func() { dotQ3KRowFn = old }()
					benchQMatMulNK(b, 1, shape.n, 1024, Q3_K)
				})
			}
		})
	}
}
