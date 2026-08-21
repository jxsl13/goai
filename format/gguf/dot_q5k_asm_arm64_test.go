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

func TestDotQ5KBlockNeonKnownValue(t *testing.T) {
	x := [qkK]float32{}
	raw := [160]byte{} // qh[32] followed by qs[128]
	coeff := [16]float32{}
	for i := range x {
		x[i] = 1
	}
	for i := range 32 {
		raw[i] = 0xff
	}
	for i := 32; i < len(raw); i++ {
		raw[i] = 0x21
	}
	for pair := range 4 {
		coeff[pair*4+0] = 1
		coeff[pair*4+2] = 1
	}
	got := dotQ5KBlockNeon(&x[0], &raw[0], &coeff[0], &qKByteToF32Indexes[0])
	// 128 low values of 17 plus 128 high values of 18.
	if got != 4480 {
		t.Fatalf("dotQ5KBlockNeon = %v (%#08x), want 4480", got, math.Float32bits(got))
	}
}

func TestDotQ5KAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		raw := make([]byte, k/qkK*q5kBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for sb := 0; sb*qkK < k; sb++ {
			d := []float32{0.25, 0.5, 1, 2}[(trial+sb)%4]
			dmin := []float32{0.125, 0.25, 0.75, 1.5}[(trial+2*sb)%4]
			base := sb * q5kBlockSize
			//perfscan:ignore PS4001 randomized correctness gate writes two scalar f16 headers per 256-weight block
			binary.LittleEndian.PutUint16(raw[base:], f32ToF16(d))
			binary.LittleEndian.PutUint16(raw[base+2:], f32ToF16(dmin))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got := dotQ5_KRowASM(x, raw, k)
		want := dotQ5_KRow(x, raw, k)
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

var q5dsink float64

func BenchmarkDotQ5KAsm(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ5_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q5dsink = dotQ5_KRowASM(x, raw, k)
	}
}

func BenchmarkDotQ5KScalar(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ5_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q5dsink = dotQ5_KRow(x, raw, k)
	}
}

func BenchmarkQMatMulQ5KPaths(b *testing.B) {
	type path struct {
		name string
		dot  func([]float32, []byte, int) float64
	}
	paths := []path{{"scalar", dotQ5_KRow}, {"neon", dotQ5_KRowASM}}
	if os.Getenv("GOAI_GGUF_Q5K_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					old := dotQ5KRowFn
					dotQ5KRowFn = path.dot
					defer func() { dotQ5KRowFn = old }()
					benchQMatMulNK(b, 1, shape.n, 1024, Q5_K)
				})
			}
		})
	}
}
