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

func TestDotQ6KAsm(t *testing.T) {
	for _, k := range []int{256, 512, 4096} {
		x := make([]float32, k)
		w := make([]float32, k)
		for i := range x {
			x[i] = float32(math.Sin(float64(i) * 0.01))
			w[i] = float32(math.Cos(float64(i) * 0.013))
		}
		raw := quantizeQ6_K(w)
		got := dotQ6_KRowASM(x, raw, k)
		want := dotQ6_KRow(x, raw, k)
		rel := math.Abs(got-want) / (math.Abs(want) + 1e-9)
		if rel > 1e-4 {
			t.Fatalf("k=%d: asm %v vs scalar %v (rel %g)", k, got, want, rel)
		}
		t.Logf("k=%d: asm=%v scalar=%v rel=%g", k, got, want, rel)
	}
}

func TestDotQ6KAsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	maxRel := 0.0
	for trial := range 100 {
		k := []int{256, 512, 4096}[trial%3]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64())
		}
		raw := make([]byte, k/qkK*q6kBlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for sb := 0; sb*qkK < k; sb++ {
			d := []float32{0.25, 0.5, 1, 2}[(trial+sb)%4]
			//perfscan:ignore PS4001 randomized correctness gate writes one scalar f16 header per 256-weight block
			binary.LittleEndian.PutUint16(raw[sb*q6kBlockSize+208:], f32ToF16(d))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got := dotQ6_KRowASM(x, raw, k)
		want := dotQ6_KRow(x, raw, k)
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
	t.Logf("maximum scalar-relative error across random raw blocks: %g", maxRel)
}

var q6dsink float64

func BenchmarkDotQ6KAsm(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ6_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q6dsink = dotQ6_KRowASM(x, raw, k)
	}
}

func BenchmarkDotQ6KScalar(b *testing.B) {
	const k = 4096
	x := make([]float32, k)
	w := make([]float32, k)
	for i := range x {
		x[i] = float32(i % 7)
		w[i] = float32(i % 5)
	}
	raw := quantizeQ6_K(w)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q6dsink = dotQ6_KRow(x, raw, k)
	}
}

func BenchmarkQMatMulQ6KPaths(b *testing.B) {
	type path struct {
		name string
		dot  func([]float32, []byte, int) float64
	}
	paths := []path{{"scalar", dotQ6_KRow}, {"neon", dotQ6_KRowASM}}
	if os.Getenv("GOAI_GGUF_Q6K_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					old := dotQ6KRowFn
					dotQ6KRowFn = path.dot
					defer func() { dotQ6KRowFn = old }()
					benchQMatMulNK(b, 1, shape.n, 1024, Q6_K)
				})
			}
		})
	}
}
