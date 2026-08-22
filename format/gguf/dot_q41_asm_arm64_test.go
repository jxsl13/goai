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

	"github.com/jxsl13/goai/tensor"
)

func TestDotQ41RowNeonKnownValue(t *testing.T) {
	x := [blockElems]float32{}
	raw := [q41BlockSize]byte{}
	for i := range x {
		x[i] = 1
	}
	binary.LittleEndian.PutUint16(raw[0:2], f32ToF16(1))
	binary.LittleEndian.PutUint16(raw[2:4], f32ToF16(-2))
	for i := 4; i < len(raw); i++ {
		raw[i] = 0xf0
	}
	got := dotQ41RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], 1)
	const want = 176
	if got != want {
		t.Fatalf("dotQ41RowNeon = %v (%#016x), want %d", got, math.Float64bits(got), want)
	}
}

func TestDotQ41AsmRandomRaw(t *testing.T) {
	rng := rand.New(rand.NewSource(20260822))
	maxMixed := 0.0
	for trial := range 120 {
		k := []int{32, 64, 256, 4096}[trial%4]
		x := make([]float32, k)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		raw := make([]byte, k/blockElems*q41BlockSize)
		if _, err := rng.Read(raw); err != nil {
			t.Fatal(err)
		}
		for b := 0; b*blockElems < k; b++ {
			d := [...]float32{0, 0.0625, 0.25, 1, 4}[(trial+b)%5]
			m := [...]float32{-8, -1.5, 0, 0.75, 16}[(trial+2*b)%5]
			o := b * q41BlockSize
			//perfscan:ignore PS4001 randomized gate writes one strided scale/minimum pair per Q4_1 block
			binary.LittleEndian.PutUint16(raw[o:o+2], f32ToF16(d))
			binary.LittleEndian.PutUint16(raw[o+2:o+4], f32ToF16(m))
		}
		xBefore, rawBefore := slices.Clone(x), slices.Clone(raw)
		got := dotQ41RowASM(x, raw, k)
		want := dotQ41Row(x, raw, k)
		delta := math.Abs(got - want)
		mixed := delta / (1 + math.Abs(want))
		maxMixed = max(maxMixed, mixed)
		if delta > 2e-5+2e-5*math.Abs(want) {
			t.Fatalf("trial=%d k=%d: asm=%v scalar=%v delta=%g", trial, k, got, want, delta)
		}
		if !slices.Equal(x, xBefore) || !bytes.Equal(raw, rawBefore) {
			t.Fatalf("trial=%d k=%d: kernel mutated an input", trial, k)
		}
	}
	t.Logf("maximum mixed error across arbitrary raw rows: %g", maxMixed)
}

func TestDotQ41AsmCancellationHeavy(t *testing.T) {
	const k = 4096
	x := make([]float32, k)
	raw, err := Quantize(tensorFromBenchF32(k), Q4_1)
	if err != nil {
		t.Fatal(err)
	}
	copy(raw[len(raw)/2:], raw[:len(raw)/2])
	for i := range k / 2 {
		v := float32(math.Sin(float64(i)*0.13) * 64)
		x[i], x[k/2+i] = v, -v
	}
	x[k-1] += 0.25
	got := dotQ41RowASM(x, raw, k)
	want := dotQ41Row(x, raw, k)
	if delta := math.Abs(got - want); delta > 2e-5+2e-5*math.Abs(want) {
		t.Fatalf("cancellation-heavy asm=%v scalar=%v delta=%g", got, want, delta)
	}
}

func tensorFromBenchF32(n int) *tensor.Tensor {
	return tensor.FromFloat32(tensor.Shape{n}, benchF32(n))
}

func TestDotQ41AsmAllocs(t *testing.T) {
	const k = 4096
	x := benchF32(k)
	raw, err := Quantize(tensorFromBenchF32(k), Q4_1)
	if err != nil {
		t.Fatal(err)
	}
	if got := testing.AllocsPerRun(1000, func() {
		q41DotSink = dotQ41RowASM(x, raw, k)
	}); got != 0 {
		t.Fatalf("dotQ41RowASM allocations = %g, want 0", got)
	}
}

var q41DotSink float64

func dotQ41Materialized(row []float32, raw []byte, k int) float64 {
	w, _ := dequantQ4_1(tensor.Shape{k}, raw)
	var acc float64
	for i, weight := range w.Storage().F32() {
		acc += float64(row[i]) * float64(weight)
	}
	return acc
}

func BenchmarkDotQ41Paths(b *testing.B) {
	const k = 4096
	x := benchF32(k)
	raw, err := Quantize(tensorFromBenchF32(k), Q4_1)
	if err != nil {
		b.Fatal(err)
	}
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar-fused", dotQ41Row}, {"neon-fused", dotQ41RowASM}, {"materialized", dotQ41Materialized}}
	if os.Getenv("GOAI_GGUF_Q41_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	for _, path := range paths {
		b.Run(path.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				q41DotSink = path.dot(x, raw, k)
			}
		})
	}
}

func BenchmarkQMatMulQ41Paths(b *testing.B) {
	paths := []struct {
		name string
		dot  func([]float32, []byte, int) float64
	}{{"scalar", dotQ41Row}, {"neon", dotQ41RowASM}}
	if os.Getenv("GOAI_GGUF_Q41_NEON_FIRST") != "" {
		paths[0], paths[1] = paths[1], paths[0]
	}
	old := dotQ41RowFn
	defer func() { dotQ41RowFn = old }()
	for _, shape := range []struct {
		name string
		n    int
	}{{"N64_K1024", 64}, {"N4096_K1024", 4096}} {
		b.Run(shape.name, func(b *testing.B) {
			for _, path := range paths {
				b.Run(path.name, func(b *testing.B) {
					dotQ41RowFn = path.dot
					benchQMatMulNK(b, 1, shape.n, 1024, Q4_1)
				})
			}
		})
	}
}
