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

func scalarQ8RowReference(x []float32, raw []byte) float32 {
	var acc float64
	for b := 0; b*blockElems < len(x); b++ {
		blk := raw[b*34 : b*34+34]
		//perfscan:ignore PS4001 scalar oracle intentionally decodes one scale per quant block
		d := f16ToF32(binary.LittleEndian.Uint16(blk))
		for i, q := range blk[2:] {
			acc += float64(x[b*blockElems+i]) * float64(d*float32(int8(q)))
		}
	}
	return float32(acc)
}

func TestDotQ8RowNeonKnownValue(t *testing.T) {
	x := [blockElems]float32{}
	raw := [34]byte{}
	for i := range x {
		x[i] = 1
		raw[i+2] = 1
	}
	binary.LittleEndian.PutUint16(raw[:2], 0x3c00) // f16(1)
	got := dotQ8RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], 1)
	if got != blockElems {
		t.Fatalf("dotQ8RowNeon = %v (%#08x), want %d", got, math.Float32bits(got), blockElems)
	}
}

func TestDotQ8RowNeonMatchesScalarAndPreservesInputs(t *testing.T) {
	rng := rand.New(rand.NewSource(20260821))
	blockCases := [...]int{1, 2, 7, 128}
	maxRelativeError := 0.0
	for trial := range 100 {
		blocks := blockCases[trial%len(blockCases)]
		x := make([]float32, blocks*blockElems)
		raw := make([]byte, blocks*34)
		for i := range x {
			x[i] = float32(rng.NormFloat64() * math.Pow(2, float64(rng.Intn(9)-4)))
		}
		for b := range blocks {
			// Positive finite f16 scales spanning small and large magnitudes.
			scales := [...]uint16{0x2800, 0x3400, 0x3c00, 0x4400, 0x5000}
			//perfscan:ignore PS4001 test setup writes one intentionally varied scale per block
			binary.LittleEndian.PutUint16(raw[b*34:], scales[b%len(scales)])
			for i := 2; i < 34; i++ {
				raw[b*34+i] = byte(int8(rng.Intn(255) - 127))
			}
		}
		xBefore := slices.Clone(x)
		rawBefore := slices.Clone(raw)
		got := dotQ8RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], blocks)
		want := scalarQ8RowReference(x, raw)
		diff := math.Abs(float64(got - want))
		scale := max(1, math.Abs(float64(want)))
		relativeError := diff / scale
		maxRelativeError = max(maxRelativeError, relativeError)
		if relativeError > 1e-4 {
			t.Fatalf("trial=%d blocks=%d: neon=%v scalar=%v relative error=%g", trial, blocks, got, want, relativeError)
		}
		if !slices.Equal(x, xBefore) || !bytes.Equal(raw, rawBefore) {
			t.Fatalf("trial=%d blocks=%d: kernel mutated an input", trial, blocks)
		}
	}
	if maxRelativeError > 1e-4 {
		t.Fatalf("maximum scalar-relative error=%g exceeds 1e-4", maxRelativeError)
	}
	t.Logf("maximum scalar-relative error across 100 raw-row trials: %.9g", maxRelativeError)
}

func BenchmarkDotQ8RowARM64(b *testing.B) {
	const k = 4096
	x := benchF32(k)
	raw := make([]byte, k/blockElems*34)
	for block := 0; block < k/blockElems; block++ {
		//perfscan:ignore PS4001 benchmark setup writes one scale per quant block outside timed work
		binary.LittleEndian.PutUint16(raw[block*34:], 0x3400)
		for i := 2; i < 34; i++ {
			raw[block*34+i] = byte(int8((block*31+i*17)%255 - 127))
		}
	}
	type mode struct {
		name string
		dot  func() float32
	}
	modes := []mode{
		{"scalar", func() float32 { return scalarQ8RowReference(x, raw) }},
		{"neon", func() float32 {
			return dotQ8RowNeon(&x[0], &raw[0], &f16Table[0], &qKByteToF32Indexes[0], k/blockElems)
		}},
	}
	if os.Getenv("GOAI_Q8_NEON_FIRST") != "" {
		modes[0], modes[1] = modes[1], modes[0]
	}
	var sink float32
	for _, mode := range modes {
		b.Run(mode.name, func(b *testing.B) {
			b.SetBytes(k * 4)
			b.ReportAllocs()
			for b.Loop() {
				sink = mode.dot()
			}
		})
	}
	_ = sink
}

func BenchmarkQMatMulQ8_0M1ARM64Kernel(b *testing.B) {
	type mode struct {
		name string
		fn   func(row []float32, weight []byte, n, k, rowBytes int, outf []float32)
	}
	modes := []mode{{"scalar", nil}, {"neon", q8FusedDecodeM1Neon}}
	if os.Getenv("GOAI_Q8_NEON_FIRST") != "" {
		modes[0], modes[1] = modes[1], modes[0]
	}
	old := q8FusedDecodeM1
	defer func() { q8FusedDecodeM1 = old }()
	for _, mode := range modes {
		b.Run(mode.name+"/N64", func(b *testing.B) {
			q8FusedDecodeM1 = mode.fn
			benchQMatMulNK(b, 1, 64, 1024, Q8_0)
		})
		b.Run(mode.name+"/N4096", func(b *testing.B) {
			q8FusedDecodeM1 = mode.fn
			benchQMatMulNK(b, 1, 4096, 1024, Q8_0)
		})
	}
}
