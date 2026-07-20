package gguf

import (
	"io"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// Perf guards for the dequant/quant/QMatMul hot paths
// (docs/perf-notes-lowlevel.md). 256K elements ≈ one 512×512 weight.

const benchN = 1 << 18

func benchF32(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i))) * 3
	}
	return x
}

func benchDequant(b *testing.B, qt QuantType) {
	x := benchF32(benchN)
	t := tensor.FromFloat32(tensor.Shape{benchN}, x)
	raw, err := Quantize(t, qt)
	if err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for range b.N {
		if _, err := Dequantize(raw, qt, benchN); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDequantQ8_0(b *testing.B) { benchDequant(b, Q8_0) }
func BenchmarkDequantQ4_0(b *testing.B) { benchDequant(b, Q4_0) }
func BenchmarkDequantQ4_K(b *testing.B) { benchDequant(b, Q4_K) }
func BenchmarkDequantQ6_K(b *testing.B) { benchDequant(b, Q6_K) }

// benchDequantRaw drives the READ-only formats (no encoder exists) with
// deterministic pseudo-random block bytes — decoding is total over any input.
func benchDequantRaw(b *testing.B, qt QuantType) {
	need, err := byteSize(uint32(qt), benchN)
	if err != nil {
		b.Fatal(err)
	}
	raw := make([]byte, need)
	s := uint32(12345)
	for i := range raw {
		s = s*1664525 + 1013904223
		raw[i] = byte(s >> 24)
	}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for range b.N {
		if _, err := Dequantize(raw, qt, benchN); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDequantQ2_K(b *testing.B)    { benchDequantRaw(b, Q2_K) }
func BenchmarkDequantQ3_K(b *testing.B)    { benchDequantRaw(b, Q3_K) }
func BenchmarkDequantQ5_K(b *testing.B)    { benchDequantRaw(b, Q5_K) }
func BenchmarkDequantIQ2_XXS(b *testing.B) { benchDequantRaw(b, IQ2_XXS) }
func BenchmarkDequantIQ2_XS(b *testing.B)  { benchDequantRaw(b, IQ2_XS) }
func BenchmarkDequantIQ2_S(b *testing.B)   { benchDequantRaw(b, IQ2_S) }
func BenchmarkDequantIQ3_XXS(b *testing.B) { benchDequantRaw(b, IQ3_XXS) }
func BenchmarkDequantIQ3_S(b *testing.B)   { benchDequantRaw(b, IQ3_S) }
func BenchmarkDequantIQ4_NL(b *testing.B)  { benchDequantRaw(b, IQ4_NL) }
func BenchmarkDequantIQ4_XS(b *testing.B)  { benchDequantRaw(b, IQ4_XS) }
func BenchmarkDequantIQ1_S(b *testing.B)   { benchDequantRaw(b, IQ1_S) }
func BenchmarkDequantIQ1_M(b *testing.B)   { benchDequantRaw(b, IQ1_M) }
func BenchmarkDequantMXFP4(b *testing.B)   { benchDequantRaw(b, MXFP4) }

func benchQuantize(b *testing.B, qt QuantType) {
	t := tensor.FromFloat32(tensor.Shape{benchN}, benchF32(benchN))
	b.SetBytes(4 * benchN)
	b.ResetTimer()
	for range b.N {
		if _, err := Quantize(t, qt); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQuantizeQ8_0(b *testing.B) { benchQuantize(b, Q8_0) }
func BenchmarkQuantizeQ4_K(b *testing.B) { benchQuantize(b, Q4_K) }

// decodeTensor F16 path: the per-element f16ToF32 conversion.
func BenchmarkDecodeF32(b *testing.B) {
	raw := make([]byte, 4*benchN)
	for i := range raw {
		raw[i] = byte(i*7 + 1)
	}
	ti := tensorInfo{name: "t", shape: tensor.Shape{benchN}, ggType: tF32}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for range b.N {
		if _, err := decodeTensor(ti, raw); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDecodeF16(b *testing.B) {
	raw := make([]byte, 2*benchN)
	for i := range benchN {
		raw[2*i] = byte(i)
		raw[2*i+1] = byte(i >> 3 & 0x3F) // varied exponents, no inf/nan needed
	}
	ti := tensorInfo{name: "t", shape: tensor.Shape{benchN}, ggType: tF16}
	b.SetBytes(int64(len(raw)))
	b.ResetTimer()
	for range b.N {
		if _, err := decodeTensor(ti, raw); err != nil {
			b.Fatal(err)
		}
	}
}

func benchQMatMul(b *testing.B, m int, qt QuantType) {
	const n, k = 64, 1024
	w := tensor.FromFloat32(tensor.Shape{n * k}, benchF32(n*k))
	raw, err := Quantize(w, qt)
	if err != nil {
		b.Fatal(err)
	}
	xv := benchF32(m * k)
	x := tensor.FromFloat32(tensor.Shape{m, k}, xv)
	b.SetBytes(int64(m) * n * k * 4)
	b.ResetTimer()
	for range b.N {
		if _, err := QMatMul(x, raw, qt, n, k); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMatMulQ8_0_M1(b *testing.B)  { benchQMatMul(b, 1, Q8_0) }
func BenchmarkQMatMulQ4_K_M1(b *testing.B)  { benchQMatMul(b, 1, Q4_K) }
func BenchmarkQMatMulQ4_K_M16(b *testing.B) { benchQMatMul(b, 16, Q4_K) }

// Write path: F64 tensor forced through the f32Data conversion.
func BenchmarkWriteF64Tensor(b *testing.B) {
	vals := make([]float64, benchN)
	for i := range vals {
		vals[i] = float64(i)
	}
	f := &File{
		Version:  3,
		Metadata: map[string]any{},
		Tensors:  map[string]*tensor.Tensor{"w": tensor.FromFloat64(tensor.Shape{benchN}, vals)},
	}
	b.SetBytes(4 * benchN)
	b.ResetTimer()
	for range b.N {
		if err := Write(io.Discard, f); err != nil {
			b.Fatal(err)
		}
	}
}
