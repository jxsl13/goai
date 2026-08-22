package gguf

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

var qmatmulPairSink0, qmatmulPairSink1 *tensor.Tensor

func TestQMatMulPairMatchesIndependentQ4K(t *testing.T) {
	const n, k = 257, 256
	x := tensor.FromFloat32(tensor.Shape{1, k}, benchF32(k))
	weights := make([][]byte, 2)
	for pair := range weights {
		w := tensor.FromFloat32(tensor.Shape{n * k}, benchF32(n*k))
		for i := range w.Storage().F32() {
			w.Storage().F32()[i] += float32(pair) * 0.03125
		}
		var err error
		weights[pair], err = Quantize(w, Q4_K)
		if err != nil {
			t.Fatal(err)
		}
	}
	want0, err := QMatMul(x, weights[0], Q4_K, n, k)
	if err != nil {
		t.Fatal(err)
	}
	want1, err := QMatMul(x, weights[1], Q4_K, n, k)
	if err != nil {
		t.Fatal(err)
	}
	got0, got1, err := QMatMulPair(x, weights[0], weights[1], Q4_K, n, k)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		if math.Float32bits(got0.Storage().F32()[i]) != math.Float32bits(want0.Storage().F32()[i]) {
			t.Fatalf("weight0 output %d differs: %08x != %08x", i, math.Float32bits(got0.Storage().F32()[i]), math.Float32bits(want0.Storage().F32()[i]))
		}
		if math.Float32bits(got1.Storage().F32()[i]) != math.Float32bits(want1.Storage().F32()[i]) {
			t.Fatalf("weight1 output %d differs: %08x != %08x", i, math.Float32bits(got1.Storage().F32()[i]), math.Float32bits(want1.Storage().F32()[i]))
		}
	}
}

func TestQMatMulPairRejectsUnsupportedInputs(t *testing.T) {
	const n, k = 1, 256
	raw := make([]byte, k/qkK*q4kBlockSize)
	if _, _, err := QMatMulPair(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, raw, Q6_K, n, k); err == nil {
		t.Fatal("QMatMulPair accepted Q6_K")
	}
	if _, _, err := QMatMulPair(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, raw, Q4_K, n, k); err == nil {
		t.Fatal("QMatMulPair accepted F64 activations")
	}
	if _, _, err := QMatMulPair(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, raw, Q4_K, n, k); err == nil {
		t.Fatal("QMatMulPair accepted M greater than one")
	}
	if _, _, err := QMatMulPair(tensor.New(tensor.F32, tensor.Shape{1, k}), raw[:len(raw)-1], raw, Q4_K, n, k); err == nil {
		t.Fatal("QMatMulPair accepted a truncated weight")
	}
	if _, _, err := QMatMulPair(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, raw, Q4_K, 0, k); err == nil {
		t.Fatal("QMatMulPair accepted zero output rows")
	}
	if _, _, err := QMatMulPair(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, raw, Q4_K, n, 0); err == nil {
		t.Fatal("QMatMulPair accepted zero input columns")
	}
}

func BenchmarkQMatMulPairQ4K_M1_N4096(b *testing.B) {
	const n, k = 4096, 1024
	w := tensor.FromFloat32(tensor.Shape{n * k}, benchF32(n*k))
	raw0, err := Quantize(w, Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	for i := range w.Storage().F32() {
		w.Storage().F32()[i] += 0.03125
	}
	raw1, err := Quantize(w, Q4_K)
	if err != nil {
		b.Fatal(err)
	}
	x := tensor.FromFloat32(tensor.Shape{1, k}, benchF32(k))
	b.Run("independent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			qmatmulPairSink0, err = QMatMul(x, raw0, Q4_K, n, k)
			if err != nil {
				b.Fatal(err)
			}
			qmatmulPairSink1, err = QMatMul(x, raw1, Q4_K, n, k)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("paired", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			qmatmulPairSink0, qmatmulPairSink1, err = QMatMulPair(x, raw0, raw1, Q4_K, n, k)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
