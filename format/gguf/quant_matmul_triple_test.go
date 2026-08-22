package gguf

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

var qmatmulTripleSink [3]*tensor.Tensor

func TestQMatMulTripleMatchesIndependentMixedQuantsAndShapes(t *testing.T) {
	const k = 256
	ns := [3]int{513, 257, 129}
	qts := [3]QuantType{Q4_K, Q4_K, Q6_K}
	var weights [3][]byte
	for matrix := range weights {
		w := tensor.FromFloat32(tensor.Shape{ns[matrix] * k}, benchF32(ns[matrix]*k))
		for i := range w.Storage().F32() {
			w.Storage().F32()[i] += float32(matrix) * 0.03125
		}
		var err error
		weights[matrix], err = Quantize(w, qts[matrix])
		if err != nil {
			t.Fatal(err)
		}
	}
	x := tensor.FromFloat32(tensor.Shape{1, k}, benchF32(k))
	var want [3]*tensor.Tensor
	for matrix := range want {
		var err error
		want[matrix], err = QMatMul(x, weights[matrix], qts[matrix], ns[matrix], k)
		if err != nil {
			t.Fatal(err)
		}
	}
	got, err := QMatMulTriple(x, weights, qts, ns, k)
	if err != nil {
		t.Fatal(err)
	}
	for matrix := range got {
		if !got[matrix].Shape().Equal(tensor.Shape{1, ns[matrix]}) {
			t.Fatalf("output %d shape = %v, want (1, %d)", matrix, got[matrix].Shape(), ns[matrix])
		}
		for i := range ns[matrix] {
			if math.Float32bits(got[matrix].Storage().F32()[i]) != math.Float32bits(want[matrix].Storage().F32()[i]) {
				t.Fatalf("output %d element %d differs: %08x != %08x", matrix, i, math.Float32bits(got[matrix].Storage().F32()[i]), math.Float32bits(want[matrix].Storage().F32()[i]))
			}
		}
	}
}

func TestQMatMulTripleRejectsUnsupportedInputs(t *testing.T) {
	const n, k = 1, 256
	q4 := make([]byte, k/qkK*q4kBlockSize)
	q6 := make([]byte, k/qkK*q6kBlockSize)
	weights := [3][]byte{q4, q4, q6}
	qts := [3]QuantType{Q4_K, Q4_K, Q6_K}
	ns := [3]int{n, n, n}
	x := tensor.New(tensor.F32, tensor.Shape{1, k})

	badQT := qts
	badQT[2] = Q8_0
	if _, err := QMatMulTriple(x, weights, badQT, ns, k); err == nil {
		t.Fatal("QMatMulTriple accepted Q8_0")
	}
	if _, err := QMatMulTriple(tensor.New(tensor.F64, tensor.Shape{1, k}), weights, qts, ns, k); err == nil {
		t.Fatal("QMatMulTriple accepted F64 activations")
	}
	if _, err := QMatMulTriple(tensor.New(tensor.F32, tensor.Shape{2, k}), weights, qts, ns, k); err == nil {
		t.Fatal("QMatMulTriple accepted M greater than one")
	}
	truncated := weights
	truncated[0] = truncated[0][:len(truncated[0])-1]
	if _, err := QMatMulTriple(x, truncated, qts, ns, k); err == nil {
		t.Fatal("QMatMulTriple accepted a truncated weight")
	}
	zeroN := ns
	zeroN[1] = 0
	if _, err := QMatMulTriple(x, weights, qts, zeroN, k); err == nil {
		t.Fatal("QMatMulTriple accepted zero output rows")
	}
	if _, err := QMatMulTriple(tensor.New(tensor.F32, tensor.Shape{1, 0}), weights, qts, ns, 0); err == nil {
		t.Fatal("QMatMulTriple accepted zero input columns")
	}
}

func BenchmarkQMatMulTripleMixedQKV_M1(b *testing.B) {
	const k = 2048
	ns := [3]int{2048, 256, 256}
	qts := [3]QuantType{Q4_K, Q4_K, Q6_K}
	var weights [3][]byte
	for matrix := range weights {
		w := tensor.FromFloat32(tensor.Shape{ns[matrix] * k}, benchF32(ns[matrix]*k))
		for i := range w.Storage().F32() {
			w.Storage().F32()[i] += float32(matrix) * 0.03125
		}
		var err error
		weights[matrix], err = Quantize(w, qts[matrix])
		if err != nil {
			b.Fatal(err)
		}
	}
	x := tensor.FromFloat32(tensor.Shape{1, k}, benchF32(k))
	b.Run("independent", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for range b.N {
			for matrix := range weights {
				var err error
				qmatmulTripleSink[matrix], err = QMatMul(x, weights[matrix], qts[matrix], ns[matrix], k)
				if err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	b.Run("triple", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		var err error
		for range b.N {
			qmatmulTripleSink, err = QMatMulTriple(x, weights, qts, ns, k)
			if err != nil {
				b.Fatal(err)
			}
		}
	})
}
