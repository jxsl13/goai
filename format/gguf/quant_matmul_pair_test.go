package gguf

import (
	"math"
	"runtime"
	"sync/atomic"
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

func TestQMatMulPairApplyMatchesComposedAndAlignsChunks(t *testing.T) {
	const n, k = 5633, 256
	previous := runtime.GOMAXPROCS(4)
	defer runtime.GOMAXPROCS(previous)
	x := tensor.FromFloat32(tensor.Shape{1, k}, benchF32(k))
	w := tensor.FromFloat32(tensor.Shape{n * k}, benchF32(n*k))
	raw0, err := Quantize(w, Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	for i := range w.Storage().F32() {
		w.Storage().F32()[i] += 0.03125
	}
	raw1, err := Quantize(w, Q4_K)
	if err != nil {
		t.Fatal(err)
	}
	want, up, err := QMatMulPair(x, raw0, raw1, Q4_K, n, k)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		want.Storage().F32()[i] += up.Storage().F32()[i]
	}
	var calls, unaligned atomic.Int64
	got, err := QMatMulPairApply(x, raw0, raw1, Q4_K, n, k, func(gate, up []float32) {
		calls.Add(1)
		if len(gate)%8 != 0 {
			unaligned.Add(1)
		}
		for i := range gate {
			gate[i] += up[i]
		}
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls.Load() < 2 {
		t.Fatalf("consumer calls = %d, want parallel chunks", calls.Load())
	}
	if unaligned.Load() > 1 {
		t.Fatalf("unaligned consumer chunks = %d, want final chunk only", unaligned.Load())
	}
	for i := range n {
		if math.Float32bits(got.Storage().F32()[i]) != math.Float32bits(want.Storage().F32()[i]) {
			t.Fatalf("output %d = %08x, want %08x", i, math.Float32bits(got.Storage().F32()[i]), math.Float32bits(want.Storage().F32()[i]))
		}
	}
}

func TestQMatMulPairApplyScratchPoolSteadyState(t *testing.T) {
	values, holder := borrowQMatMulPairScratch(5632)
	releaseQMatMulPairScratch(values, holder)
	allocs := testing.AllocsPerRun(100, func() {
		values, holder := borrowQMatMulPairScratch(5632)
		releaseQMatMulPairScratch(values, holder)
	})
	if allocs != 0 {
		t.Fatalf("steady-state scratch allocs = %g, want 0", allocs)
	}
	values, holder = borrowQMatMulPairScratch(qmatmulPairScratchMaxF32 + 1)
	if holder != nil || cap(values) <= qmatmulPairScratchMaxF32 {
		t.Fatalf("oversize scratch holder=%p cap=%d, want unpooled", holder, cap(values))
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
	called := false
	if _, err := QMatMulPairApply(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, raw, Q4_K, n, k, nil); err == nil {
		t.Fatal("QMatMulPairApply accepted a nil consumer")
	}
	if _, err := QMatMulPairApply(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, raw, Q4_K, n, k, func(_, _ []float32) { called = true }); err == nil {
		t.Fatal("QMatMulPairApply accepted F64 activations")
	}
	if called {
		t.Fatal("QMatMulPairApply invoked consumer after validation failure")
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
