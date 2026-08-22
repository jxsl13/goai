package gguf

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func TestQMatMulTQ2MatchesDequantizedReference(t *testing.T) {
	const n, k = 5, 4 * tq2BlockElems
	raw := makeTQ2Raw(n * k)
	weights := make([]float32, n*k)
	dequantTQ2_0Into(weights, raw)
	for _, tc := range []struct {
		name  string
		dtype tensor.Dtype
		m     int
	}{{"F32_M1", tensor.F32, 1}, {"F32_M3", tensor.F32, 3}, {"F64_M1", tensor.F64, 1}, {"F64_M3", tensor.F64, 3}} {
		t.Run(tc.name, func(t *testing.T) {
			x := tensor.New(tc.dtype, tensor.Shape{tc.m, k})
			for mi := range tc.m {
				for ki := range k {
					x.SetF64(math.Sin(float64(mi*k+ki)*0.017), mi, ki)
				}
			}
			got, err := QMatMul(x, raw, TQ2_0, n, k)
			if err != nil {
				t.Fatal(err)
			}
			for mi := range tc.m {
				for ni := range n {
					var want float64
					for ki := range k {
						want += x.AtF64(mi, ki) * float64(weights[ni*k+ki])
					}
					if diff := math.Abs(got.AtF64(mi, ni) - float64(float32(want))); diff > 1e-5*(math.Abs(want)+1) {
						t.Fatalf("m=%d n=%d: got %g, want %g", mi, ni, got.AtF64(mi, ni), float32(want))
					}
				}
			}
		})
	}
}

func TestQMatMulTQ2RejectsInvalidInputs(t *testing.T) {
	const n, k = 3, tq2BlockElems
	raw := makeTQ2Raw(n * k)
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k - 1}), raw, TQ2_0, n, k); err == nil {
		t.Fatal("QMatMul accepted an activation width different from K")
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw[:len(raw)-1], TQ2_0, n, k); err == nil {
		t.Fatal("QMatMul accepted a truncated TQ2_0 weight matrix")
	}
}

var tq2QMatMulSink *tensor.Tensor

func TestQMatMulTQ2ScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, tq2BlockElems
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeTQ2Raw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			tq2QMatMulSink, err = QMatMul(x, raw, TQ2_0, n, k)
			if err != nil {
				panic(err)
			}
		})
	}
	one, many := allocs(1), allocs(31)
	if many != one {
		t.Fatalf("QMatMul allocations scale with output rows: N1=%g N31=%g", one, many)
	}
}

func TestQMatMulTQ2SelectorScope(t *testing.T) {
	const n, k = 3, tq2BlockElems
	raw := makeTQ2Raw(n * k)
	old := dotTQ2RowFn
	defer func() { dotTQ2RowFn = old }()
	calls := 0
	dotTQ2RowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotTQ2Row(row, weight, width)
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, TQ2_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}
	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, TQ2_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, TQ2_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}

func benchQMatMulTQ2NK(b *testing.B, m, n, k int) {
	x := tensor.FromFloat32(tensor.Shape{m, k}, benchF32(m*k))
	raw := makeTQ2Raw(n * k)
	b.SetBytes(int64(m * n * k * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for b.Loop() {
		var err error
		tq2QMatMulSink, err = QMatMul(x, raw, TQ2_0, n, k)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkQMatMulTQ2_0_M1(b *testing.B) { benchQMatMulTQ2NK(b, 1, 64, 1024) }

func BenchmarkQMatMulTQ2_0_M1_N4096(b *testing.B) { benchQMatMulTQ2NK(b, 1, 4096, 1024) }

func BenchmarkQMatMulTQ2_0_M16(b *testing.B) { benchQMatMulTQ2NK(b, 16, 64, 1024) }
