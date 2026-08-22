package gguf

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func TestQMatMulQ1MatchesDequantizedReference(t *testing.T) {
	const n, k = 5, 4 * q1BlockElems
	raw := makeQ1Raw(n * k)
	weights := make([]float32, n*k)
	dequantQ1_0Into(weights, raw)
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
			got, err := QMatMul(x, raw, Q1_0, n, k)
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

func TestQMatMulQ1RejectsInvalidInputs(t *testing.T) {
	const n, k = 3, q1BlockElems
	raw := makeQ1Raw(n * k)
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k - 1}), raw, Q1_0, n, k); err == nil {
		t.Fatal("QMatMul accepted an activation width different from K")
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw[:len(raw)-1], Q1_0, n, k); err == nil {
		t.Fatal("QMatMul accepted a truncated Q1_0 weight matrix")
	}
}

var q1QMatMulSink *tensor.Tensor

func TestQMatMulQ1ScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, q1BlockElems
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeQ1Raw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			q1QMatMulSink, err = QMatMul(x, raw, Q1_0, n, k)
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

func TestQMatMulQ1SelectorScope(t *testing.T) {
	const n, k = 3, q1BlockElems
	raw := makeQ1Raw(n * k)
	old := dotQ1RowFn
	defer func() { dotQ1RowFn = old }()
	calls := 0
	dotQ1RowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotQ1Row(row, weight, width)
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, Q1_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}
	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, Q1_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, Q1_0, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}
