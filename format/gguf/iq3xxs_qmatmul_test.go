package gguf

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeIQ3XXSRaw(n int) []byte {
	raw := make([]byte, n/qkK*iq3xxsBlockSize)
	for b := 0; b*qkK < n; b++ {
		blk := raw[b*iq3xxsBlockSize : (b+1)*iq3xxsBlockSize]
		//perfscan:ignore PS4001 test fixture writes one f16 scale per strided 98-byte quant block
		binary.LittleEndian.PutUint16(blk, f32ToF16([]float32{0.03125, -0.0625, 0.125, 0.5}[b%4]))
		for i := 2; i < len(blk); i++ {
			blk[i] = byte((b*139 + i*67 + 23) & 0xff)
		}
	}
	return raw
}

func TestDequantIQ3XXSIntoMatchesTensorDecoderExactly(t *testing.T) {
	const n = 3 * qkK
	raw := makeIQ3XXSRaw(n)
	want, err := dequantIQ3_XXS(tensor.Shape{n}, raw)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, n)
	dequantIQ3_XXSInto(got, raw)
	for i, v := range want.Storage().F32() {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("weight %d: caller-owned decode %g (%08x), tensor decode %g (%08x)", i, got[i], math.Float32bits(got[i]), v, math.Float32bits(v))
		}
	}
}

func TestDotIQ3XXSRowMatchesMaterializedReferenceExactly(t *testing.T) {
	const k = 4 * qkK
	raw := makeIQ3XXSRaw(k)
	row := make([]float32, k)
	rng := rand.New(rand.NewSource(20260822))
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	weights := make([]float32, k)
	dequantIQ3_XXSInto(weights, raw)
	var want float64
	for i, w := range weights {
		want += float64(row[i]) * float64(w)
	}
	got := dotIQ3XXSRow(row, raw, k)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("fused scalar %g (%016x), materialized %g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestQMatMulIQ3XXSMatchesDequantizedReference(t *testing.T) {
	const n, k = 5, 2 * qkK
	raw := makeIQ3XXSRaw(n * k)
	weights := make([]float32, n*k)
	dequantIQ3_XXSInto(weights, raw)
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
			got, err := QMatMul(x, raw, IQ3_XXS, n, k)
			if err != nil {
				t.Fatal(err)
			}
			for mi := range tc.m {
				for ni := range n {
					var want float64
					for ki := range k {
						want += x.AtF64(mi, ki) * float64(weights[ni*k+ki])
					}
					gotv := got.AtF64(mi, ni)
					if diff := math.Abs(gotv - float64(float32(want))); diff > 1e-5*(math.Abs(want)+1) {
						t.Fatalf("m=%d n=%d: got %g, want %g", mi, ni, gotv, float32(want))
					}
				}
			}
		})
	}
}

var iq3XXSQMatMulSink *tensor.Tensor

func TestQMatMulIQ3XXSScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, qkK
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeIQ3XXSRaw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			iq3XXSQMatMulSink, err = QMatMul(x, raw, IQ3_XXS, n, k)
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

func TestQMatMulIQ3XXSSelectorScope(t *testing.T) {
	const n, k = 3, qkK
	raw := makeIQ3XXSRaw(n * k)
	old := dotIQ3XXSRowFn
	defer func() { dotIQ3XXSRowFn = old }()
	calls := 0
	dotIQ3XXSRowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotIQ3XXSRow(row, weight, width)
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, IQ3_XXS, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}
	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, IQ3_XXS, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, IQ3_XXS, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}
