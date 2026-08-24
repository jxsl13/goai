package cpu_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/cpu"
	"github.com/jxsl13/goai/tensor"
)

func moeCombineInputs(rng *rand.Rand, tks, d, e int) []*tensor.Tensor {
	return moeCombineInputsDtype(rng, tensor.F64, tks, d, e)
}

func moeCombineInputsDtype(rng *rand.Rand, dtype tensor.Dtype, tks, d, e int) []*tensor.Tensor {
	f := func(shape tensor.Shape, positive bool) *tensor.Tensor {
		t := tensor.New(dtype, shape)
		for i := 0; i < t.Numel(); i++ {
			v := rng.NormFloat64()
			if positive {
				v = math.Abs(v) + 0.125
			}
			switch dtype {
			case tensor.F64:
				t.Storage().F64()[i] = v
			case tensor.F32:
				t.Storage().F32()[i] = float32(v)
			default:
				panic("unsupported benchmark dtype")
			}
		}
		return t
	}
	in := []*tensor.Tensor{f(tensor.Shape{tks, e}, true)}
	for i := 0; i < e; i++ {
		in = append(in, f(tensor.Shape{tks, d}, false))
	}
	return in
}

func requireMoECombineBitsEqual(t *testing.T, got, want *tensor.Tensor) {
	t.Helper()
	if got.Dtype() != want.Dtype() || !got.Shape().Equal(want.Shape()) {
		t.Fatalf("metadata mismatch: got %v %v, want %v %v", got.Dtype(), got.Shape(), want.Dtype(), want.Shape())
	}
	switch got.Dtype() {
	case tensor.F64:
		gs, ws := got.Storage().F64(), want.Storage().F64()
		for i := range gs {
			if math.Float64bits(gs[i]) != math.Float64bits(ws[i]) {
				t.Fatalf("idx=%d got=%#016x (%v) want=%#016x (%v)", i, math.Float64bits(gs[i]), gs[i], math.Float64bits(ws[i]), ws[i])
			}
		}
	case tensor.F32:
		gs, ws := got.Storage().F32(), want.Storage().F32()
		for i := range gs {
			if math.Float32bits(gs[i]) != math.Float32bits(ws[i]) {
				t.Fatalf("idx=%d got=%#08x (%v) want=%#08x (%v)", i, math.Float32bits(gs[i]), gs[i], math.Float32bits(ws[i]), ws[i])
			}
		}
	default:
		t.Fatalf("unsupported dtype %v", got.Dtype())
	}
}

func TestMoECombineCPUByteIdenticalToRef(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, cfg := range [][3]int{{1, 1, 1}, {1, 3, 2}, {2, 5, 3}, {3, 7, 4}, {5, 9, 8}, {17, 31, 8}} {
			in := moeCombineInputsDtype(rng, dtype, cfg[0], cfg[1], cfg[2])
			gc, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, in, nil)
			if err != nil {
				t.Fatalf("cpu dtype=%v cfg=%v: %v", dtype, cfg, err)
			}
			gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMoECombine, in, nil)
			if err != nil {
				t.Fatalf("ref dtype=%v cfg=%v: %v", dtype, cfg, err)
			}
			t.Run(fmt.Sprintf("%v/%dx%dx%d", dtype, cfg[0], cfg[1], cfg[2]), func(t *testing.T) {
				requireMoECombineBitsEqual(t, gc[0], gr[0])
			})
		}
	}
}

func TestMoECombineCPUExceptionalAndReductionOrderBits(t *testing.T) {
	for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, tc := range []struct {
			name    string
			weights []float64
			experts [][]float64
		}{
			{
				name:    "ordered-cancellation",
				weights: []float64{1, 1, 1},
				experts: [][]float64{{3e16, -0.0, 7}, {-3e16, 0, -7}, {3, -0.0, 1}},
			},
			{
				name:    "zero-denominator",
				weights: []float64{math.Copysign(0, -1), 0, 0},
				experts: [][]float64{{1, math.Inf(1), math.NaN()}, {-1, 2, 3}, {4, 5, 6}},
			},
			{
				name:    "nan-denominator",
				weights: []float64{1, math.NaN(), 2},
				experts: [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			},
			{
				name:    "infinite-denominator",
				weights: []float64{1, math.Inf(1), 2},
				experts: [][]float64{{1, -0.0, 3}, {4, 5, 6}, {7, 8, math.Inf(-1)}},
			},
		} {
			t.Run(fmt.Sprintf("%v/%s", dtype, tc.name), func(t *testing.T) {
				in := moeCombineInputsDtype(rand.New(rand.NewSource(17)), dtype, 1, len(tc.experts[0]), len(tc.weights))
				for i, v := range tc.weights {
					in[0].SetF64(v, 0, i)
				}
				for i, values := range tc.experts {
					for j, v := range values {
						in[i+1].SetF64(v, 0, j)
					}
				}
				gc, err := backend.Execute(backend.NewContext(), backend.OpMoECombine, in, nil)
				if err != nil {
					t.Fatal(err)
				}
				gr, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()), backend.OpMoECombine, in, nil)
				if err != nil {
					t.Fatal(err)
				}
				requireMoECombineBitsEqual(t, gc[0], gr[0])
			})
		}
	}
}

func benchMoECombine(b *testing.B, tks, d, e int, ref bool) {
	benchMoECombineDtype(b, tensor.F64, tks, d, e, ref)
}

func benchMoECombineDtype(b *testing.B, dtype tensor.Dtype, tks, d, e int, ref bool) {
	in := moeCombineInputsDtype(rand.New(rand.NewSource(1)), dtype, tks, d, e)
	ctx := backend.NewContext()
	if ref {
		ctx = ctx.WithBackend(backend.Reference())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMoECombine, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoECombineInterleave(b *testing.B) {
	for _, tc := range []struct {
		name      string
		tks, d, e int
	}{
		{name: "DecodeE8", tks: 1, d: 4096, e: 8},
		{name: "PrefillE8", tks: 128, d: 2048, e: 8},
		{name: "HighExpertE64", tks: 32, d: 2048, e: 64},
	} {
		for _, dtype := range []tensor.Dtype{tensor.F64, tensor.F32} {
			b.Run(fmt.Sprintf("%s/%v", tc.name, dtype), func(b *testing.B) {
				benchMoECombineDtype(b, dtype, tc.tks, tc.d, tc.e, false)
			})
		}
	}
}

func BenchmarkMoECombineInterleaveControlZeroDenom(b *testing.B) {
	in := moeCombineInputsDtype(rand.New(rand.NewSource(1)), tensor.F64, 64, 2048, 8)
	clear(in[0].Storage().F64())
	ctx := backend.NewContext()
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		if _, err := backend.Execute(ctx, backend.OpMoECombine, in, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMoECombine_cpu(b *testing.B) { benchMoECombine(b, 2048, 2048, 8, false) }
func BenchmarkMoECombine_ref(b *testing.B) { benchMoECombine(b, 2048, 2048, 8, true) }
