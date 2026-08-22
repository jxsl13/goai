package gguf

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeIQ1MRaw(n int) []byte {
	raw := make([]byte, n/qkK*iq1mBlockSize)
	for b := 0; b*qkK < n; b++ {
		blk := raw[b*iq1mBlockSize : (b+1)*iq1mBlockSize]
		for i := range blk {
			blk[i] = byte((b*149 + i*79 + 43) & 0xff)
		}
		putIQ1MScale(blk, []float32{0.03125, -0.0625, 0.125, 0.5}[b%4])
	}
	return raw
}

func putIQ1MScale(blk []byte, scale float32) {
	dbits := f32ToF16(scale)
	for i := range 4 {
		//perfscan:ignore PS4001 fixture embeds one f16 nibble into each strided IQ1_M scale word
		w := binary.LittleEndian.Uint16(blk[48+i*2:])
		w = w&0x0fff | (dbits>>(4*i)&0xf)<<12
		binary.LittleEndian.PutUint16(blk[48+i*2:], w)
	}
}

func TestDequantIQ1MIntoMatchesTensorDecoderExactly(t *testing.T) {
	const n = 3 * qkK
	raw := makeIQ1MRaw(n)
	want, err := dequantIQ1_M(tensor.Shape{n}, raw)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, n)
	dequantIQ1_MInto(got, raw)
	for i, v := range want.Storage().F32() {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("weight %d: caller-owned decode %g (%08x), tensor decode %g (%08x)", i, got[i], math.Float32bits(got[i]), v, math.Float32bits(v))
		}
	}
}

func TestDotIQ1MRowMatchesMaterializedReferenceExactly(t *testing.T) {
	const k = 4 * qkK
	raw := makeIQ1MRaw(k)
	row := make([]float32, k)
	rng := rand.New(rand.NewSource(20260822))
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	weights := make([]float32, k)
	dequantIQ1_MInto(weights, raw)
	var want float64
	for i, w := range weights {
		want += float64(row[i]) * float64(w)
	}
	got := dotIQ1MRow(row, raw, k)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("fused scalar %g (%016x), materialized %g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestQMatMulIQ1MMatchesDequantizedReference(t *testing.T) {
	const n, k = 5, 2 * qkK
	raw := makeIQ1MRaw(n * k)
	weights := make([]float32, n*k)
	dequantIQ1_MInto(weights, raw)
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
			got, err := QMatMul(x, raw, IQ1_M, n, k)
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

func TestQMatMulIQ1MRejectsInvalidInputs(t *testing.T) {
	const n, k = 3, qkK
	raw := makeIQ1MRaw(n * k)
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k - 1}), raw, IQ1_M, n, k); err == nil {
		t.Fatal("QMatMul accepted an activation width different from K")
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw[:len(raw)-1], IQ1_M, n, k); err == nil {
		t.Fatal("QMatMul accepted a truncated IQ1_M weight matrix")
	}
}

var iq1MQMatMulSink *tensor.Tensor

func TestQMatMulIQ1MScratchAllocationsDoNotScaleWithOutputRows(t *testing.T) {
	const m, k = 2, qkK
	x := tensor.FromFloat32(tensor.Shape{m, k}, make([]float32, m*k))
	allocs := func(n int) float64 {
		raw := makeIQ1MRaw(n * k)
		return testing.AllocsPerRun(100, func() {
			var err error
			iq1MQMatMulSink, err = QMatMul(x, raw, IQ1_M, n, k)
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

func TestQMatMulIQ1MSelectorScope(t *testing.T) {
	const n, k = 3, qkK
	raw := makeIQ1MRaw(n * k)
	old := dotIQ1MRowFn
	defer func() { dotIQ1MRowFn = old }()
	calls := 0
	dotIQ1MRowFn = func(row []float32, weight []byte, width int) float64 {
		calls++
		return dotIQ1MRow(row, weight, width)
	}
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{1, k}), raw, IQ1_M, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != n {
		t.Fatalf("contiguous F32 M1 selector calls = %d, want %d", calls, n)
	}
	calls = 0
	if _, err := QMatMul(tensor.New(tensor.F32, tensor.Shape{2, k}), raw, IQ1_M, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F32 M2 dispatched M1 row leaf %d times", calls)
	}
	if _, err := QMatMul(tensor.New(tensor.F64, tensor.Shape{1, k}), raw, IQ1_M, n, k); err != nil {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("F64 M1 dispatched F32 row leaf %d times", calls)
	}
}
