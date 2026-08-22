package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"math/rand"
	"slices"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeQ1Raw(n int) []byte {
	raw := make([]byte, n/q1BlockElems*q1BlockSize)
	for b := 0; b*q1BlockElems < n; b++ {
		blk := raw[b*q1BlockSize : (b+1)*q1BlockSize]
		binary.LittleEndian.PutUint16(blk, f32ToF16([]float32{0.03125, -0.0625, 0.125, 0.5}[b%4]))
		for i := range q1BlockElems {
			if (b*131+i*29+7)%5 >= 2 {
				blk[2+i/8] |= 1 << uint(i%8)
			}
		}
	}
	return raw
}

func TestQ1FormatAPIsMatchPinnedLayout(t *testing.T) {
	if Q1_0 != 41 || q1BlockElems != 128 || q1BlockSize != 18 {
		t.Fatalf("Q1_0 layout = type %d, elems %d, bytes %d", Q1_0, q1BlockElems, q1BlockSize)
	}
	raw := make([]byte, q1BlockSize)
	binary.LittleEndian.PutUint16(raw, f32ToF16(0.5))
	raw[2] = 0x81

	want := make([]float32, q1BlockElems)
	for i := range want {
		want[i] = -0.5
	}
	want[0], want[7] = 0.5, 0.5

	public, err := Dequantize(raw, Q1_0, q1BlockElems)
	if err != nil {
		t.Fatal(err)
	}
	rawTensor, err := (QuantTensor{Data: raw, GGType: tQ1_0, Shape: tensor.Shape{q1BlockElems}}).Dequantize()
	if err != nil {
		t.Fatal(err)
	}
	eager, err := decodeTensor(tensorInfo{name: "q1", shape: tensor.Shape{q1BlockElems}, ggType: tQ1_0}, raw)
	if err != nil {
		t.Fatal(err)
	}
	for name, got := range map[string][]float32{
		"Dequantize":             public.Storage().F32(),
		"QuantTensor.Dequantize": rawTensor.Storage().F32(),
		"decodeTensor":           eager.Storage().F32(),
	} {
		for i, v := range want {
			if math.Float32bits(got[i]) != math.Float32bits(v) {
				t.Fatalf("%s weight %d = %g (%08x), want %g (%08x)", name, i, got[i], math.Float32bits(got[i]), v, math.Float32bits(v))
			}
		}
	}
}

func TestQuantizeQ1MatchesPinnedReferenceLayout(t *testing.T) {
	x := make([]float32, 2*q1BlockElems)
	for i := range q1BlockElems {
		if i < q1BlockElems/2 {
			x[i] = 2
		} else {
			x[i] = -2
		}
		if i%2 == 0 {
			x[q1BlockElems+i] = 1
		} else {
			x[q1BlockElems+i] = -3
		}
	}
	before := slices.Clone(x)
	raw, err := Quantize(tensor.FromFloat32(tensor.Shape{len(x)}, x), Q1_0)
	if err != nil {
		t.Fatal(err)
	}
	want := make([]byte, 2*q1BlockSize)
	for b := range 2 {
		binary.LittleEndian.PutUint16(want[b*q1BlockSize:], f32ToF16(2))
	}
	for i := range 8 {
		want[2+i] = 0xff
		want[q1BlockSize+2+i] = 0x55
		want[q1BlockSize+2+8+i] = 0x55
	}
	if !bytes.Equal(raw, want) {
		t.Fatalf("Q1_0 bytes = %x, want %x", raw, want)
	}
	if !slices.Equal(x, before) {
		t.Fatal("Quantize modified its input")
	}
}

func TestDequantQ1IntoMatchesTensorDecoderExactly(t *testing.T) {
	const n = 3 * q1BlockElems
	raw := makeQ1Raw(n)
	want, err := dequantQ1_0(tensor.Shape{n}, raw)
	if err != nil {
		t.Fatal(err)
	}
	got := make([]float32, n)
	dequantQ1_0Into(got, raw)
	for i, v := range want.Storage().F32() {
		if math.Float32bits(got[i]) != math.Float32bits(v) {
			t.Fatalf("weight %d = %g (%08x), want %g (%08x)", i, got[i], math.Float32bits(got[i]), v, math.Float32bits(v))
		}
	}
}

func TestDotQ1RowMatchesMaterializedReferenceExactly(t *testing.T) {
	const k = 8 * q1BlockElems
	raw := makeQ1Raw(k)
	row := make([]float32, k)
	rng := rand.New(rand.NewSource(20260822))
	for i := range row {
		row[i] = float32(rng.NormFloat64())
	}
	weights := make([]float32, k)
	dequantQ1_0Into(weights, raw)
	var want float64
	for i, w := range weights {
		want += float64(row[i]) * float64(w)
	}
	got := dotQ1Row(row, raw, k)
	if math.Float64bits(got) != math.Float64bits(want) {
		t.Fatalf("fused scalar %g (%016x), materialized %g (%016x)", got, math.Float64bits(got), want, math.Float64bits(want))
	}
}

func TestQ1RejectsInvalidInputs(t *testing.T) {
	if _, err := Dequantize(make([]byte, q1BlockSize), Q1_0, q1BlockElems-1); err == nil {
		t.Fatal("Dequantize accepted a non-block-aligned element count")
	}
	if _, err := Dequantize(make([]byte, q1BlockSize-1), Q1_0, q1BlockElems); err == nil {
		t.Fatal("Dequantize accepted a truncated block")
	}
	if _, err := Quantize(tensor.New(tensor.F32, tensor.Shape{q1BlockElems - 1}), Q1_0); err == nil {
		t.Fatal("Quantize accepted a non-block-aligned tensor")
	}
}
