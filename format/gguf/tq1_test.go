package gguf

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"math"
	"slices"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func makeTQ1Raw(n int) []byte {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(math.Sin(float64(i)*0.37) * (1 + float64(i%11)))
	}
	return quantizeTQ1_0(x)
}

func TestTQ1LayoutAndKnownBlocks(t *testing.T) {
	if TQ1_0 != 34 || tq1BlockElems != 256 || tq1BlockSize != 54 {
		t.Fatalf("TQ1_0 layout = type %d, elems %d, bytes %d", TQ1_0, tq1BlockElems, tq1BlockSize)
	}
	for _, tc := range []struct {
		name   string
		packed byte
		tail   byte
		want   float32
	}{{"negative", 0, 0, -0.5}, {"zero", 128, 127, 0}, {"positive", 255, 253, 0.5}} {
		t.Run(tc.name, func(t *testing.T) {
			raw := make([]byte, tq1BlockSize)
			for i := 0; i < tq1PackedBytes; i++ {
				raw[i] = tc.packed
			}
			for i := tq1PackedBytes; i < tq1PackedBytes+tq1TailBytes; i++ {
				raw[i] = tc.tail
			}
			binary.LittleEndian.PutUint16(raw[52:], f32ToF16(0.5))
			got, err := Dequantize(raw, TQ1_0, tq1BlockElems)
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range got.Storage().F32() {
				if v != tc.want {
					t.Fatalf("value[%d] = %v, want %v", i, v, tc.want)
				}
			}
		})
	}
}

func TestTQ1DecodeEntryPointsAgree(t *testing.T) {
	raw := makeTQ1Raw(2 * tq1BlockElems)
	shape := tensor.Shape{2, tq1BlockElems}
	public, err := Dequantize(raw, TQ1_0, shape.Numel())
	if err != nil {
		t.Fatal(err)
	}
	rawTensor, err := (QuantTensor{Data: raw, GGType: tTQ1_0, Shape: shape}).Dequantize()
	if err != nil {
		t.Fatal(err)
	}
	eager, err := decodeTensor(tensorInfo{name: "tq1", shape: shape, ggType: tTQ1_0}, raw)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(public.Storage().F32(), rawTensor.Storage().F32()) ||
		!slices.Equal(public.Storage().F32(), eager.Storage().F32()) {
		t.Fatal("TQ1_0 eager, raw, and public decoders disagree")
	}
}

func TestQuantizeTQ1MatchesPinnedReferenceBytes(t *testing.T) {
	x := make([]float32, tq1BlockElems)
	for i := range x {
		x[i] = float32((i*17+i/7)%3 - 1)
	}
	before := slices.Clone(x)
	got, err := Quantize(tensor.FromFloat32(tensor.Shape{len(x)}, x), TQ1_0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := hex.DecodeString("51d1581bf1720951d1581bf1720951d1581bf1720951d1581bf1720951d1581b3d32b69522bf663d32b69522bf663d32561af172003c")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, want) {
		t.Fatalf("TQ1_0 bytes = %x, want pinned llama.cpp bytes %x", got, want)
	}
	if !slices.Equal(x, before) {
		t.Fatal("Quantize modified its input")
	}
}

func TestDotTQ1RowMatchesMaterializedDecodeExactly(t *testing.T) {
	for _, k := range []int{tq1BlockElems, 4 * tq1BlockElems, 16 * tq1BlockElems} {
		raw := makeTQ1Raw(k)
		weights := make([]float32, k)
		dequantTQ1_0Into(weights, raw)
		row := make([]float32, k)
		for i := range row {
			row[i] = float32(math.Cos(float64(i)*0.071) * 3)
		}
		var want float64
		for i, w := range weights {
			want += float64(row[i]) * float64(w)
		}
		if got := dotTQ1Row(row, raw, k); got != want {
			t.Fatalf("K=%d fused dot = %.17g, materialized = %.17g", k, got, want)
		}
	}
}

func TestTQ1RejectsInvalidInputs(t *testing.T) {
	if _, err := Dequantize(make([]byte, tq1BlockSize), TQ1_0, tq1BlockElems-1); err == nil {
		t.Fatal("Dequantize accepted non-block-aligned TQ1_0 values")
	}
	if _, err := Dequantize(make([]byte, tq1BlockSize-1), TQ1_0, tq1BlockElems); err == nil {
		t.Fatal("Dequantize accepted truncated TQ1_0 data")
	}
	if _, err := Quantize(tensor.New(tensor.F32, tensor.Shape{tq1BlockElems - 1}), TQ1_0); err == nil {
		t.Fatal("Quantize accepted a non-block-aligned tensor")
	}
}
