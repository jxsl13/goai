package gguf

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

func quantSampleFile() *File {
	return &File{
		Version: 3,
		Metadata: map[string]any{
			"general.architecture": "demo",
			"general.alignment":    uint32(32),
		},
		Tensors: map[string]*tensor.Tensor{
			"weight.q8": tensor.FromFloat64(tensor.Shape{2, 32}, randF32Block(64, 11)),
			"weight.q4": tensor.FromFloat64(tensor.Shape{64}, randF32Block(64, 12)),
			"norm.f32":  tensor.FromFloat64(tensor.Shape{8}, randF32Block(8, 13)),
		},
	}
}

// §V15 (§T123): a tensor written quantized reads back exactly as Dequantize(Quantize(·))
// — the write path stores the Quantize bytes and Read runs the same dequant — while an
// un-quantized tensor round-trips at full F32 precision.
func TestWriteQuantizedRoundTrip(t *testing.T) {
	f := quantSampleFile()
	quant := map[string]QuantType{"weight.q8": Q8_0, "weight.q4": Q4_0}

	var buf bytes.Buffer
	if err := WriteQuantized(&buf, f, quant); err != nil {
		t.Fatal(err)
	}
	back, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}

	for name, qt := range quant {
		orig := f.Tensors[name]
		q, _ := Quantize(orig, qt)
		want, _ := Dequantize(q, qt, orig.Numel())
		got := back.Tensors[name]
		if !got.Shape().Equal(orig.Shape()) {
			t.Errorf("%s shape %v != %v", name, got.Shape(), orig.Shape())
		}
		var maxErr, changed float64
		for i := range orig.Numel() {
			gi := got.AtF64(tensor.Unravel(i, got.Shape())...)
			wi := want.Storage().F32()[i]
			maxErr = math.Max(maxErr, math.Abs(gi-float64(wi)))
			changed = math.Max(changed, math.Abs(gi-orig.AtF64(tensor.Unravel(i, orig.Shape())...)))
		}
		if maxErr > 1e-6 {
			t.Errorf("%s: read-back differs from Dequantize(Quantize) by %g", name, maxErr)
		}
		if changed == 0 {
			t.Errorf("%s: written quantized but read-back is bit-exact — not quantized", name)
		}
	}
	// the un-quantized tensor round-trips exactly at F32 precision (the writer stores F32)
	nf := f.Tensors["norm.f32"]
	bf := back.Tensors["norm.f32"]
	for i := range nf.Numel() {
		if float32(bf.AtF64(i)) != float32(nf.AtF64(i)) {
			t.Errorf("norm.f32 [%d] not F32-exact: %v != %v", i, bf.AtF64(i), nf.AtF64(i))
		}
	}
}

// §V15: quantizing shrinks the file — a Q8_0 tensor takes ~34/128 of its F32 bytes.
func TestWriteQuantizedSmaller(t *testing.T) {
	f := quantSampleFile()
	var full, quant bytes.Buffer
	if err := Write(&full, f); err != nil {
		t.Fatal(err)
	}
	if err := WriteQuantized(&quant, f, map[string]QuantType{"weight.q8": Q8_0, "weight.q4": Q4_0}); err != nil {
		t.Fatal(err)
	}
	if quant.Len() >= full.Len() {
		t.Errorf("quantized file %d bytes not smaller than F32 %d", quant.Len(), full.Len())
	}
}

// Writing the same file+quant twice is byte-deterministic.
func TestWriteQuantizedDeterministic(t *testing.T) {
	f := quantSampleFile()
	q := map[string]QuantType{"weight.q8": Q8_0}
	var a, b bytes.Buffer
	if err := WriteQuantized(&a, f, q); err != nil {
		t.Fatal(err)
	}
	if err := WriteQuantized(&b, f, q); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Error("two quantized writes differ")
	}
}

// FuzzWriteQuantizedRoundTrip: a file whose tensor is quantized on write reads back as
// Dequantize(Quantize(·)) exactly, and the writer never errors on valid input (§V15).
func FuzzWriteQuantizedRoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5})
	f.Add([]byte("quant-write-fuzz"))
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float64, blockElems)
		for i := range vals {
			if i < len(data) {
				vals[i] = float64(int8(data[i])) * 0.5
			}
		}
		w := tensor.FromFloat64(tensor.Shape{blockElems}, vals)
		file := &File{Version: 3, Metadata: map[string]any{"general.architecture": "fz"},
			Tensors: map[string]*tensor.Tensor{"w": w}}
		var buf bytes.Buffer
		if err := WriteQuantized(&buf, file, map[string]QuantType{"w": Q8_0}); err != nil {
			t.Fatal(err)
		}
		back, err := Read(&buf)
		if err != nil {
			t.Fatal(err)
		}
		q, _ := Quantize(w, Q8_0)
		want, _ := Dequantize(q, Q8_0, blockElems)
		for i := range blockElems {
			if math.Abs(back.Tensors["w"].AtF64(i)-float64(want.Storage().F32()[i])) > 1e-6 {
				t.Fatalf("[%d]: read-back != Dequantize(Quantize)", i)
			}
		}
	})
}

// A model can be written to a GGUF file with its weight matrices quantized to Q8_0,
// producing a compact file that still reads back (dequantized) into usable weights.
func ExampleWriteQuantized() {
	f := &File{
		Version:  3,
		Metadata: map[string]any{"general.architecture": "demo"},
		Tensors:  map[string]*tensor.Tensor{"w": tensor.FromFloat64(tensor.Shape{32}, randF32Block(32, 0))},
	}
	var buf bytes.Buffer
	_ = WriteQuantized(&buf, f, map[string]QuantType{"w": Q8_0})
	back, _ := Read(&buf)
	fmt.Println(back.Tensors["w"].Numel(), "values read back")
	// Output: 32 values read back
}
