package gguf

import (
	"bytes"
	"fmt"
	"maps"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §V15 (§T150): ReadRaw keeps each tensor in its quantized byte form; dequantizing a QuantTensor
// reproduces EXACTLY the tensor Read would have produced (Read = parse + decode; ReadRaw = parse
// + keep-bytes, and QuantTensor.Dequantize runs the same decode). Covers a quantized (Q8_0) and
// an F32 tensor, and checks the raw bytes are the compact quantized form, not an expanded float.
func TestReadRawRoundTrip(t *testing.T) {
	q := tensor.FromFloat64(tensor.Shape{4, 32}, randF32Block(128, 1))
	fl := tensor.FromFloat64(tensor.Shape{3, 5}, randF32Block(15, 2))
	file := &File{
		Version:  3,
		Metadata: map[string]any{"general.architecture": "test", "n": uint32(7)},
		Tensors:  map[string]*tensor.Tensor{"qw": q, "fw": fl},
	}
	var buf bytes.Buffer
	if err := WriteQuantized(&buf, file, map[string]QuantType{"qw": Q8_0}); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()

	dq, err := Read(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	rf, err := ReadRaw(bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}

	if !maps.Equal(rf.Metadata, dq.Metadata) {
		t.Errorf("ReadRaw metadata %v != Read %v", rf.Metadata, dq.Metadata)
	}
	for name, want := range dq.Tensors {
		qt, ok := rf.Tensors[name]
		if !ok {
			t.Fatalf("ReadRaw missing tensor %s", name)
		}
		got, err := qt.Dequantize()
		if err != nil {
			t.Fatalf("%s Dequantize: %v", name, err)
		}
		if !got.Shape().Equal(want.Shape()) {
			t.Errorf("%s shape %v != %v", name, got.Shape(), want.Shape())
		}
		for i := range want.Numel() {
			idx := tensor.Unravel(i, want.Shape())
			if got.AtF64(idx...) != want.AtF64(idx...) {
				t.Errorf("%s[%v]: ReadRaw %v != Read %v", name, idx, got.AtF64(idx...), want.AtF64(idx...))
				break
			}
		}
	}
	// the quantized tensor stayed quantized: right type and compact byte count (not expanded F32)
	if rf.Tensors["qw"].GGType != uint32(Q8_0) {
		t.Errorf("qw GGType %d != Q8_0 (%d)", rf.Tensors["qw"].GGType, Q8_0)
	}
	if got, want := len(rf.Tensors["qw"].Data), 128/32*34; got != want {
		t.Errorf("qw raw %d bytes != %d (Q8_0 blocks)", got, want)
	}
}

// FuzzReadRaw: no byte stream may panic ReadRaw (it shares parse with Read, plus the per-tensor
// byte slicing).
func FuzzReadRaw(f *testing.F) {
	if raw, err := os.ReadFile("testdata/sample.gguf"); err == nil {
		f.Add(raw)
	}
	f.Add([]byte("GGUF\x03\x00\x00\x00"))
	f.Add([]byte{})
	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("ReadRaw panicked on %d bytes: %v", len(data), r)
			}
		}()
		rf, err := ReadRaw(bytes.NewReader(data))
		if err == nil { // a valid parse: dequantizing each tensor must also not panic
			for _, qt := range rf.Tensors {
				_, _ = qt.Dequantize()
			}
		}
	})
}

// ReadRaw loads a GGUF keeping every tensor in its quantized byte form (QuantTensor), so a
// quantized model runs without ever materializing full-precision weights.
func ExampleReadRaw() {
	w := tensor.FromFloat64(tensor.Shape{4, 32}, randF32Block(128, 0))
	var buf bytes.Buffer
	_ = WriteQuantized(&buf, &File{Version: 3, Metadata: map[string]any{"a": uint32(1)},
		Tensors: map[string]*tensor.Tensor{"w": w}}, map[string]QuantType{"w": Q8_0})
	rf, _ := ReadRaw(bytes.NewReader(buf.Bytes()))
	qt := rf.Tensors["w"]
	fmt.Println(qt.Shape, len(qt.Data), "quantized bytes (vs", 128*4, "f32)")
	// Output: (4, 32) 136 quantized bytes (vs 512 f32)
}
