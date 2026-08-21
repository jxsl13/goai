package gguf

import (
	"bytes"
	"fmt"
	"maps"
	"os"
	"reflect"
	"runtime"
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

func TestOpenRawMatchesBuffered(t *testing.T) {
	path := writeSynthModel(t, 12, 256, 512)
	buffered, err := openRawFile(path, false)
	if err != nil {
		t.Fatal(err)
	}
	defer buffered.Close()
	mapped, err := OpenRaw(path)
	if err != nil {
		t.Fatal(err)
	}
	defer mapped.Close()

	if !reflect.DeepEqual(buffered.Metadata, mapped.Metadata) {
		t.Fatalf("metadata differs\nbuffered: %#v\nmapped:   %#v", buffered.Metadata, mapped.Metadata)
	}
	if len(buffered.Tensors) != len(mapped.Tensors) {
		t.Fatalf("tensor count %d != %d", len(buffered.Tensors), len(mapped.Tensors))
	}
	for name, want := range buffered.Tensors {
		got, ok := mapped.Tensors[name]
		if !ok {
			t.Fatalf("mapped result is missing %q", name)
		}
		if got.GGType != want.GGType || !got.Shape.Equal(want.Shape) || !bytes.Equal(got.Data, want.Data) {
			t.Fatalf("tensor %q differs: type %d/%d shape %v/%v bytes-equal=%v",
				name, got.GGType, want.GGType, got.Shape, want.Shape, bytes.Equal(got.Data, want.Data))
		}
		if cap(got.Data) != len(got.Data) {
			t.Fatalf("tensor %q mapping view has cap %d, want len %d", name, cap(got.Data), len(got.Data))
		}
	}
	switch runtime.GOOS {
	case "darwin", "dragonfly", "freebsd", "linux", "netbsd", "openbsd":
		if mapped.owner == nil {
			t.Fatal("OpenRaw used the buffered fallback for a regular non-empty temporary file")
		}
	}
}

func TestOpenRawMappingLifetime(t *testing.T) {
	path := writeSynthModel(t, 4, 128, 64)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	releases := 0
	rf, err := openRawMapped(data, func() error {
		releases++
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if releases != 0 {
		t.Fatalf("mapping released before Close: %d", releases)
	}
	if err := rf.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rf.Close(); err != nil {
		t.Fatal(err)
	}
	if releases != 1 {
		t.Fatalf("Close released mapping %d times, want exactly once", releases)
	}
	copyOfHandle := *rf
	if err := copyOfHandle.Close(); err != nil {
		t.Fatal(err)
	}
	if releases != 1 {
		t.Fatalf("closing a RawFileHandle value-copy released mapping %d times, want exactly once", releases)
	}

	releases = 0
	if _, err := openRawMapped(data[:len(data)/2], func() error {
		releases++
		return nil
	}); err == nil {
		t.Fatal("truncated mapping parsed without error")
	}
	if releases != 1 {
		t.Fatalf("failed parse released mapping %d times, want exactly once", releases)
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

// RawFileHandle keeps an OpenRaw mapping alive until Close. Keep the handle open for as long as
// the model retains any QuantTensor.Data view.
func ExampleRawFileHandle() {
	w := tensor.FromFloat64(tensor.Shape{4, 32}, randF32Block(128, 0))
	f, _ := os.CreateTemp("", "goai-openraw-*.gguf")
	defer os.Remove(f.Name())
	_ = WriteQuantized(f, &File{Version: 3, Metadata: map[string]any{"a": uint32(1)},
		Tensors: map[string]*tensor.Tensor{"w": w}}, map[string]QuantType{"w": Q8_0})
	_ = f.Close()

	raw, _ := OpenRaw(f.Name())
	defer raw.Close()
	fmt.Println(raw.Tensors["w"].Shape, len(raw.Tensors["w"].Data))
	// Output: (4, 32) 136
}
