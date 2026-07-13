package gguf

import (
	"bytes"
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// metaEqual compares two metadata values for bit-exact round-trip equality. Floats are
// compared by their bit pattern so a NaN metadata value (a valid f32/f64 bit pattern)
// counts as round-tripped — value equality would wrongly report NaN≠NaN.
func metaEqual(a, b any) bool {
	switch av := a.(type) {
	case float32:
		bv, ok := b.(float32)
		return ok && math.Float32bits(av) == math.Float32bits(bv)
	case float64:
		bv, ok := b.(float64)
		return ok && math.Float64bits(av) == math.Float64bits(bv)
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !metaEqual(av[i], bv[i]) {
				return false
			}
		}
		return true
	default:
		return a == b
	}
}

// filesEqual compares two parsed GGUF files for round-trip equality: version,
// metadata (deep), and every tensor's shape + F32 contents.
func filesEqual(t *testing.T, a, b *File) {
	t.Helper()
	if a.Version != b.Version {
		t.Errorf("version %d != %d", a.Version, b.Version)
	}
	if len(a.Metadata) != len(b.Metadata) {
		t.Errorf("metadata count %d != %d", len(a.Metadata), len(b.Metadata))
	}
	for k, av := range a.Metadata {
		if bv, ok := b.Metadata[k]; !ok || !metaEqual(av, bv) {
			t.Errorf("metadata %q: %#v != %#v", k, av, b.Metadata[k])
		}
	}
	if len(a.Tensors) != len(b.Tensors) {
		t.Fatalf("tensor count %d != %d", len(a.Tensors), len(b.Tensors))
	}
	for name, ta := range a.Tensors {
		tb, ok := b.Tensors[name]
		if !ok {
			t.Errorf("tensor %q missing after round-trip", name)
			continue
		}
		if !ta.Shape().Equal(tb.Shape()) {
			t.Errorf("tensor %q shape %v != %v", name, ta.Shape(), tb.Shape())
			continue
		}
		// the writer stores F32, so round-trip equality holds at F32 precision
		for i := range ta.Numel() {
			idx := tensor.Unravel(i, ta.Shape())
			if va, vb := float32(ta.AtF64(idx...)), float32(tb.AtF64(idx...)); va != vb {
				t.Errorf("tensor %q[%d]: %v != %v", name, i, va, vb)
				break
			}
		}
	}
}

// §V15: reading the golden sample, writing it, and reading it back reproduces the
// file — metadata types and tensor values survive the Read→Write→Read cycle.
func TestWriteRoundTripSample(t *testing.T) {
	orig, err := ReadFile("testdata/sample.gguf")
	if err != nil {
		t.Fatal(err)
	}
	var buf bytes.Buffer
	if err := Write(&buf, orig); err != nil {
		t.Fatal(err)
	}
	back, err := Read(&buf)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	filesEqual(t, orig, back)
}

// §V15: a file exercising all 13 metadata value types (including a nested array of
// arrays) and several tensor shapes round-trips exactly.
func TestWriteRoundTripAllTypes(t *testing.T) {
	f := &File{
		Version: 3,
		Metadata: map[string]any{
			"general.alignment": uint32(32),
			"k.u8":              uint8(200),
			"k.i8":              int8(-42),
			"k.u16":             uint16(60000),
			"k.i16":             int16(-3000),
			"k.u32":             uint32(4000000000),
			"k.i32":             int32(-123456),
			"k.f32":             float32(3.5),
			"k.bool":            true,
			"k.string":          "héllo gguf",
			"k.u64":             uint64(1) << 40,
			"k.i64":             int64(-(1 << 40)),
			"k.f64":             float64(2.718281828),
			"k.arr_i32":         []any{int32(1), int32(2), int32(3)},
			"k.arr_str":         []any{"a", "bb", "ccc"},
			"k.arr_empty":       []any{},
			"k.arr_nested":      []any{[]any{int32(1), int32(2)}, []any{int32(3)}},
		},
		Tensors: map[string]*tensor.Tensor{
			"scalar":  tensor.FromFloat64(tensor.Shape{1}, []float64{42}),
			"vec":     tensor.FromFloat64(tensor.Shape{4}, []float64{1, -2, 3.5, 0.25}),
			"mat.2x3": tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6}),
		},
	}
	var buf bytes.Buffer
	if err := Write(&buf, f); err != nil {
		t.Fatal(err)
	}
	back, err := Read(&buf)
	if err != nil {
		t.Fatalf("read back: %v", err)
	}
	filesEqual(t, f, back)
}

// Writing the same file twice yields byte-identical output (sorted keys/names), the
// determinism safetensors also guarantees (§V15).
func TestWriteDeterministic(t *testing.T) {
	f, err := ReadFile("testdata/sample.gguf")
	if err != nil {
		t.Fatal(err)
	}
	var a, b bytes.Buffer
	if err := Write(&a, f); err != nil {
		t.Fatal(err)
	}
	if err := Write(&b, f); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.Bytes(), b.Bytes()) {
		t.Error("two writes of the same file differ")
	}
}

// §V15: the data section honours general.alignment — every tensor offset lands on the
// boundary, so a non-default alignment still round-trips.
func TestWriteAlignment(t *testing.T) {
	f := &File{
		Version:  3,
		Metadata: map[string]any{"general.alignment": uint32(64)},
		Tensors: map[string]*tensor.Tensor{
			"a": tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3}),
			"b": tensor.FromFloat64(tensor.Shape{5}, []float64{5, 4, 3, 2, 1}),
		},
	}
	var buf bytes.Buffer
	if err := Write(&buf, f); err != nil {
		t.Fatal(err)
	}
	back, err := Read(&buf)
	if err != nil {
		t.Fatal(err)
	}
	filesEqual(t, f, back)
}

// FuzzWriteRoundTrip builds a File from fuzz bytes, writes it, reads it back, and
// requires exact equality — the write path must never produce a file its own reader
// rejects or reads differently (§V15).
func FuzzWriteRoundTrip(f *testing.F) {
	f.Add([]byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08})
	f.Add([]byte("gguf-fuzz-seed-corpus"))
	f.Fuzz(func(t *testing.T, data []byte) {
		file := fileFromFuzz(data)
		var buf bytes.Buffer
		if err := Write(&buf, file); err != nil {
			t.Fatalf("write valid file: %v", err)
		}
		back, err := Read(&buf)
		if err != nil {
			t.Fatalf("read back own output: %v", err)
		}
		filesEqual(t, file, back)
	})
}

// fileFromFuzz derives a valid File (varied metadata types + tensors) from arbitrary
// bytes, so the writer/reader pair is exercised on many shapes without hand cases.
func fileFromFuzz(data []byte) *File {
	meta := map[string]any{
		"general.architecture": "fuzz",
		"n.bytes":              uint32(len(data)),
		"first.bool":           len(data) > 0 && data[0]&1 == 1,
	}
	arr := make([]any, 0, len(data))
	for _, b := range data {
		arr = append(arr, int8(b))
	}
	meta["byte.array"] = arr
	if len(data) >= 8 {
		meta["as.f64"] = math.Float64frombits(le64(data))
		meta["as.i64"] = int64(le64(data))
	}
	// two tensors: a vector of the byte values, and a 2-column matrix when even
	vals := make([]float64, len(data))
	for i, b := range data {
		vals[i] = float64(int8(b))
	}
	tensors := map[string]*tensor.Tensor{
		"vec": tensor.FromFloat64(tensor.Shape{len(data)}, vals),
	}
	if n := len(data); n >= 2 && n%2 == 0 {
		tensors["mat"] = tensor.FromFloat64(tensor.Shape{n / 2, 2}, vals)
	}
	return &File{Version: 3, Metadata: meta, Tensors: tensors}
}

func le64(b []byte) uint64 {
	var v uint64
	for i := range 8 {
		v |= uint64(b[i]) << (8 * i)
	}
	return v
}

// A GGUF file can be built in memory and written out, then read back unchanged.
func ExampleWrite() {
	f := &File{
		Version:  3,
		Metadata: map[string]any{"general.architecture": "demo", "answer": uint32(42)},
		Tensors:  map[string]*tensor.Tensor{"w": tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3})},
	}
	var buf bytes.Buffer
	_ = Write(&buf, f)
	back, _ := Read(&buf)
	fmt.Println(back.Metadata["general.architecture"], back.Metadata["answer"], back.Tensors["w"].Storage().F32())
	// Output: demo 42 [1 2 3]
}
