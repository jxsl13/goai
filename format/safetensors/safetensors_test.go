package safetensors

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §V1: read a file written by the official Python safetensors lib.
func TestLoadPythonReference(t *testing.T) {
	ts, meta, err := LoadFile("testdata/ref.safetensors")
	if err != nil {
		t.Fatalf("load reference (run `make golden`): %v", err)
	}
	if meta["format"] != "goai-golden" {
		t.Errorf("metadata lost: %v", meta)
	}

	w := ts["w"]
	if w == nil || w.Dtype() != tensor.F64 || !w.Shape().Equal(tensor.Shape{2, 3}) {
		t.Fatalf("w: %v", w)
	}
	want := [][3]float64{{1.5, -2.0, 3.25}, {4.0, 5.5, -6.75}}
	for i := range 2 {
		for j := range 3 {
			if w.AtF64(i, j) != want[i][j] {
				t.Errorf("w(%d,%d) = %v, want %v", i, j, w.AtF64(i, j), want[i][j])
			}
		}
	}

	b := ts["b"]
	if b == nil || b.Dtype() != tensor.F32 || !b.Shape().Equal(tensor.Shape{3}) {
		t.Fatalf("b: %v", b)
	}
	if b.AtF64(0) != 0.5 || b.AtF64(2) != 2.5 {
		t.Errorf("b values wrong: %v %v", b.AtF64(0), b.AtF64(2))
	}

	s := ts["scalar"]
	if s == nil || s.Ndim() != 0 || s.AtF64() != 42 {
		t.Fatalf("scalar: %v", s)
	}
}

// Go write → Go read round-trip preserves everything (incl. non-contiguous
// input, which Save materializes).
func TestRoundTrip(t *testing.T) {
	m := tensor.FromFloat64(tensor.Shape{2, 3}, []float64{1, 2, 3, 4, 5, 6})
	tr, _ := m.Transpose(0, 1) // non-contiguous view
	in := map[string]*tensor.Tensor{
		"m":      m,
		"mt":     tr,
		"vec32":  tensor.FromFloat32(tensor.Shape{4}, []float32{1.5, -2.5, 0, 3.5}),
		"scalar": tensor.FromFloat64(tensor.Shape{}, []float64{-7}),
		"empty":  tensor.New(tensor.F64, tensor.Shape{0}),
	}
	var buf bytes.Buffer
	if err := Save(&buf, in, map[string]string{"k": "v"}); err != nil {
		t.Fatal(err)
	}
	out, meta, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if meta["k"] != "v" {
		t.Error("metadata lost")
	}
	if len(out) != len(in) {
		t.Fatalf("got %d tensors, want %d", len(out), len(in))
	}
	for name, want := range in {
		got := out[name]
		wc := want.Contiguous()
		if got.Dtype() != wc.Dtype() || !got.Shape().Equal(wc.Shape()) {
			t.Fatalf("%s: meta mismatch %v %v", name, got.Dtype(), got.Shape())
		}
		for i := range wc.Numel() {
			idx := tensor.Unravel(i, wc.Shape())
			if got.AtF64(idx...) != wc.AtF64(idx...) {
				t.Errorf("%s[%d]: %v != %v", name, i, got.AtF64(idx...), wc.AtF64(idx...))
			}
		}
	}
}

// Writer determinism: identical input → identical bytes (§V13).
func TestDeterministicWrite(t *testing.T) {
	in := map[string]*tensor.Tensor{
		"a": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2}),
		"b": tensor.FromFloat32(tensor.Shape{2}, []float32{3, 4}),
	}
	var b1, b2 bytes.Buffer
	if err := Save(&b1, in, nil); err != nil {
		t.Fatal(err)
	}
	if err := Save(&b2, in, nil); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(b1.Bytes(), b2.Bytes()) {
		t.Error("Save must be byte-deterministic")
	}
}

// Strict validation: gaps, overlaps, size mismatches, unknown dtypes, oversized
// headers all error rather than mis-read.
func TestValidation(t *testing.T) {
	mk := func(header string, dataLen int) []byte {
		var buf bytes.Buffer
		h := []byte(header)
		var l [8]byte
		for i := range 8 {
			l[i] = byte(uint64(len(h)) >> (8 * i))
		}
		buf.Write(l[:])
		buf.Write(h)
		buf.Write(make([]byte, dataLen))
		return buf.Bytes()
	}

	cases := map[string][]byte{
		"gap": mk(`{"a":{"dtype":"F64","shape":[1],"data_offsets":[8,16]}}`, 16),
		"size mismatch": mk(`{"a":{"dtype":"F64","shape":[2],"data_offsets":[0,8]}}`, 8),
		"beyond data": mk(`{"a":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}}`, 8),
		"unknown dtype": mk(`{"a":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}`, 4),
		"trailing bytes": mk(`{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]}}`, 16),
		"overlap": mk(`{"a":{"dtype":"F64","shape":[1],"data_offsets":[0,8]},"b":{"dtype":"F64","shape":[1],"data_offsets":[4,12]}}`, 12),
	}
	for name, data := range cases {
		if _, _, err := Load(bytes.NewReader(data)); err == nil {
			t.Errorf("%s: must error", name)
		}
	}

	// header cap
	var buf bytes.Buffer
	var l [8]byte
	huge := uint64(maxHeaderSize + 1)
	for i := range 8 {
		l[i] = byte(huge >> (8 * i))
	}
	buf.Write(l[:])
	if _, _, err := Load(&buf); err == nil {
		t.Error("oversized header must error")
	}
}
