package npz

import (
	"bytes"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// FuzzLoad (§V15): no hostile byte stream may panic; malformed input must error.
func FuzzLoad(f *testing.F) {
	if raw, err := os.ReadFile("testdata/ref.npz"); err == nil {
		f.Add(raw)
	}
	var buf bytes.Buffer
	_ = Save(&buf, map[string]*tensor.Tensor{"x": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})})
	f.Add(buf.Bytes())
	f.Add([]byte{})
	f.Add([]byte("PK\x03\x04garbage"))

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Load panicked on %d bytes: %v", len(data), r)
			}
		}()
		_, _ = Load(bytes.NewReader(data), int64(len(data))) // error is fine, panic is not
	})
}

// FuzzRoundTrip (§V15): a set of named tensors survives Save→Load bit-exactly.
func FuzzRoundTrip(f *testing.F) {
	f.Add(2, 3, true, []byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, d0, d1 int, isF64 bool, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 16 || d1 > 16 {
			return
		}
		dtype := tensor.F32
		if isF64 {
			dtype = tensor.F64
		}
		a := tensor.New(dtype, tensor.Shape{d0, d1})
		b := tensor.New(dtype, tensor.Shape{d1})
		for i := range a.Numel() {
			idx := tensor.Unravel(i, a.Shape())
			if len(raw) > 0 {
				a.SetF64(float64(int8(raw[i%len(raw)]))*0.25, idx...)
			}
		}
		in := map[string]*tensor.Tensor{"a": a, "b": b}
		var buf bytes.Buffer
		if err := Save(&buf, in); err != nil {
			t.Fatalf("save: %v", err)
		}
		got, err := Load(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if len(got) != 2 {
			t.Fatalf("got %d arrays", len(got))
		}
		for name, want := range in {
			g := got[name]
			if g == nil || !g.Shape().Equal(want.Shape()) || g.Dtype() != want.Dtype() {
				t.Fatalf("%q drift", name)
			}
			for i := range want.Numel() {
				idx := tensor.Unravel(i, want.Shape())
				if g.AtF64(idx...) != want.AtF64(idx...) {
					t.Fatalf("%q elem %d mismatch", name, i)
				}
			}
		}
	})
}
