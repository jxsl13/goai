package npy

import (
	"bytes"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// FuzzLoad (§V15): no hostile byte stream may panic; malformed input must error.
func FuzzLoad(f *testing.F) {
	for _, p := range []string{"testdata/ref_f64.npy", "testdata/ref_f32.npy"} {
		if raw, err := os.ReadFile(p); err == nil {
			f.Add(raw)
		}
	}
	var buf bytes.Buffer
	_ = Save(&buf, tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2}))
	f.Add(buf.Bytes())
	f.Add([]byte{})
	f.Add([]byte("\x93NUMPY\x01\x00"))
	f.Add(append([]byte("\x93NUMPY\x01\x00\xff\xff"), []byte("garbage")...))

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Load panicked on %d bytes: %v", len(data), r)
			}
		}()
		_, _ = Load(bytes.NewReader(data)) // error is fine, panic is not
	})
}

// FuzzRoundTrip (§V15): any tensor GoAI builds survives Save→Load bit-exactly
// (NaN/Inf included), across both dtypes and small shapes.
func FuzzRoundTrip(f *testing.F) {
	f.Add(2, 3, true, []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Fuzz(func(t *testing.T, d0, d1 int, isF64 bool, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 24 || d1 > 24 {
			return
		}
		dtype := tensor.F32
		if isF64 {
			dtype = tensor.F64
		}
		in := tensor.New(dtype, tensor.Shape{d0, d1})
		n := in.Numel()
		for i := range n { // fill from the fuzz bytes (repeating) so values vary, incl. NaN patterns
			idx := tensor.Unravel(i, in.Shape())
			var v float64
			if len(raw) > 0 {
				v = float64(int8(raw[i%len(raw)])) * 0.5
			}
			in.SetF64(v, idx...)
		}
		var buf bytes.Buffer
		if err := Save(&buf, in); err != nil {
			t.Fatalf("save: %v", err)
		}
		out, err := Load(&buf)
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if !out.Shape().Equal(in.Shape()) || out.Dtype() != in.Dtype() {
			t.Fatalf("shape/dtype drift: %v/%v vs %v/%v", out.Shape(), out.Dtype(), in.Shape(), in.Dtype())
		}
		for i := range n {
			idx := tensor.Unravel(i, in.Shape())
			if in.AtF64(idx...) != out.AtF64(idx...) {
				t.Fatalf("elem %d: %g != %g", i, in.AtF64(idx...), out.AtF64(idx...))
			}
		}
	})
}
