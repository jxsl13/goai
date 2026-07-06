package npy

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// FuzzRead: no byte stream may panic; malformed input must error.
func FuzzRead(f *testing.F) {
	for _, p := range []string{"testdata/mat_f64.npy", "testdata/mat_f32.npy", "testdata/vec_f64.npy"} {
		if raw, err := os.ReadFile(p); err == nil {
			f.Add(raw)
		}
	}
	f.Add([]byte("\x93NUMPY\x01\x00"))
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Read panicked on %d bytes: %v", len(data), r)
			}
		}()
		_, _ = Read(bytes.NewReader(data))
	})
}

// FuzzRoundTrip: Write→Read is bit-exact for arbitrary values/shapes.
func FuzzRoundTrip(f *testing.F) {
	f.Add(3, 2, []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Fuzz(func(t *testing.T, d0, d1 int, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 32 || d1 > 32 {
			return
		}
		n := d0 * d1
		if len(raw) < n*8 {
			return
		}
		vals := make([]float64, n)
		for i := range vals {
			vals[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
		}
		in := tensor.FromFloat64(tensor.Shape{d0, d1}, vals)
		var buf bytes.Buffer
		if err := Write(&buf, in); err != nil {
			t.Fatal(err)
		}
		out, err := Read(&buf)
		if err != nil {
			t.Fatalf("read after write: %v", err)
		}
		if !out.Shape().Equal(in.Shape()) {
			t.Fatalf("shape drift: %v vs %v", out.Shape(), in.Shape())
		}
		for i := range in.Numel() {
			idx := tensor.Unravel(i, in.Shape())
			if math.Float64bits(out.AtF64(idx...)) != math.Float64bits(in.AtF64(idx...)) {
				t.Fatalf("value drift at %d", i)
			}
		}
	})
}
