package safetensors

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// FuzzLoad: no hostile byte stream may panic; malformed input must error.
func FuzzLoad(f *testing.F) {
	if raw, err := os.ReadFile("testdata/ref.safetensors"); err == nil {
		f.Add(raw)
	}
	var buf bytes.Buffer
	_ = Save(&buf, map[string]*tensor.Tensor{"a": tensor.FromFloat64(tensor.Shape{2}, []float64{1, 2})}, nil)
	f.Add(buf.Bytes())
	f.Add([]byte{})
	f.Add([]byte{0, 0, 0, 0, 0, 0, 0, 0})

	f.Fuzz(func(t *testing.T, data []byte) {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("Load panicked on %d bytes: %v", len(data), r)
			}
		}()
		_, _, _ = Load(bytes.NewReader(data)) // error is fine, panic is not
	})
}

// FuzzRoundTrip: any tensor we build must survive Save→Load bit-exactly (NaN
// included), across both dtypes and small shapes.
func FuzzRoundTrip(f *testing.F) {
	f.Add(2, 3, true, []byte{1, 2, 3, 4, 5, 6, 7, 8})
	f.Fuzz(func(t *testing.T, d0, d1 int, isF64 bool, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 32 || d1 > 32 {
			return
		}
		n := d0 * d1
		elemBytes := 4
		if isF64 {
			elemBytes = 8
		}
		if len(raw) < n*elemBytes {
			return
		}
		var in *tensor.Tensor
		if isF64 {
			vals := make([]float64, n)
			for i := range vals {
				vals[i] = math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:]))
			}
			in = tensor.FromFloat64(tensor.Shape{d0, d1}, vals)
		} else {
			vals := make([]float32, n)
			for i := range vals {
				vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
			}
			in = tensor.FromFloat32(tensor.Shape{d0, d1}, vals)
		}
		var buf bytes.Buffer
		if err := Save(&buf, map[string]*tensor.Tensor{"t": in}, nil); err != nil {
			t.Fatal(err)
		}
		out, _, err := Load(&buf)
		if err != nil {
			t.Fatalf("load after save: %v", err)
		}
		got := out["t"]
		if got.Dtype() != in.Dtype() || !got.Shape().Equal(in.Shape()) {
			t.Fatalf("meta drift: %v%v vs %v%v", got.Dtype(), got.Shape(), in.Dtype(), in.Shape())
		}
		for i := range in.Numel() {
			idx := tensor.Unravel(i, in.Shape())
			// bit-exact compare handles NaN/±0/±Inf
			gb, ib := math.Float64bits(got.AtF64(idx...)), math.Float64bits(in.AtF64(idx...))
			if gb != ib {
				t.Fatalf("value drift at %d: %x vs %x", i, gb, ib)
			}
		}
	})
}
