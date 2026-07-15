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

// §V15 regression: the shape-product cap must be overflow-safe. The old
// post-multiply check wrapped uint64 — shape [2^40, 2^40] gives 2^80 ≡ 0 —
// so a hostile header passed the cap and Load returned a nil-error tensor
// claiming 2^80 elements over empty storage.
func TestHostileNumelOverflow(t *testing.T) {
	mk := func(shape string) []byte {
		hdr := `{"t":{"dtype":"F32","shape":` + shape + `,"data_offsets":[0,0]}}`
		for len(hdr)%8 != 0 {
			hdr += " "
		}
		var b bytes.Buffer
		binary.Write(&b, binary.LittleEndian, uint64(len(hdr)))
		b.WriteString(hdr)
		return b.Bytes()
	}
	cases := map[string][]byte{
		"wrap to zero [2^40,2^40]":     mk("[1099511627776,1099511627776]"),
		"wrap [2^30,2^30,2^30,2^30]":   mk("[1073741824,1073741824,1073741824,1073741824]"),
		"over cap plain [2^40+1 x 2]":  mk("[1099511627777,2]"),
		"over cap product [2^39,2^39]": mk("[549755813888,549755813888]"),
	}
	for name, data := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panicked: %v", name, r)
				}
			}()
			if _, _, err := Load(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: must error, cap bypassed", name)
			}
		}()
	}
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

// f16Tensor builds an F16/BF16 tensor from raw uint16 bits (verbatim, no conversion).
func f16Tensor(dt tensor.Dtype, shape tensor.Shape, bits []uint16) *tensor.Tensor {
	t := tensor.New(dt, shape)
	copy(t.Storage().U16(), bits)
	return t
}

// §V15 (§R22 spec): F16 and BF16 tensors round-trip through safetensors bit-exactly — the raw
// uint16 bits are stored verbatim (no widening), and the HuggingFace dtype strings are used.
func TestRoundTripF16BF16(t *testing.T) {
	// includes +0, a normal, a subnormal, NaN and ±Inf bit patterns.
	bits := []uint16{0x0000, 0x3C00, 0x0001, 0x7E00, 0x7C00, 0xFC00}
	for _, dt := range []tensor.Dtype{tensor.F16, tensor.BF16} {
		in := f16Tensor(dt, tensor.Shape{2, 3}, bits)
		var buf bytes.Buffer
		if err := Save(&buf, map[string]*tensor.Tensor{"w": in}, nil); err != nil {
			t.Fatalf("%v: save: %v", dt, err)
		}
		out, _, err := Load(&buf)
		if err != nil {
			t.Fatalf("%v: load: %v", dt, err)
		}
		got := out["w"]
		if got.Dtype() != dt || !got.Shape().Equal(in.Shape()) {
			t.Fatalf("%v: meta drift %v%v", dt, got.Dtype(), got.Shape())
		}
		gb, ib := got.Storage().U16(), in.Storage().U16()
		for i := range ib {
			if gb[i] != ib[i] {
				t.Errorf("%v: bit drift at %d: %04x vs %04x", dt, i, gb[i], ib[i])
			}
		}
	}
}

// §V15: a file mixing F16, BF16 and F32 tensors round-trips, each keeping its own dtype.
func TestRoundTripMixedDtypes(t *testing.T) {
	in := map[string]*tensor.Tensor{
		"half":  f16Tensor(tensor.F16, tensor.Shape{2}, []uint16{0x3C00, 0x4000}),  // 1.0, 2.0
		"bhalf": f16Tensor(tensor.BF16, tensor.Shape{2}, []uint16{0x3F80, 0x4000}), // 1.0, 2.0
		"full":  tensor.FromFloat64(tensor.Shape{2}, []float64{3, 4}),
	}
	var buf bytes.Buffer
	if err := Save(&buf, in, nil); err != nil {
		t.Fatal(err)
	}
	out, _, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range in {
		got := out[name]
		if got == nil || got.Dtype() != want.Dtype() {
			t.Fatalf("%q dtype drift: %v", name, got)
		}
	}
	if got := out["bhalf"].AtF64(1); math.Abs(got-2) > 1e-6 {
		t.Errorf("bf16 value %g, want 2", got)
	}
}

// §V15 / §R22: the emitted header carries the exact HuggingFace dtype string and the data is the
// raw little-endian uint16 bits (definitional-format check, not a round-trip).
func TestF16HeaderAndBytes(t *testing.T) {
	in := f16Tensor(tensor.BF16, tensor.Shape{1}, []uint16{0x3F80}) // bf16 1.0
	var buf bytes.Buffer
	if err := Save(&buf, map[string]*tensor.Tensor{"x": in}, nil); err != nil {
		t.Fatal(err)
	}
	raw := buf.Bytes()
	hlen := binary.LittleEndian.Uint64(raw[:8])
	header := string(raw[8 : 8+hlen])
	if !bytes.Contains([]byte(header), []byte(`"dtype":"BF16"`)) {
		t.Errorf("header %q missing BF16 dtype", header)
	}
	payload := raw[8+hlen:]
	if len(payload) != 2 || binary.LittleEndian.Uint16(payload) != 0x3F80 {
		t.Errorf("payload %x, want 2 bytes 0x3F80 LE", payload)
	}
}

// §V15 fuzz: arbitrary 16-bit tensors survive Save→Load bit-exactly (the stored bits are verbatim).
func FuzzRoundTrip16(f *testing.F) {
	f.Add(2, 3, true, []byte{0, 0x3C, 0, 0x40, 1, 0, 0, 0x7E, 0, 0x7C, 0, 0xFC})
	f.Fuzz(func(t *testing.T, d0, d1 int, isBF16 bool, raw []byte) {
		if d0 < 0 || d1 < 0 || d0 > 32 || d1 > 32 {
			return
		}
		n := d0 * d1
		if len(raw) < n*2 {
			return
		}
		dt := tensor.F16
		if isBF16 {
			dt = tensor.BF16
		}
		bits := make([]uint16, n)
		for i := range bits {
			bits[i] = binary.LittleEndian.Uint16(raw[i*2:])
		}
		in := f16Tensor(dt, tensor.Shape{d0, d1}, bits)
		var buf bytes.Buffer
		if err := Save(&buf, map[string]*tensor.Tensor{"t": in}, nil); err != nil {
			t.Fatal(err)
		}
		out, _, err := Load(&buf)
		if err != nil {
			t.Fatalf("load after save: %v", err)
		}
		got := out["t"]
		if got.Dtype() != dt || !got.Shape().Equal(in.Shape()) {
			t.Fatalf("meta drift: %v%v", got.Dtype(), got.Shape())
		}
		gb, ib := got.Storage().U16(), in.Storage().U16()
		for i := range ib {
			if gb[i] != ib[i] {
				t.Fatalf("bit drift at %d: %04x vs %04x", i, gb[i], ib[i])
			}
		}
	})
}
