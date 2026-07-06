package gguf

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

type expected map[string]struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
}

func loadExpected(t *testing.T) expected {
	t.Helper()
	raw, err := os.ReadFile("testdata/expected.json")
	if err != nil {
		t.Fatalf("read expected (run `make golden`): %v", err)
	}
	var e expected
	if err := json.Unmarshal(raw, &e); err != nil {
		t.Fatal(err)
	}
	return e
}

// §V1: read the official-lib-written GGUF; every tensor (raw f32, f16, Q8_0,
// Q4_0) dequantizes to exactly the reference dequantization.
func TestReadSampleParity(t *testing.T) {
	f, err := ReadFile("testdata/sample.gguf")
	if err != nil {
		t.Fatalf("read sample (run `make golden`): %v", err)
	}
	if f.Version < 2 {
		t.Fatalf("version %d", f.Version)
	}
	// metadata
	if v, ok := f.Metadata["goai.test_u32"].(uint32); !ok || v != 7 {
		t.Errorf("metadata u32: %v", f.Metadata["goai.test_u32"])
	}
	if v, ok := f.Metadata["goai.note"].(string); !ok || v != "golden" {
		t.Errorf("metadata string: %v", f.Metadata["goai.note"])
	}
	if v, ok := f.Metadata["general.architecture"].(string); !ok || v != "goai-test" {
		t.Errorf("architecture: %v", f.Metadata["general.architecture"])
	}

	exp := loadExpected(t)
	for name, want := range exp {
		tt := f.Tensors[name]
		if tt == nil {
			t.Fatalf("tensor %q missing", name)
		}
		if tt.Dtype() != tensor.F32 {
			t.Fatalf("%s: dtype %v, want f32 (dequantized)", name, tt.Dtype())
		}
		if !tt.Shape().Equal(tensor.Shape(want.Shape)) {
			t.Fatalf("%s: shape %v, want %v (dims must be reversed from file order)", name, tt.Shape(), want.Shape)
		}
		for i, w := range want.Values {
			idx := tensor.Unravel(i, tt.Shape())
			got := tt.AtF64(idx...)
			// dequantization is deterministic f32 math → match to f32 exactness
			if math.Abs(got-w) > 1e-7*math.Max(1, math.Abs(w)) {
				t.Fatalf("%s[%d]: got %.9g want %.9g", name, i, got, w)
			}
		}
	}
}

func TestF16Conversion(t *testing.T) {
	cases := map[uint16]float64{
		0x3C00: 1.0,
		0xBC00: -1.0,
		0x0000: 0.0,
		0x7BFF: 65504.0, // max finite
		0x3555: 0.333251953125,
		0x0001: math.Pow(2, -24), // smallest subnormal
	}
	for bits, want := range cases {
		if got := float64(f16ToF32(bits)); math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Errorf("f16(%#04x) = %v, want %v", bits, got, want)
		}
	}
	if !math.IsInf(float64(f16ToF32(0x7C00)), 1) {
		t.Error("f16 +inf")
	}
	if !math.IsNaN(float64(f16ToF32(0x7E00))) {
		t.Error("f16 NaN")
	}
}

func TestBadInputs(t *testing.T) {
	if _, err := Read(bytes.NewReader([]byte("nope"))); err == nil {
		t.Error("bad magic must error")
	}
	// magic ok but truncated
	if _, err := Read(bytes.NewReader([]byte{0x47, 0x47, 0x55, 0x46, 3, 0})); err == nil {
		t.Error("truncated header must error")
	}
}
