package gguf

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/jxsl13/goai/tensor"
)

// §T555 §V16 tier-1: BOTH directions match gguf-py exactly — dequant on random
// blocks (f32-exact) and encode on random floats (byte-exact).
func TestMXFP4MatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/mxfp4_golden.json")
	if err != nil {
		t.Skip("golden missing:", err)
	}
	var golden struct {
		Data []byte    `json:"data"`
		Want []float32 `json:"want"`
		Vals []float32 `json:"vals"`
		Enc  []byte    `json:"enc"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(golden.Data, MXFP4, len(golden.Want))
	if err != nil {
		t.Fatal(err)
	}
	g := got.Storage().F32()
	for i := range golden.Want {
		if g[i] != golden.Want[i] {
			t.Fatalf("dequant [%d]: go %g vs gguf-py %g", i, g[i], golden.Want[i])
		}
	}
	vt := tensor.FromFloat32(tensor.Shape{len(golden.Vals)}, golden.Vals)
	enc, err := Quantize(vt, MXFP4)
	if err != nil {
		t.Fatal(err)
	}
	if len(enc) != len(golden.Enc) {
		t.Fatalf("encode length %d vs %d", len(enc), len(golden.Enc))
	}
	for i := range enc {
		if enc[i] != golden.Enc[i] {
			t.Fatalf("encode [%d]: go %#x vs gguf-py %#x", i, enc[i], golden.Enc[i])
		}
	}
}

// Round trip: encode→decode error bounded by half the E2M1 step at the block
// scale (≤ d·1 for the coarse top of the table; generous bound d·2 guards it).
func TestMXFP4RoundTripBound(t *testing.T) {
	vals := make([]float32, 64)
	for i := range vals {
		vals[i] = float32(i%13) * 0.37 * float32(1-2*(i%2))
	}
	vt := tensor.FromFloat32(tensor.Shape{64}, vals)
	enc, err := Quantize(vt, MXFP4)
	if err != nil {
		t.Fatal(err)
	}
	dec, err := Dequantize(enc, MXFP4, 64)
	if err != nil {
		t.Fatal(err)
	}
	d := dec.Storage().F32()
	for b := range 2 {
		scale := e8m0ToF32Half(enc[b*mxfp4BlockSize])
		for i := range 32 {
			idx := b*32 + i
			if diff := d[idx] - vals[idx]; diff > 2*scale || diff < -2*scale {
				t.Fatalf("[%d]: error %g exceeds 2·scale %g", idx, diff, 2*scale)
			}
		}
	}
}

// Hostile sizes error; junk decodes without panicking.
func TestMXFP4Hostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 16), MXFP4, 32); err == nil {
		t.Fatal("short data must error")
	}
	junk := make([]byte, 17)
	for i := range junk {
		junk[i] = byte(i*97 + 255)
	}
	if _, err := Dequantize(junk, MXFP4, 32); err != nil {
		t.Fatal(err)
	}
}

// FuzzMXFP4RoundTrip: quantize→dequantize never panics and stays finite for
// finite inputs.
func FuzzMXFP4RoundTrip(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4})
	f.Fuzz(func(t *testing.T, data []byte) {
		vals := make([]float32, 32)
		for i := range vals {
			if i < len(data) {
				vals[i] = float32(int8(data[i])) * 0.25
			}
		}
		vt := tensor.FromFloat32(tensor.Shape{32}, vals)
		enc, err := Quantize(vt, MXFP4)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Dequantize(enc, MXFP4, 32); err != nil {
			t.Fatal(err)
		}
	})
}
