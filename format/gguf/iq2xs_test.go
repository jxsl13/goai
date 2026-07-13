package gguf

import (
	"encoding/json"
	"os"
	"testing"
)

// §T554 part 2, §V16 tier-1: Go IQ2_XS dequant == gguf-py on random blocks,
// f32-exact (golden: seed 43, 4 blocks).
func TestIQ2XSDequantMatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/iq2xs_golden.json")
	if err != nil {
		t.Skip("golden missing:", err)
	}
	var golden struct {
		Data []byte    `json:"data"`
		Want []float32 `json:"want"`
	}
	if err := json.Unmarshal(raw, &golden); err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(golden.Data, IQ2_XS, len(golden.Want))
	if err != nil {
		t.Fatal(err)
	}
	g := got.Storage().F32()
	for i := range golden.Want {
		if g[i] != golden.Want[i] {
			t.Fatalf("[%d]: go %g vs gguf-py %g", i, g[i], golden.Want[i])
		}
	}
}

// Hostile sizes error; junk bytes decode without panicking.
func TestIQ2XSHostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 73), IQ2_XS, 256); err == nil {
		t.Fatal("short data must error")
	}
	junk := make([]byte, 74)
	for i := range junk {
		junk[i] = byte(i*53 + 7)
	}
	if _, err := Dequantize(junk, IQ2_XS, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ2XS: any block-size byte soup decodes without panicking.
func FuzzDequantIQ2XS(f *testing.F) {
	f.Add(make([]byte, 74))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 74 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:74], IQ2_XS, 256)
	})
}
