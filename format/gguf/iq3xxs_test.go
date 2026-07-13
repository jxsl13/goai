package gguf

import (
	"encoding/json"
	"os"
	"testing"
)

// §T554 part 3, §V16 tier-1: Go IQ3_XXS dequant == gguf-py, f32-exact (seed 44).
func TestIQ3XXSDequantMatchesGGUFPy(t *testing.T) {
	raw, err := os.ReadFile("testdata/iq3xxs_golden.json")
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
	got, err := Dequantize(golden.Data, IQ3_XXS, len(golden.Want))
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

// Hostile sizes error; junk decodes without panicking.
func TestIQ3XXSHostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 97), IQ3_XXS, 256); err == nil {
		t.Fatal("short data must error")
	}
	junk := make([]byte, 98)
	for i := range junk {
		junk[i] = byte(i*29 + 3)
	}
	if _, err := Dequantize(junk, IQ3_XXS, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ3XXS: any block-size byte soup decodes without panicking.
func FuzzDequantIQ3XXS(f *testing.F) {
	f.Add(make([]byte, 98))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 98 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:98], IQ3_XXS, 256)
	})
}
