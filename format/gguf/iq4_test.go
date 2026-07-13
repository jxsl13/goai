package gguf

import (
	"encoding/json"
	"os"
	"testing"
)

func iq4Golden(t *testing.T, file string, qt QuantType) {
	t.Helper()
	raw, err := os.ReadFile(file)
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
	got, err := Dequantize(golden.Data, qt, len(golden.Want))
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

// §T554 part 4, §V16 tier-1: both IQ4 variants match gguf-py f32-exactly.
func TestIQ4NLDequantMatchesGGUFPy(t *testing.T) {
	iq4Golden(t, "testdata/iq4nl_golden.json", IQ4_NL)
}
func TestIQ4XSDequantMatchesGGUFPy(t *testing.T) {
	iq4Golden(t, "testdata/iq4xs_golden.json", IQ4_XS)
}

// Hostile sizes error; junk decodes without panicking.
func TestIQ4Hostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 17), IQ4_NL, 32); err == nil {
		t.Fatal("short IQ4_NL must error")
	}
	if _, err := Dequantize(make([]byte, 135), IQ4_XS, 256); err == nil {
		t.Fatal("short IQ4_XS must error")
	}
	junk := make([]byte, 136)
	for i := range junk {
		junk[i] = byte(i*41 + 13)
	}
	if _, err := Dequantize(junk[:18], IQ4_NL, 32); err != nil {
		t.Fatal(err)
	}
	if _, err := Dequantize(junk, IQ4_XS, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ4XS: any block-size byte soup decodes without panicking.
func FuzzDequantIQ4XS(f *testing.F) {
	f.Add(make([]byte, 136))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 136 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:136], IQ4_XS, 256)
	})
}
