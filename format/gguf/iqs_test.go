package gguf

import (
	"encoding/json"
	"os"
	"testing"
)

func iqGolden(t *testing.T, file string, qt QuantType) {
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

// §T554 part 5, §V16 tier-1: IQ3_S and IQ2_S match gguf-py f32-exactly.
func TestIQ3SDequantMatchesGGUFPy(t *testing.T) { iqGolden(t, "testdata/iq3s_golden.json", IQ3_S) }
func TestIQ2SDequantMatchesGGUFPy(t *testing.T) { iqGolden(t, "testdata/iq2s_golden.json", IQ2_S) }

// Hostile sizes error; junk decodes without panicking.
func TestIQSHostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 109), IQ3_S, 256); err == nil {
		t.Fatal("short IQ3_S must error")
	}
	if _, err := Dequantize(make([]byte, 81), IQ2_S, 256); err == nil {
		t.Fatal("short IQ2_S must error")
	}
	junk := make([]byte, 110)
	for i := range junk {
		junk[i] = byte(i*61 + 17)
	}
	if _, err := Dequantize(junk, IQ3_S, 256); err != nil {
		t.Fatal(err)
	}
	if _, err := Dequantize(junk[:82], IQ2_S, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ3S / FuzzDequantIQ2S: byte soup never panics.
func FuzzDequantIQ3S(f *testing.F) {
	f.Add(make([]byte, 110))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 110 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:110], IQ3_S, 256)
	})
}

func FuzzDequantIQ2S(f *testing.F) {
	f.Add(make([]byte, 82))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 82 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:82], IQ2_S, 256)
	})
}
