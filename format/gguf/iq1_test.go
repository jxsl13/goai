package gguf

import "testing"

// §T554 part 6, §V16 tier-1: the ternary pair matches gguf-py f32-exactly.
func TestIQ1SDequantMatchesGGUFPy(t *testing.T) { iqGolden(t, "testdata/iq1s_golden.json", IQ1_S) }
func TestIQ1MDequantMatchesGGUFPy(t *testing.T) { iqGolden(t, "testdata/iq1m_golden.json", IQ1_M) }

// Hostile sizes error; junk decodes without panicking.
func TestIQ1Hostile(t *testing.T) {
	if _, err := Dequantize(make([]byte, 49), IQ1_S, 256); err == nil {
		t.Fatal("short IQ1_S must error")
	}
	if _, err := Dequantize(make([]byte, 55), IQ1_M, 256); err == nil {
		t.Fatal("short IQ1_M must error")
	}
	junk := make([]byte, 56)
	for i := range junk {
		junk[i] = byte(i*71 + 5)
	}
	if _, err := Dequantize(junk[:50], IQ1_S, 256); err != nil {
		t.Fatal(err)
	}
	if _, err := Dequantize(junk, IQ1_M, 256); err != nil {
		t.Fatal(err)
	}
}

// FuzzDequantIQ1M: byte soup never panics (the split-scale reassembly is the
// most index-happy decoder of the family).
func FuzzDequantIQ1M(f *testing.F) {
	f.Add(make([]byte, 56))
	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) < 56 {
			return
		}
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("panicked: %v", r)
			}
		}()
		_, _ = Dequantize(data[:56], IQ1_M, 256)
	})
}
