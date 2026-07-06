package gguf

import (
	"bytes"
	"os"
	"testing"
)

// FuzzRead: no byte stream may panic. GGUF has no writer yet (read-only format
// for us), so this is a pure robustness fuzz seeded with the real sample and
// header fragments.
func FuzzRead(f *testing.F) {
	if raw, err := os.ReadFile("testdata/sample.gguf"); err == nil {
		f.Add(raw)
	}
	f.Add([]byte("GGUF\x03\x00\x00\x00"))
	f.Add([]byte{0x47, 0x47, 0x55, 0x46})
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
