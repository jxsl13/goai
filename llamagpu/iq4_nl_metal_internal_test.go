//go:build darwin && cgo

package llamagpu

import (
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// The full-device decoder opts into native IQ4_NL residency even though generic host-bound
// QuantLinear deliberately remains on the faster M2 ARM64 path.
func TestMetalUploadIQ4NLUsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 64
	wq := make([]byte, n*(k/32)*18)
	for block := range len(wq) / 18 {
		binary.LittleEndian.PutUint16(wq[block*18:], 0x3000)
		for i := 2; i < 18; i++ {
			wq[block*18+i] = byte(block*17 + i*29)
		}
	}
	raw, err := metalUploadQWeight(wq, uint32(gguf.IQ4_NL), n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, ok := raw.(*metal.ResidentQWeight)
	if !ok {
		t.Fatalf("IQ4_NL upload type %T, want *metal.ResidentQWeight", raw)
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
}
