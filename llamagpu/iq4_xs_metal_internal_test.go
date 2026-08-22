//go:build darwin && cgo

package llamagpu

import (
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

// The full-device decoder opts into native IQ4_XS residency even though generic host-bound
// QuantLinear remains on the faster M2 ARM64 path unless its separate gate proves otherwise.
func TestMetalUploadIQ4XSUsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 512
	wq := make([]byte, n*(k/256)*136)
	for block := range len(wq) / 136 {
		base := block * 136
		//perfscan:ignore PS4001 strided f16 fields in heterogeneous IQ4_XS blocks cannot use a same-layout bulk copy
		binary.LittleEndian.PutUint16(wq[base:], 0x2000)
		binary.LittleEndian.PutUint16(wq[base+2:], 0xaaaa)
		for i := 4; i < 8; i++ {
			wq[base+i] = 0x11
		}
		for i := 8; i < 136; i++ {
			wq[base+i] = byte(block*17 + i*29)
		}
	}
	raw, err := metalUploadQWeight(wq, uint32(gguf.IQ4_XS), n, k)
	if err != nil {
		t.Fatal(err)
	}
	rw, ok := raw.(*metal.ResidentQWeight)
	if !ok {
		t.Fatalf("IQ4_XS upload type %T, want *metal.ResidentQWeight", raw)
	}
	if err := rw.Close(); err != nil {
		t.Fatal(err)
	}
}
