//go:build darwin && cgo

package llamagpu

import (
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func TestMetalUploadTQUsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 512
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockBytes int
	}{{"TQ1_0", gguf.TQ1_0, 54}, {"TQ2_0", gguf.TQ2_0, 66}} {
		t.Run(tc.name, func(t *testing.T) {
			weight := make([]byte, n*(k/256)*tc.blockBytes)
			for block := range len(weight) / tc.blockBytes {
				base := block * tc.blockBytes
				for i := range tc.blockBytes - 2 {
					weight[base+i] = byte(block*17 + i*29)
				}
				//perfscan:ignore PS4001 TQ scale fields are strided at heterogeneous block tails
				binary.LittleEndian.PutUint16(weight[base+tc.blockBytes-2:], 0x3000)
			}
			raw, err := metalUploadQWeight(weight, uint32(tc.qt), n, k)
			if err != nil {
				t.Fatal(err)
			}
			resident, ok := raw.(*metal.ResidentQWeight)
			if !ok {
				t.Fatalf("upload type %T, want *metal.ResidentQWeight", raw)
			}
			if err := resident.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}
