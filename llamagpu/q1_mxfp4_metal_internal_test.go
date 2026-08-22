//go:build darwin && cgo

package llamagpu

import (
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func TestMetalUploadQ1MXFP4UsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 512
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockElems int
		blockBytes int
	}{{"Q1_0", gguf.Q1_0, 128, 18}, {"MXFP4", gguf.MXFP4, 32, 17}} {
		t.Run(tc.name, func(t *testing.T) {
			weight := make([]byte, n*(k/tc.blockElems)*tc.blockBytes)
			for block := range len(weight) / tc.blockBytes {
				base := block * tc.blockBytes
				if tc.qt == gguf.Q1_0 {
					//perfscan:ignore PS4001 intentionally strided f16 scales in a resident-upload fixture
					binary.LittleEndian.PutUint16(weight[base:], 0x3000)
					for i := 2; i < tc.blockBytes; i++ {
						weight[base+i] = byte(block*17 + i*29)
					}
				} else {
					weight[base] = 123
					for i := 1; i < tc.blockBytes; i++ {
						weight[base+i] = byte(block*17 + i*29)
					}
				}
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
