//go:build darwin && cgo

package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func TestMetalUploadIQ2UsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 512
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockBytes int
	}{{"IQ2_XXS", gguf.IQ2_XXS, 66}, {"IQ2_XS", gguf.IQ2_XS, 74}, {"IQ2_S", gguf.IQ2_S, 82}} {
		t.Run(tc.name, func(t *testing.T) {
			weight := make([]byte, n*(k/256)*tc.blockBytes)
			for block := range len(weight) / tc.blockBytes {
				base := block * tc.blockBytes
				weight[base], weight[base+1] = 0x00, 0x20
				for i := 2; i < tc.blockBytes; i++ {
					weight[base+i] = byte(block*17 + i*29)
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
