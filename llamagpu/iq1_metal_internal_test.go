//go:build darwin && cgo

package llamagpu

import (
	"testing"

	"github.com/jxsl13/goai/backend/metal"
	"github.com/jxsl13/goai/format/gguf"
)

func TestMetalUploadIQ1UsesRecorderOnlyResidentPath(t *testing.T) {
	if !metal.Available() {
		t.Skip("metal device unavailable")
	}
	const n, k = 7, 512
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockBytes int
	}{{"IQ1_S", gguf.IQ1_S, 50}, {"IQ1_M", gguf.IQ1_M, 56}} {
		t.Run(tc.name, func(t *testing.T) {
			weight := make([]byte, n*(k/256)*tc.blockBytes)
			for block := range len(weight) / tc.blockBytes {
				base := block * tc.blockBytes
				for i := range tc.blockBytes {
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
