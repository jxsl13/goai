package nlp

import (
	"bytes"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

func TestPhi3IQ2RowSlicingPreservesCompressedBytes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockBytes int
	}{{"IQ2_XXS", gguf.IQ2_XXS, 66}, {"IQ2_XS", gguf.IQ2_XS, 74}, {"IQ2_S", gguf.IQ2_S, 82}} {
		t.Run(tc.name, func(t *testing.T) {
			blockElems, err := ggBlockElems(uint32(tc.qt))
			if err != nil {
				t.Fatal(err)
			}
			if blockElems != 256 {
				t.Fatalf("block elements = %d, want 256", blockElems)
			}
			const rows, cols = 6, 512
			rowBytes := (cols / blockElems) * tc.blockBytes
			raw := make([]byte, rows*rowBytes)
			for i := range raw {
				raw[i] = byte(i*29 + 17)
			}
			got, err := quantSliceRows(gguf.QuantTensor{
				Shape:  []int{rows, cols},
				GGType: uint32(tc.qt),
				Data:   raw,
			}, 2, 5)
			if err != nil {
				t.Fatal(err)
			}
			if got.GGType != uint32(tc.qt) || len(got.Shape) != 2 || got.Shape[0] != 3 || got.Shape[1] != cols {
				t.Fatalf("slice metadata = type %d shape %v", got.GGType, got.Shape)
			}
			if !bytes.Equal(got.Data, raw[2*rowBytes:5*rowBytes]) {
				t.Fatal("Phi-3 row slice changed compressed IQ2 bytes")
			}
		})
	}
}
