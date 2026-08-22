package nlp

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

func TestPhi3Q1MXFP4RowSlicingPreservesCompressedBytes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockElems int
		blockBytes int
	}{{"Q1_0", gguf.Q1_0, 128, 18}, {"MXFP4", gguf.MXFP4, 32, 17}} {
		t.Run(tc.name, func(t *testing.T) {
			blockElems, err := ggBlockElems(uint32(tc.qt))
			if err != nil {
				t.Fatal(err)
			}
			if blockElems != tc.blockElems {
				t.Fatalf("block elements = %d, want %d", blockElems, tc.blockElems)
			}
			const rows, cols = 6, 512
			rowBytes := (cols / blockElems) * tc.blockBytes
			raw := make([]byte, rows*rowBytes)
			for block := range len(raw) / tc.blockBytes {
				base := block * tc.blockBytes
				if tc.qt == gguf.Q1_0 {
					//perfscan:ignore PS4001 intentionally strided f16 scales in a row-slice fixture
					binary.LittleEndian.PutUint16(raw[base:], 0x3000)
					for i := 2; i < tc.blockBytes; i++ {
						raw[base+i] = byte(block*29 + i*17)
					}
				} else {
					raw[base] = 123
					for i := 1; i < tc.blockBytes; i++ {
						raw[base+i] = byte(block*29 + i*17)
					}
				}
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
				t.Fatal("Phi-3 row slice changed compressed quant bytes")
			}
		})
	}
}
