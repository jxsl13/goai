package nlp

import (
	"bytes"
	"encoding/binary"
	"testing"

	"github.com/jxsl13/goai/format/gguf"
)

func TestPhi3TQRowSlicingPreservesCompressedBytes(t *testing.T) {
	for _, tc := range []struct {
		name       string
		qt         gguf.QuantType
		blockBytes int
	}{{"TQ1_0", gguf.TQ1_0, 54}, {"TQ2_0", gguf.TQ2_0, 66}} {
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
			for block := range len(raw) / tc.blockBytes {
				base := block * tc.blockBytes
				for i := range tc.blockBytes - 2 {
					raw[base+i] = byte(block*29 + i*17)
				}
				//perfscan:ignore PS4001 TQ scale fields are strided at heterogeneous block tails
				binary.LittleEndian.PutUint16(raw[base+tc.blockBytes-2:], 0x3000)
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
				t.Fatal("Phi-3 row slice changed compressed TQ bytes")
			}
		})
	}
}
