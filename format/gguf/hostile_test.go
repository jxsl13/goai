package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// §V15/B28 regression: hostile inputs must return errors, never panic. The
// huge-dim case previously panicked (2^63 → negative int; n·4 wrapped to 0 and
// slipped past the size check into tensor.New).
func TestHostileInputsErrorNotPanic(t *testing.T) {
	mk := func(dim uint64) []byte {
		var b bytes.Buffer
		b.Write([]byte{0x47, 0x47, 0x55, 0x46})
		binary.Write(&b, binary.LittleEndian, uint32(3))
		binary.Write(&b, binary.LittleEndian, uint64(1)) // tensors
		binary.Write(&b, binary.LittleEndian, uint64(0)) // kvs
		binary.Write(&b, binary.LittleEndian, uint64(1)) // name len
		b.WriteString("t")
		binary.Write(&b, binary.LittleEndian, uint32(1)) // ndims
		binary.Write(&b, binary.LittleEndian, dim)
		binary.Write(&b, binary.LittleEndian, uint32(0)) // F32
		binary.Write(&b, binary.LittleEndian, uint64(0)) // offset
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
		b.Write(make([]byte, 64))
		return b.Bytes()
	}

	cases := map[string]uint64{
		"dim 2^63 (negative int)":  1 << 63,
		"dim 2^62 (n*4 wraps)":     1 << 62,
		"dim just over cap":        (1 << 40) + 1,
		"huge but positive int":    1 << 50,
	}
	for name, dim := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: reader PANICKED: %v", name, r)
				}
			}()
			if _, err := Read(bytes.NewReader(mk(dim))); err == nil {
				t.Errorf("%s: must error", name)
			}
		}()
	}
}
