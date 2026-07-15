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
		"dim 2^63 (negative int)": 1 << 63,
		"dim 2^62 (n*4 wraps)":    1 << 62,
		"dim just over cap":       (1 << 40) + 1,
		"huge but positive int":   1 << 50,
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

// §V15/B28 regression: the dim-product cap must be overflow-safe. The old
// post-multiply check wrapped uint64 — dims 2^40 × 2^40 give 2^80 ≡ 0 — so a
// hostile header passed the cap and Read returned a nil-error tensor whose
// shape claims 2^80 elements over empty storage.
func TestHostileDimProductOverflow(t *testing.T) {
	mk := func(dims ...uint64) []byte {
		var b bytes.Buffer
		b.Write([]byte{0x47, 0x47, 0x55, 0x46})
		binary.Write(&b, binary.LittleEndian, uint32(3))
		binary.Write(&b, binary.LittleEndian, uint64(1)) // tensors
		binary.Write(&b, binary.LittleEndian, uint64(0)) // kvs
		binary.Write(&b, binary.LittleEndian, uint64(1)) // name len
		b.WriteString("t")
		binary.Write(&b, binary.LittleEndian, uint32(len(dims)))
		for _, d := range dims {
			binary.Write(&b, binary.LittleEndian, d)
		}
		binary.Write(&b, binary.LittleEndian, uint32(0)) // F32
		binary.Write(&b, binary.LittleEndian, uint64(0)) // offset
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
		b.Write(make([]byte, 64))
		return b.Bytes()
	}
	cases := map[string][]byte{
		"wrap to zero (2^40 x 2^40)": mk(1<<40, 1<<40),
		"wrap (2^30 x4)":             mk(1<<30, 1<<30, 1<<30, 1<<30),
		"over cap product":           mk(1<<39, 1<<39),
	}
	for name, data := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: PANICKED: %v", name, r)
				}
			}()
			if _, err := Read(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: Read must error, cap bypassed", name)
			}
			if _, err := ReadRaw(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: ReadRaw must error, cap bypassed", name)
			}
		}()
	}
}

// §B47: the tensor-section bounds check must use the subtraction form — an offset
// near 2^64 makes offset+need WRAP uint64, sneak past an addition-form check, and
// panic the slice. Both Read (decodeTensor) and ReadRaw share the check.
func TestHostileTensorOffsetOverflow(t *testing.T) {
	mk := func(offset uint64) []byte {
		var b bytes.Buffer
		b.Write([]byte{0x47, 0x47, 0x55, 0x46})
		binary.Write(&b, binary.LittleEndian, uint32(3))
		binary.Write(&b, binary.LittleEndian, uint64(1)) // tensors
		binary.Write(&b, binary.LittleEndian, uint64(0)) // kvs
		binary.Write(&b, binary.LittleEndian, uint64(1)) // name len
		b.WriteString("t")
		binary.Write(&b, binary.LittleEndian, uint32(1)) // ndims
		binary.Write(&b, binary.LittleEndian, uint64(2)) // dim 2 → need 8 bytes
		binary.Write(&b, binary.LittleEndian, uint32(0)) // F32
		binary.Write(&b, binary.LittleEndian, offset)    // hostile offset
		for b.Len()%32 != 0 {
			b.WriteByte(0)
		}
		b.Write(make([]byte, 64))
		return b.Bytes()
	}
	offsets := map[string]uint64{
		"offset 2^64−10 (add wraps small)": ^uint64(0) - 9,
		"offset 2^64−1":                    ^uint64(0),
		"offset just past section":         1 << 20,
	}
	for name, off := range offsets {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: PANICKED: %v", name, r)
				}
			}()
			if _, err := Read(bytes.NewReader(mk(off))); err == nil {
				t.Errorf("%s: Read must error", name)
			}
			if _, err := ReadRaw(bytes.NewReader(mk(off))); err == nil {
				t.Errorf("%s: ReadRaw must error", name)
			}
		}()
	}
}
