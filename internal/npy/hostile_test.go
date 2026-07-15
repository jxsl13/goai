package npy

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// §V15 regression (fuzz-found): a v2/v3 header claiming a multi-GB length must
// be rejected before allocation, not make([]byte, hlen) and OOM-crash.
func TestHostileHeaderLength(t *testing.T) {
	mk := func(major byte, hlen uint32) []byte {
		var b bytes.Buffer
		b.Write(magic)
		b.WriteByte(major)
		b.WriteByte(0)
		binary.Write(&b, binary.LittleEndian, hlen)
		return b.Bytes() // no actual header body → must error, not allocate hlen
	}
	cases := map[string][]byte{
		"v2 4GB-1 header": mk(2, 0xFFFFFFFF),
		"v2 1.7GB header": mk(2, 0x69d7481e), // the fuzz-found seed
		"v3 over-cap":     mk(3, (64<<20)+1),
	}
	for name, data := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panicked: %v", name, r)
				}
			}()
			if _, err := Read(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: must error", name)
			}
		}()
	}
}

// §V15 regression: the element-count cap must use an overflow-safe product.
// The old post-multiply check let shape (2^24, 2^40) wrap uint64 to exactly 0
// (2^24 * 2^40 = 2^64), bypassing the cap: Read returned a nil-error tensor
// whose shape claims 2^64 elements over empty storage.
func TestHostileNumelOverflow(t *testing.T) {
	mk := func(shape string) []byte {
		hdr := "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape + ", }"
		pad := (64 - (10+len(hdr)+1)%64) % 64
		hdr += strings.Repeat(" ", pad) + "\n"
		var b bytes.Buffer
		b.Write(magic)
		b.Write([]byte{1, 0})
		binary.Write(&b, binary.LittleEndian, uint16(len(hdr)))
		b.WriteString(hdr)
		return b.Bytes()
	}
	cases := map[string][]byte{
		"wrap to zero (2^24 x 2^40)":  mk("(16777216, 1099511627776)"),
		"wrap to small (2^25 x 2^40)": mk("(33554432, 1099511627776)"),
		"over cap plain":              mk("(1099511627776, 2)"),
	}
	for name, data := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panicked: %v", name, r)
				}
			}()
			if _, err := Read(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: must error, cap bypassed", name)
			}
		}()
	}
}
