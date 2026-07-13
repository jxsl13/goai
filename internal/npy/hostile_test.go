package npy

import (
	"bytes"
	"encoding/binary"
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
