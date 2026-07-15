package npy

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

// Regression: the shape-product cap must be overflow-safe. The old
// post-multiply check wrapped int64 — shape (2^30, 2^34) gives exactly
// 2^64 ≡ 0 — so a hostile header passed the cap and Load returned a
// nil-error tensor claiming 2^64 elements over empty storage.
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
		"wrap to zero (2^30 x 2^34)": mk("(1073741824, 17179869184)"),
		"wrap to small":              mk("(1073741824, 17179869185)"),
		"over cap plain":             mk("(1073741825, 2)"),
	}
	for name, data := range cases {
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("%s: panicked: %v", name, r)
				}
			}()
			if _, err := Load(bytes.NewReader(data)); err == nil {
				t.Errorf("%s: must error, cap bypassed", name)
			}
		}()
	}
}
