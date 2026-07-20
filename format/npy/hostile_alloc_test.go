package npy_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/npy"
)

// A malformed .npy whose header declares a huge shape must NOT allocate the whole
// tensor before discovering the payload is missing. LoadFile cross-checks the
// declared payload against the real file size (§B-DoS): a 138-byte file claiming a
// 1 GiB f64 array previously allocated 1024 MiB before failing with EOF; it now
// errors before tensor.New with zero payload allocation.
//
// The assertion is on the ALLOCATION, not just the error — an error that arrives
// after a multi-gigabyte make (or an unrecoverable OOM on an 8 TiB claim) is the
// exact failure this guards. The check is skipped only if the guard is removed,
// which is what makes the bound the real subject of the test.
func TestHostileNpyHugeShapeDoesNotAllocate(t *testing.T) {
	mkFile := func(shape string) string {
		hdr := "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape + ", }"
		for len(hdr)%64 != 63 {
			hdr += " "
		}
		hdr += "\n"
		buf := append([]byte("\x93NUMPY"), 1, 0)
		buf = binary.LittleEndian.AppendUint16(buf, uint16(len(hdr)))
		buf = append(buf, hdr...)
		p := filepath.Join(t.TempDir(), "hostile.npy")
		if err := os.WriteFile(p, buf, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	for _, shape := range []string{"(134217728,)", "(16777216, 8)"} { // 1 GiB, 1 GiB f64
		var m0, m1 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m0)
		_, err := npy.LoadFile(mkFile(shape))
		runtime.ReadMemStats(&m1)
		if err == nil {
			t.Errorf("shape %s: want an error for a header exceeding the file, got nil", shape)
		} else if !strings.Contains(err.Error(), "refusing to allocate") {
			t.Errorf("shape %s: unexpected error %v", shape, err)
		}
		if got := (m1.TotalAlloc - m0.TotalAlloc) >> 20; got > 16 {
			t.Errorf("shape %s: allocated %d MiB from a tiny hostile file — the size guard did not fire", shape, got)
		}
	}
}
