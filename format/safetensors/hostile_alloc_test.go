package safetensors_test

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/jxsl13/goai/format/safetensors"
)

// A malformed .safetensors whose header declares a tensor larger than the file
// must NOT allocate it before the read fails. LoadFile cross-checks each tensor's
// declared data-offset end against the real file size (§B-DoS): a 79-byte file
// claiming a 1 GiB F64 tensor previously allocated 1024 MiB before failing with
// EOF; it now errors before tensor.New with zero payload allocation.
//
// The assertion is on the ALLOCATION, not just the error — safetensors read paths
// take untrusted downloaded files, and an 8 TiB declared tensor would otherwise
// reach an unrecoverable out-of-memory make.
func TestHostileSafetensorsHugeTensorDoesNotAllocate(t *testing.T) {
	mkFile := func(hdr string) string {
		buf := binary.LittleEndian.AppendUint64(nil, uint64(len(hdr)))
		buf = append(buf, hdr...)
		p := filepath.Join(t.TempDir(), "hostile.safetensors")
		if err := os.WriteFile(p, buf, 0o644); err != nil {
			t.Fatal(err)
		}
		return p
	}
	cases := map[string]string{
		"1 GiB F64, empty payload":               `{"w":{"dtype":"F64","shape":[134217728],"data_offsets":[0,1073741824]}}`,
		"1 GiB F32, empty payload":               `{"w":{"dtype":"F32","shape":[268435456],"data_offsets":[0,1073741824]}}`,
		"huge second tensor after a valid first": `{"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"w":{"dtype":"F64","shape":[134217728],"data_offsets":[4,1073741828]}}`,
	}
	for name, hdr := range cases {
		var m0, m1 runtime.MemStats
		runtime.GC()
		runtime.ReadMemStats(&m0)
		_, _, err := safetensors.LoadFile(mkFile(hdr))
		runtime.ReadMemStats(&m1)
		if err == nil {
			t.Errorf("%s: want an error for a tensor exceeding the file, got nil", name)
		} else if !strings.Contains(err.Error(), "refusing to allocate") {
			t.Errorf("%s: unexpected error %v", name, err)
		}
		if got := (m1.TotalAlloc - m0.TotalAlloc) >> 20; got > 16 {
			t.Errorf("%s: allocated %d MiB from a tiny hostile file — the size guard did not fire", name, got)
		}
	}
}
