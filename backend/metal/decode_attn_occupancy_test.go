//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestDecodeAttnOccupancy records the register pressure of the decode attention pipelines, because
// it explains why the dk=64 specialization is shaped the way it is and warns against "improving" it.
//
// maxTotalThreadsPerThreadgroup is dictated by how many registers a kernel needs per thread. On an
// M2 Pro:
//
//	mha_decode generic  1024
//	mha_decode dk=64     384
//	mha_decode dk=128   1024
//
// The generic and dk=128 kernels reach 1024 not because they are lean but because they SPILL their
// q[128]/acc[128] arrays to memory — the compiler needs few registers precisely because the arrays
// are not in any. dk=64 keeps q[64]+acc[64] resident and pays for it with a 384-thread ceiling.
//
// That looks like a defect and is not: the register-resident dk=64 kernel measured 5.9-8.7x FASTER
// than the spilling generic one. Occupancy is the price of the win, not a missed one, so a change
// that raises this number by shrinking the per-thread arrays is very likely to be landing back on
// the spill. Anyone reading "384 out of 1024" as headroom should read that history first.
//
// Reported, not asserted: the value is compiler- and device-dependent.
func TestDecodeAttnOccupancy(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	g, d64, d128 := ProbeDecodeAttnOccupancy()
	if g < 0 {
		t.Skip("pipelines unavailable")
	}
	fmt.Printf("OCC mha_decode generic=%d dk64=%d dk128=%d\n", g, d64, d128)
	if d64 <= 0 || d64 > 1024 {
		t.Errorf("implausible dk64 occupancy %d", d64)
	}
}
