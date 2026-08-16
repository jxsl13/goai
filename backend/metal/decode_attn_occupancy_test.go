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
//	mha_decode generic   1024
//	mha_decode dk=64      384
//	mha_decode dk=128    1024
//	split-K pass 1        384
//	split-K pass 2        448
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
// Split-K pass 1 carries the same 384 ceiling, from the same cause — acc[64] plus q[64] per thread.
// That matters because TestDecodeAttnRoofline ruled out bandwidth (8-14% of peak) and compute (~1%),
// leaving the per-key dependent chain, and this confirms register pressure is a real constraint on
// it rather than a guess.
//
// What it does NOT license is simply shrinking the accumulator. Softmax needs the FULL 64-dim dot
// before its exp, so splitting dk across passes means materialising per-key probabilities somewhere
// instead of just halving acc[] — a redesign of the inner loop, not a tweak. And the history above
// warns which way it can go: the 1024-thread variants reach that number by SPILLING, and spilling
// measured 5.9-8.7x slower. Any restructuring has to raise occupancy without putting the arrays back
// in memory, and must be measured against the current kernel rather than against the ceiling.
//
// Reported, not asserted: the value is compiler- and device-dependent.
func TestDecodeAttnOccupancy(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	// build the split-K pipelines so their occupancy is observable too
	q, _ := NewDeviceBufferF32(make([]float32, 2048))
	k, _ := NewDeviceBufferF32(make([]float32, 512*256))
	v, _ := NewDeviceBufferF32(make([]float32, 512*256))
	o, _ := NewDeviceBufferF32(make([]float32, 2048))
	if r, err := NewRecorder(); err == nil {
		_ = r.MHAAt(q, k, v, o, 0, 1, 512, 2048, 32, 4, 64, 1, 0, 0.125)
		r.Commit()
		r.Wait()
		r.Free()
	}
	for _, b := range []*DeviceBuffer{q, k, v, o} {
		b.Release()
	}
	g, d64, d128 := ProbeDecodeAttnOccupancy()
	if g < 0 {
		t.Skip("pipelines unavailable")
	}
	p1, p2 := ProbeSplitKOccupancy()
	fmt.Printf("OCC mha_decode generic=%d dk64=%d dk128=%d splitK_p1=%d splitK_p2=%d\n", g, d64, d128, p1, p2)
	if d64 <= 0 || d64 > 1024 {
		t.Errorf("implausible dk64 occupancy %d", d64)
	}
}
