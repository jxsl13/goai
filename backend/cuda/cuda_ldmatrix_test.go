//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"testing"
)

// TestZZLdmatrixLayout empirically prints the ldmatrix.x4.b16 fragment layout: for each (lane,reg),
// which source (matrix,row,col) its two b16 halves hold. This maps the layout the int8 mma.s8 A/B
// fragments must satisfy to be loaded via ldmatrix (the lever to beat cuBLAS-f16 on prefill).
func TestZZLdmatrixLayout(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	out := ldmatrixProbe()
	if out == nil {
		t.Fatal("ldmatrixProbe failed (ldmatrix may not compile in NVRTC)")
	}
	dec := func(v uint16) (m, r, c int) { return int(v>>6) & 3, int(v>>3) & 7, int(v) & 7 }
	// Print lanes 0, 1, 2, 3, 4, 8 to see the pattern (gid=lane>>2, tid=lane&3 in mma terms).
	for _, lane := range []int{0, 1, 2, 3, 4, 5, 8, 16, 24} {
		var s string
		for reg := 0; reg < 4; reg++ {
			v := out[lane*4+reg]
			lo, hi := uint16(v&0xFFFF), uint16(v>>16)
			m0, r0, c0 := dec(lo)
			m1, r1, c1 := dec(hi)
			s += "  r" + itoaL(reg) + "=[m" + itoaL(m0) + "r" + itoaL(r0) + "c" + itoaL(c0) + "|m" + itoaL(m1) + "r" + itoaL(r1) + "c" + itoaL(c1) + "]"
		}
		t.Logf("lane %2d (gid=%d tid=%d):%s", lane, lane>>2, lane&3, s)
	}
	t.Log("ldmatrix.x4.b16 LAYOUT MAPPED (see per-lane [matrix row col] of each reg's lo|hi b16 half)")
}

func itoaL(n int) string {
	if n == 0 {
		return "0"
	}
	var b [4]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// TestZZLdmatrixX2Layout maps ldmatrix.x2.b16 and checks it yields the mma.s8 B fragment
// (rb0 = N-row gid / K-b16 {tid*2,tid*2+1} of matrix0; rb1 = same cols of matrix1) with NO transpose.
func TestZZLdmatrixX2Layout(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	out := ldmatrixProbe2()
	if out == nil {
		t.Fatal("ldmatrixProbe2 failed")
	}
	dec := func(v uint16) (m, r, c int) { return int(v>>6) & 3, int(v>>3) & 7, int(v) & 7 }
	ok := true
	for lane := 0; lane < 32; lane++ {
		gid, tid := lane>>2, lane&3
		for reg := 0; reg < 2; reg++ {
			v := out[lane*2+reg]
			lo, hi := uint16(v&0xFFFF), uint16(v>>16)
			m0, r0, c0 := dec(lo)
			m1, r1, c1 := dec(hi)
			// expected: matrix==reg, row==gid, cols == tid*2 (lo) and tid*2+1 (hi)
			if m0 != reg || r0 != gid || c0 != tid*2 || m1 != reg || r1 != gid || c1 != tid*2+1 {
				if lane < 8 {
					t.Logf("lane %d reg %d: got [m%dr%dc%d|m%dr%dc%d], want matrix=%d row=%d cols=%d,%d",
						lane, reg, m0, r0, c0, m1, r1, c1, reg, gid, tid*2, tid*2+1)
				}
				ok = false
			}
		}
	}
	if ok {
		t.Log("ldmatrix.x2.b16 MATCHES the mma.s8 B fragment layout (reg=Khalf, row=gid, cols=tid*2,+1) — NO transpose needed for [N][K] weights")
	} else {
		t.Log("ldmatrix.x2 does NOT directly match B — .trans or a different tile arrangement needed (see logged lanes)")
	}
}
