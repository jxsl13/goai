//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
)

// The matrix-unit Q4_K path (qmatmul_q4k_mm) serves M>=8 and had NO coverage when written —
// established by mutation, not assumed: zeroing its output left the whole metal suite green,
// because every existing Q4_K case runs at M=1 and never reaches it.
//
// This compares it against the cooperative kernel on the SAME weights, across shapes that exercise
// the partial-tile guards: M and N not multiples of 8, and M straddling the M>=8 threshold.
func TestQ4KMatrixUnitMatchesCooperative(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	// The path is off by default (it is slower than the cooperative kernel — see the note in
	// metal_bridge.m); enable it for the duration of this test.
	SetQ4KMatrixUnit(true)
	defer SetQ4KMatrixUnit(false)
	for _, c := range []struct{ M, K, N int }{
		{8, 256, 8},
		{8, 512, 16},
		{9, 256, 13}, // both dims partial
		{16, 768, 32},
		{33, 256, 40}, // M partial
		{64, 512, 24},
	} {
		nb := c.K / 256
		raw := make([]byte, c.N*nb*144)
		for i := range raw {
			raw[i] = byte(i*37 + 11)
		}
		x := make([]float32, c.M*c.K)
		for i := range x {
			x[i] = float32(math.Sin(float64(i)*0.37)) * 0.5
		}
		rw, err := Backend{}.UploadQuant(raw, 12, c.N, c.K)
		if err != nil {
			t.Skip(err)
		}
		rq := rw.(*ResidentQWeight)
		xb, _ := NewDeviceBufferF32(x)
		ob, _ := NewDeviceBufferF32(make([]float32, c.M*c.N))

		run := func() []float32 {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			if err := r.QMatMulResident(xb, rq, ob, c.M); err != nil {
				t.Fatal(err)
			}
			if err := r.Commit(); err != nil {
				t.Fatal(err)
			}
			if err := r.Wait(); err != nil {
				t.Fatal(err)
			}
			r.Free()
			got := make([]float32, c.M*c.N)
			if err := ob.DownloadF32(got); err != nil {
				t.Fatal(err)
			}
			return got
		}
		got := run()
		// Reference: the SCALAR kernel. It is the right comparand because it uses the same
		// per-element form this kernel does, x*(d*sc*q - dmin*m). The cooperative kernel instead
		// factors the min term out as d*sc*sum(x*q) - dmin*m*sum(x), which is algebraically equal
		// but rounds differently, so comparing against it measures that refactoring rather than
		// this kernel's correctness.
		SetQ4KMatrixUnit(false)
		SetQ4KCooperative(false)
		want := run()
		SetQ4KCooperative(true)
		SetQ4KMatrixUnit(true)

		var maxRel float64
		for i := range want {
			d := math.Abs(float64(got[i] - want[i]))
			den := math.Max(1, math.Abs(float64(want[i])))
			if r := d / den; r > maxRel {
				maxRel = r
			}
		}
		if maxRel > 2e-5 {
			t.Errorf("M=%d K=%d N=%d: matrix-unit vs cooperative max relative %.3e > 2e-5", c.M, c.K, c.N, maxRel)
		} else {
			t.Logf("M=%d K=%d N=%d: max relative %.3e", c.M, c.K, c.N, maxRel)
		}
		rq.Close()
		xb.Release()
		ob.Release()
	}
}
