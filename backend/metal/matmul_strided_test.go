//go:build darwin && cgo

package metal

import (
	"math"
	"testing"
)

func TestRecorderMatMulStridedB(t *testing.T) {
	if !Available() {
		t.Skip("metal: no gpu device — skipped")
	}
	const m, k, n, stride, offset = 3, 5, 4, 13, 6
	aHost := make([]float32, m*k)
	bHost := make([]float32, k*stride)
	for i := range aHost {
		aHost[i] = float32(i-5) * 0.07
	}
	for i := range bHost {
		bHost[i] = float32(i%17-8) * 0.03
	}
	a, err := NewDeviceBufferF32(aHost)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Release()
	b, err := NewDeviceBufferF32(bHost)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Release()
	c, err := NewDeviceBufferF32(make([]float32, m*n))
	if err != nil {
		t.Fatal(err)
	}
	defer c.Release()
	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.MatMulStridedB(a, b, c, m, k, n, stride, offset); err != nil {
		rec.Free()
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		rec.Free()
		t.Fatal(err)
	}
	rec.Free()
	got := make([]float32, m*n)
	if err := c.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for row := range m {
		for col := range n {
			var want float64
			for inner := range k {
				want += float64(aHost[row*k+inner]) * float64(bHost[inner*stride+offset+col])
			}
			if delta := math.Abs(float64(got[row*n+col]) - want); delta > 1e-5*math.Max(1, math.Abs(want)) {
				t.Fatalf("[%d,%d]: got %v want %v", row, col, got[row*n+col], want)
			}
		}
	}
}
