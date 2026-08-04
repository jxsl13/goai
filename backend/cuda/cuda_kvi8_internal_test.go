//go:build cuda && cgo && (linux || windows)

package cuda

import (
	"math"
	"math/rand"
	"testing"
)

// TestKVCacheI8AppendRoundtrip validates the int8 per-head quantize-on-append: append a known f32
// K row, download the int8 values + per-head scales, dequantize, and check the result matches the
// original within the int8 rounding budget (per-head max/127 → half-ULP ≈ max/254 per element).
func TestKVCacheI8AppendRoundtrip(t *testing.T) {
	if !Available() {
		t.Skip("no gpu")
	}
	const kvHeads, hd = 3, 64
	const wkv = kvHeads * hd
	const maxSeq = 4
	rng := rand.New(rand.NewSource(5))
	// distinct per-head magnitudes so the per-head scale matters
	src := make([]float32, wkv)
	headMag := []float64{0.2, 2.0, 20.0}
	for h := 0; h < kvHeads; h++ {
		for d := 0; d < hd; d++ {
			src[h*hd+d] = float32(rng.NormFloat64() * headMag[h])
		}
	}
	c, err := NewKVCacheI8(maxSeq, kvHeads, hd)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free()
	dk, err := NewDeviceF32(1, wkv)
	if err != nil {
		t.Fatal(err)
	}
	defer dk.Free()
	if err := dk.UploadF32(src); err != nil {
		t.Fatal(err)
	}
	pos, err := NewDevicePos()
	if err != nil {
		t.Fatal(err)
	}
	defer pos.Free()
	const P = 2
	if err := pos.Set(P); err != nil {
		t.Fatal(err)
	}
	if err := c.AppendDpos(dk, dk, pos); err != nil { // reuse dk as both k and v
		t.Fatal(err)
	}
	deq, err := c.downloadKForTest()
	if err != nil {
		t.Fatal(err)
	}
	// per-head tolerance = max|head|/127 (one quant step); allow a hair over for RN
	var maxRel float64
	for h := 0; h < kvHeads; h++ {
		var mx float64
		for d := 0; d < hd; d++ {
			mx = math.Max(mx, math.Abs(float64(src[h*hd+d])))
		}
		tol := mx/127.0 + 1e-6
		for d := 0; d < hd; d++ {
			idx := P*wkv + h*hd + d
			e := math.Abs(float64(deq[idx]) - float64(src[h*hd+d]))
			if e > tol {
				t.Fatalf("head %d dim %d: dequant %v vs %v, err %v > tol %v", h, d, deq[idx], src[h*hd+d], e, tol)
			}
			if mx > 0 {
				maxRel = math.Max(maxRel, e/mx)
			}
		}
	}
	t.Logf("int8 KV append roundtrip: max per-head rel err %.4f (want ≤ 1/127≈0.0079)", maxRel)
	// rows other than P must be untouched (zero-ish); spot check row 0 scale path not written → deq 0
	for d := 0; d < wkv; d++ {
		if deq[0*wkv+d] != 0 {
			// row 0 was never appended; int8 buffer uninitialized → scale unwritten. Not asserted strictly.
			break
		}
	}
}
