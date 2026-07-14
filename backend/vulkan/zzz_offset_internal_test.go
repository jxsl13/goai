//go:build vulkan && cgo

package vulkan

import (
	"math"
	"testing"
)

// §T613 (fused QKV): RoPEAt and MHAAt read/write at a float-element offset inside a wider
// buffer — the sub-row views of one combined QKV projection output. Parity here is EXACT:
// the offset variant runs the SAME kernel on the SAME values, so results must be
// bit-identical to the offset-0 baseline, and bytes before the view must stay untouched.

func TestRecorderRoPEAtOffsetParity(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const seq, heads, hd = 3, 4, 16
	const width = heads * hd
	const half = hd / 2
	const pad = 37 // deliberately not 256-byte aligned: offsets are element indices
	const pos = 11
	src := make([]float32, seq*width)
	for i := range src {
		src[i] = float32((i%19)-9) * 0.05
	}
	inv := make([]float32, half)
	for i := range inv {
		inv[i] = float32(1.0 / math.Pow(10000, float64(2*i)/float64(hd)))
	}
	invB, err := NewDeviceBufferF32(inv)
	if err != nil {
		t.Fatal(err)
	}
	defer invB.Release()

	// baseline: plain RoPE on an unpadded buffer.
	base, _ := NewDeviceBufferF32(src)
	defer base.Release()
	rec, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec.RoPE(base, invB, base, seq, width, heads, hd, half, pos, 1); err != nil {
		t.Fatal(err)
	}
	if err := rec.Finish(); err != nil {
		t.Fatal(err)
	}
	rec.Free()
	want := make([]float32, seq*width)
	if err := base.DownloadF32(want); err != nil {
		t.Fatal(err)
	}

	// offset run: same rows living pad elements into a wider buffer, rotated in place.
	padded := make([]float32, pad+seq*width)
	for i := range pad {
		padded[i] = float32(i) + 0.5 // sentinel prefix — must survive untouched
	}
	copy(padded[pad:], src)
	pb, _ := NewDeviceBufferF32(padded)
	defer pb.Release()
	rec2, err := NewRecorder()
	if err != nil {
		t.Fatal(err)
	}
	if err := rec2.RoPEAt(pb, invB, pb, pad, seq, width, heads, hd, half, pos, 1); err != nil {
		t.Fatal(err)
	}
	if err := rec2.Finish(); err != nil {
		t.Fatal(err)
	}
	rec2.Free()
	got := make([]float32, pad+seq*width)
	if err := pb.DownloadF32(got); err != nil {
		t.Fatal(err)
	}
	for i := range pad {
		if got[i] != float32(i)+0.5 {
			t.Fatalf("prefix clobbered at %d: %v", i, got[i])
		}
	}
	for i, w := range want {
		if got[pad+i] != w {
			t.Fatalf("RoPEAt diverges at %d: %v vs %v", i, got[pad+i], w)
		}
	}
}

func TestRecorderMHAAtOffsetParity(t *testing.T) {
	if !Available() {
		t.Skip("vulkan: no compute-capable device (§V4)")
	}
	const sq, sk, heads, kvHeads, dk = 1, 9, 4, 2, 16
	const dm = heads * dk
	const dkv = kvHeads * dk
	const pad = 53
	scale := float32(1.0 / math.Sqrt(float64(dk)))
	qSrc := make([]float32, sq*dm)
	kSrc := make([]float32, sk*dkv)
	vSrc := make([]float32, sk*dkv)
	for i := range qSrc {
		qSrc[i] = float32((i%17)-8) * 0.05
	}
	for i := range kSrc {
		kSrc[i] = float32((i%13)-6) * 0.06
		vSrc[i] = float32((i%11)-5) * 0.07
	}
	k, _ := NewDeviceBufferF32(kSrc)
	defer k.Release()
	v, _ := NewDeviceBufferF32(vSrc)
	defer v.Release()

	run := func(qBuf *DeviceBuffer, qOff int) []float32 {
		o, _ := NewDeviceBufferF32(make([]float32, sq*dm))
		defer o.Release()
		rec, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		if err := rec.MHAAt(qBuf, k, v, o, qOff, sq, sk, dm, heads, kvHeads, dk, 1, 0, scale); err != nil {
			t.Fatal(err)
		}
		if err := rec.Finish(); err != nil {
			t.Fatal(err)
		}
		rec.Free()
		out := make([]float32, sq*dm)
		if err := o.DownloadF32(out); err != nil {
			t.Fatal(err)
		}
		return out
	}

	qPlain, _ := NewDeviceBufferF32(qSrc)
	defer qPlain.Release()
	want := run(qPlain, 0)

	padded := make([]float32, pad+sq*dm)
	copy(padded[pad:], qSrc)
	qPad, _ := NewDeviceBufferF32(padded)
	defer qPad.Release()
	got := run(qPad, pad)

	for i, w := range want {
		if got[i] != w {
			t.Fatalf("MHAAt diverges at %d: %v vs %v", i, got[i], w)
		}
	}
}
