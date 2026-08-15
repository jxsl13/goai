//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// flash_mm_f32 computes prefill attention on the SIMD matrix units. It is CORRECT — this test
// holds it against the kernel prefill actually uses — but NOT yet faster, so nothing routes to it:
//
//	seq= 64  decode 200.5us  flashMM  221.6us  0.90x   151 GFLOP/s (2.2% peak)
//	seq=128  decode 520.2us  flashMM  615.9us  0.84x   218 GFLOP/s (3.2% peak)
//	seq=256  decode 1905us   flashMM 2020.9us  0.94x   266 GFLOP/s (3.9% peak)
//	seq=512  decode 7268us   flashMM 7191.2us  1.01x   299 GFLOP/s (4.4% peak)
//
// The matrix units are idle behind staging, exactly as in the first matrix-unit quantized matmul —
// and notably NOT because of dequantization, which attention has none of. Per K-tile a threadgroup
// stages 4096 floats (128 per thread) and runs a 64-iteration softmax on 8 of its 32 lanes, to feed
// 64 matrix MACs: a 64:1 ratio.
//
// The fix is the one that worked on the quantized kernel: more simdgroups per threadgroup, so one
// K/V tile serves 32 query rows instead of 8 and the softmax parallelizes across the group.
//
// The seq values matter. seq<=32 fits ONE K tile and exercises no cross-tile rescale; the first
// version passed there at 4.5e-05 and was wrong at 1.8e-02 for seq=64, because the per-row softmax
// correction was broadcast from lane 0 and so applied one row's rescale to all eight. Any test that
// stopped at seq=32 would have called that kernel correct.
func TestFlashMMMatchesReference(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const heads, kvh, dk = 32, 4, 64
	dm := heads * dk
	for _, seq := range []int{8, 32, 64, 129} {
		dkv := kvh * dk
		qh := make([]float32, seq*dm)
		kh := make([]float32, seq*dkv)
		vh := make([]float32, seq*dkv)
		for i := range qh {
			qh[i] = float32(math.Sin(float64(i)*0.31)) * 0.5
		}
		for i := range kh {
			kh[i] = float32(math.Cos(float64(i)*0.17)) * 0.5
			vh[i] = float32(math.Sin(float64(i)*0.11)) * 0.5
		}
		q, _ := NewDeviceBufferF32(qh)
		k, _ := NewDeviceBufferF32(kh)
		v, _ := NewDeviceBufferF32(vh)
		o1, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		o2, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		if e := r.MHA(q, k, v, o1, seq, seq, dm, heads, kvh, dk, 1, 0, 0.125); e != nil {
			t.Fatal(e)
		}
		if e := r.FlashMM(q, k, v, o2, seq, seq, dm, heads, kvh, 1, 0.125); e != nil {
			t.Skipf("FlashMM: %v", e)
		}
		r.Commit()
		r.Wait()
		r.Free()
		a := make([]float32, seq*dm)
		b := make([]float32, seq*dm)
		o1.DownloadF32(a)
		o2.DownloadF32(b)
		var maxRel float64
		bad := 0
		for i := range a {
			d := math.Abs(float64(a[i] - b[i]))
			if math.IsNaN(d) {
				bad++
				continue
			}
			den := math.Max(1e-3, math.Abs(float64(a[i])))
			if rr := d / den; rr > maxRel {
				maxRel = rr
			}
		}
		fmt.Printf("FMM seq=%4d  maxRel=%.3e  NaN=%d\n", seq, maxRel, bad)
		q.Release()
		k.Release()
		v.Release()
		o1.Release()
		o2.Release()
	}
}
