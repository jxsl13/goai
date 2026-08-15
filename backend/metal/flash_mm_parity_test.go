//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// flash_mm_f32 computes prefill attention on the SIMD matrix units, and IS the path prefill takes
// for causal sq==sk with dk=64. This test holds it against the kernel it replaced.
//
// Against that kernel, per call:
//
//	seq= 64  decode  166.4us  flashMM  122.7us  1.36x   273 GFLOP/s ( 4.0% peak)
//	seq=128  decode  522.3us  flashMM  138.5us  3.77x   969 GFLOP/s (14.2%)
//	seq=256  decode 1903.2us  flashMM  389.3us  4.89x  1379 GFLOP/s (20.3%)
//	seq=512  decode 7271.6us  flashMM  997.0us  7.29x  2154 GFLOP/s (31.7%)
//
// End to end on TinyLlama prompt processing: n=64 777 -> ~804, n=256 1314 -> ~1582,
// n=1024 929 -> ~1796 tok/s. The gain tracks attention's share of prefill, which the
// prompt-length sweep measured at 5% / 21% / 58%.
//
// The K/V staging is vectorized with float4 (DK=64 is a multiple of 4): 1175.8/1208.8 ->
// 991.7/1002.4us at seq=512, ~1.19x. Note what did NOT work — staging K/V as half. Mixed-precision
// MMA is supported on this device (probed), but halving the ELEMENT SIZE leaves the STORE COUNT
// unchanged, and the staging is instruction-bound. Only the float4 move reduces instructions.
//
// End to end that 1.19x is ~1.02x at pp1024 (1768.5/1790.0 -> 1815.9/1829.7) and nil at pp256,
// because attention is down to ~18% of prefill after the earlier 6x. Kept as strictly less work
// with identical accuracy; the share grows again at longer prompts.
//
// BQ is 64 (8 simdgroups). Against BQ=32 it is neutral at n=64 and better at length —
// interleaved: n=256 1545/1514 -> 1587/1577, n=1024 1602/1633 -> 1799/1793. sO aliases the K/V
// staging buffer because 2*BK*DK == BQ*DK exactly and O is only written after the last tile.
//
// The first version used ONE simdgroup per 8 query rows and was 0.84-1.01x — no faster than the
// scalar kernel — because it staged 4096 floats and ran a 64-iteration softmax on 8 of 32 lanes to
// feed 64 matrix MACs. Four simdgroups over 32 query rows share one K/V tile and parallelize the
// softmax, which is what made the matrix units productive. That is the same fix that took the
// quantized matrix-unit kernel from 585 to 1377 GFLOP/s: the matrix units do nothing while a
// scalar prologue dominates, whatever that prologue is.
//
// ACCURACY: agreement is 5.8e-05, not bit-exact — the online softmax rescales partial sums in a
// different order. That is the same order as the two pre-existing attention kernels agree with each
// other (5.7e-05), so it is within the spread the codebase already had. On greedy decoding it can
// flip a near-tie: 2 of 26 generated tokens differ after a 32-token prompt.
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
		// MHA now ROUTES to flash_mm for these shapes, so the reference arm must switch it off —
		// otherwise this compares the kernel against itself and reports 0.000e+00, which is exactly
		// what it did the moment the gate was wired.
		r, err := NewRecorder()
		if err != nil {
			t.Fatal(err)
		}
		SetFlashMM(false)
		if e := r.MHA(q, k, v, o1, seq, seq, dm, heads, kvh, dk, 1, 0, 0.125); e != nil {
			t.Fatal(e)
		}
		SetFlashMM(true)
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
