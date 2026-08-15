//go:build darwin && cgo

package metal

import (
	"fmt"
	"math"
	"testing"
)

// TestPrefillAttentionRouting records that NEITHER existing attention kernel is suited to prefill,
// so the obvious fix — reroute causal sq>1 away from the decode kernel to the flash kernel — is not
// available. Measured on an M2 Pro, heads=32 kvHeads=4 dk=64:
//
//	seq= 64  decode  164.3us (204 GFLOP/s, 3.0% peak)  flash  228.0us  0.72x
//	seq=128  decode  521.4us (257 GFLOP/s, 3.8% peak)  flash  535.5us  0.97x
//	seq=256  decode 1902.3us (282 GFLOP/s, 4.2% peak)  flash 1987.5us  0.96x
//
// They agree to 5.7e-05, so both are correct; the flash kernel is simply not faster here despite
// being the tiled one, and both sit at 3-4% of the 6.8 TFLOP/s peak.
//
// Both are quadratic, as attention must be (x3.2 then x3.6 per doubling). What matters is the
// CONSTANT: at seq=256 attention is 41.9 ms of a ~427 ms prefill (10%), and the share grows with
// prompt length. At short prompts it is ~4% and invisible, which is why it went unmeasured until
// the prefill budget was closed.
//
// Closing this needs a genuine flash-attention kernel — Q tile resident, K/V streamed in tiles
// through threadgroup memory, online softmax — not a routing change. Both current kernels
// re-read K/V per query row in the shape that matters here.
//
// This test guards the AGREEMENT (a correctness property) and reports the timings; it does not
// assert a ratio, since the point is that there is no useful ratio to assert.
func TestPrefillAttentionRouting(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const heads, kvh, dk = 32, 4, 64
	dm := heads * dk
	for _, seq := range []int{64, 128} {
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
		if err := r.MHA(q, k, v, o1, seq, seq, dm, heads, kvh, dk, 1, 0, 0.125); err != nil {
			t.Fatal(err)
		}
		if err := r.flashattn(q, k, v, o2, seq, dm, heads, dk, 1, kvh, 0.125); err != nil {
			t.Fatal(err)
		}
		if err := r.Commit(); err != nil {
			t.Fatal(err)
		}
		if err := r.Wait(); err != nil {
			t.Fatal(err)
		}
		r.Free()
		a := make([]float32, seq*dm)
		b := make([]float32, seq*dm)
		if err := o1.DownloadF32(a); err != nil {
			t.Fatal(err)
		}
		if err := o2.DownloadF32(b); err != nil {
			t.Fatal(err)
		}
		var maxRel float64
		for i := range a {
			d := math.Abs(float64(a[i] - b[i]))
			if math.IsNaN(d) {
				t.Fatalf("seq=%d: NaN at element %d", seq, i)
			}
			den := math.Max(1e-3, math.Abs(float64(a[i])))
			if rr := d / den; rr > maxRel {
				maxRel = rr
			}
		}
		fmt.Printf("prefill attention seq=%d: decode-kernel vs flash agree to %.2e\n", seq, maxRel)
		if maxRel > 2e-4 {
			t.Errorf("seq=%d: the two causal attention kernels disagree by %.2e", seq, maxRel)
		}
		q.Release()
		k.Release()
		v.Release()
		o1.Release()
		o2.Release()
	}
}
