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
// Both are quadratic, as attention must be. What matters is the CONSTANT, and measured across the
// prompt lengths prefill actually runs at it is the dominant term long before anyone would call a
// prompt long:
//
//	seq   attention x22 layers   prefill total   share    attention GFLOP/s
//	  64        4.4 ms                ~84 ms       5%          167
//	 256       41.9 ms               ~195 ms      21%          282
//	 512      160.1 ms               ~427 ms      37%          295
//	1024      ~640 ms               ~1102 ms      58%            —
//
// That is why GoAI's prompt throughput PEAKS at n=256 (1312 tok/s) and then falls to 929 at
// n=1024, while llama.cpp stays flat: pp64 1773, pp256 2142, pp512 2202, pp1024 2141. A flat curve
// against a falling one is the signature of an attention kernel that is O(n^2) with a bad constant.
//
// This reprioritizes the remaining prefill work. Attention at 3-7x would give pp1024 of
// 1516-1851 tok/s, i.e. 0.71-0.86x of llama.cpp, from 0.43x today. The quantized-GEMM project
// removes expansion — 0.99 of 3.79 ms/layer at n=64, ~1.35x on the matmul path — and expansion
// amortizes away at long prompts anyway, so it is worth strictly less.
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
