//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
	"time"
)

// TestQ4KDecodeRoofline measures what the Q4_K M=1 (decode) kernel achieves against the machine's
// MEMORY bandwidth, which is the only ceiling that matters for a weight-streaming decode: every
// weight byte is read once per token and almost nothing is recomputed.
//
// It sweeps N past cache capacity ON PURPOSE. A decode-shaped weight (K=2048, N=2048 → 2.4 MB) fits
// in the M2 Pro's last-level cache, so timing it measures CACHE bandwidth and flatters the kernel;
// a real model streams hundreds of MB from DRAM per token. Reading the small shapes as if they
// predicted decode throughput is what made an earlier round conclude the kernel was instruction-bound
// and chase per-weight ALU work. At N=131072 the weight is ~151 MB, far past any cache, and the
// number that comes out is the honest one.
//
// Measured 2026-08-15 on an M2 Pro (200 GB/s peak): 185 GB/s at 151 MB, i.e. ~92% of peak. The kernel
// is bandwidth-bound AT the bandwidth limit, so no more than ~8% is available here — and a decode
// that runs far below this is losing its time somewhere other than the quantized matmul.
//
// Reported, not asserted, except for a deliberately loose floor: absolute bandwidth is machine- and
// thermal-dependent, and a tight threshold would be a flaky test rather than a guard.
func TestQ4KDecodeRoofline(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K = 2048
	for _, N := range []int{2048, 5632, 16384, 131072} {
		nb := K / 256
		bytes := N * nb * 144
		w := make([]byte, bytes)
		for i := range w {
			w[i] = byte(i*31 + 7)
		}
		rw, err := Backend{}.UploadQuant(w, 12, N, K)
		if err != nil {
			t.Skip(err)
		}
		rq := rw.(*ResidentQWeight)
		xb, _ := NewDeviceBufferF32(make([]float32, K))
		ob, _ := NewDeviceBufferF32(make([]float32, N))

		// Many ops per command buffer so the ~150us per-submit floor is a negligible fraction of
		// the buffer rather than something to subtract: differencing two noisy samples amplified
		// run-to-run variance well past the effect being measured.
		reps := 128
		if bytes > 20<<20 {
			reps = 16
		}
		best := 1e18
		for range 40 {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			for range reps {
				if err := r.QMatMulResident(xb, rq, ob, 1); err != nil {
					t.Fatal(err)
				}
			}
			start := time.Now()
			if err := r.Commit(); err != nil {
				t.Fatal(err)
			}
			if err := r.Wait(); err != nil {
				t.Fatal(err)
			}
			if d := time.Since(start).Seconds(); d < best {
				best = d
			}
			r.Free()
		}
		per := best / float64(reps)
		gbs := float64(bytes) / per / 1e9
		fmt.Printf("Q4K roofline K=%d N=%d weight=%.1fMB per-op=%.1fus %.1f GB/s\n",
			K, N, float64(bytes)/(1<<20), per*1e6, gbs)
		if gbs < 20 {
			t.Errorf("K=%d N=%d: %.1f GB/s — the decode kernel has fallen off the bandwidth roofline entirely", K, N, gbs)
		}
		rq.Close()
		xb.Release()
		ob.Release()
	}
}
