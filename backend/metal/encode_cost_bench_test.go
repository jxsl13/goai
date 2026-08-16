//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
	"time"
)

// TestEncodeHostCost measures the HOST cost of recording one dispatch — the CPU time spent
// encoding, with no commit and no GPU work in the timed region.
//
// Every recorded op used to allocate a fresh MTLBuffer via newBufferWithBytes purely to carry a
// handful of ints (shapes, op codes). That is a driver allocation on the encode path, paid ~330x
// per decoded token. setBytes inlines constants under 4 KB straight into the command buffer and
// allocates nothing. Measured, 3 alternations: 2.79-3.86 -> 0.59-0.72 us/dispatch, about 5x.
//
// HONEST SCOPE: this did NOT move end-to-end decode throughput (138.2/142.1/139.8 vs
// 139.4/143.1/138.5 tok/s, interleaved — within noise). Encoding overlaps GPU execution, so at the
// current ratio (GPU 4.97 ms/token vs ~0.95 ms of encode) it is entirely hidden. It is kept because
// it is strictly less work, removes 23 per-dispatch driver allocations, and stops being hidden as
// GPU time shrinks. The remaining ~2.2 ms/token of wall-minus-GPU is SERIAL host time — logits
// download, sampling, submission latency — and is a separate problem this does not address.
//
// Reported with a loose ceiling; absolute microseconds are machine-dependent.
func TestEncodeHostCost(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	a, _ := NewDeviceBufferF32(make([]float32, 2048))
	b, _ := NewDeviceBufferF32(make([]float32, 2048))
	defer a.Release()
	defer b.Release()
	meas := func(n int) float64 {
		best := 1e18
		for range 30 {
			r, err := NewRecorder()
			if err != nil {
				t.Fatal(err)
			}
			start := time.Now()
			for range n {
				if err := r.BinaryN(a, b, b, 0, 2048); err != nil {
					t.Fatal(err)
				}
			}
			d := time.Since(start).Seconds() // encode only — commit/wait are outside the clock
			if d < best {
				best = d
			}
			r.Commit()
			r.Wait()
			r.Free()
		}
		return best
	}
	lo, hi := meas(32), meas(512)
	us := (hi - lo) / (512 - 32) * 1e6
	fmt.Printf("encode host cost: %.2f us/dispatch\n", us)
	if us > 2.0 {
		t.Errorf("%.2f us/dispatch — encode has regressed toward the per-dispatch buffer allocation (was 2.8-3.9 us before setBytes)", us)
	}
}
