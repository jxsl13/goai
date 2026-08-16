//go:build darwin && cgo

package metal

// TestDecodeAttnRoofline locates decode attention on the roofline, and with it prices — and rejects
// — an f16 KV cache before implementing one.
//
// f16 KV is what llama.cpp uses and it halves KV bytes, so it looks like the obvious next move for
// attention's ~1.5x degradation with context. It only pays if the kernel is bandwidth-bound.
// Measured (M2 Pro, 32 heads over 4 KV heads, dk 64, split-K enabled):
//
//	sk= 512   71.91us  14.6 GB/s ( 8.1% of ~180)   58.3 GFLOP/s (0.9% of ~6800)
//	sk=1536  146.13us  21.5 GB/s (12.0%)           86.1 GFLOP/s (1.3%)
//	sk=2048  168.03us  25.0 GB/s (13.9%)           99.8 GFLOP/s (1.5%)
//
// It is neither. At 8-14% of bandwidth and ~1% of FLOP peak the kernel is LATENCY-bound, so halving
// the bytes cannot remove more than the bytes' own share of the time: at most 4-7% of attention,
// and attention is ~34% of a token at ctx=1536, so ~0.2% end to end. Not worth a change to the
// decoder's KV memory layout.
//
// (f16 KV remains worth doing for CAPACITY — half the cache means twice the context in the same
// memory — but that is a different argument from speed, and this test only rules out the speed one.)
//
// This closes the third distinct hypothesis about this kernel. Occupancy was tested twice and
// failed: GQA-cooperative staging measured 0.53-0.75x (TestDecodeNonMatmulBreakdown) and the chunk
// sweep found the shipped chunk size already optimal (TestSplitKChunkSizeIsTuned). Bandwidth is
// ruled out here. What remains is the per-key dependent chain — an exp() and a 64-wide accumulate
// per key, with acc[64] held per thread, which is also what caps the dk=64 pipeline at 384 threads
// (TestDecodeAttnOccupancy). Any further win has to come from restructuring that inner loop, not
// from feeding it differently.
//
// Reported, not asserted on absolute rates; the peaks are this machine's.
import (
	"fmt"
	"testing"
)

func TestDecodeAttnRoofline(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const heads, kvh, dk = 32, 4, 64
	dm := heads * dk
	dkv := kvh * dk
	for _, sk := range []int{512, 1536, 2048} {
		q, _ := NewDeviceBufferF32(make([]float32, dm))
		k, _ := NewDeviceBufferF32(make([]float32, sk*dkv))
		v, _ := NewDeviceBufferF32(make([]float32, sk*dkv))
		o, _ := NewDeviceBufferF32(make([]float32, dm))
		best := 1e18
		for range 11 {
			r, _ := NewRecorder()
			for range 20 {
				_ = r.MHAAt(q, k, v, o, 0, 1, sk, dm, heads, kvh, dk, 1, 0, 0.125)
			}
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds() / 20; d < best {
				best = d
			}
			r.Free()
		}
		bytes := float64(2 * sk * dkv * 4)        // K and V, f32
		flops := float64(2 * 2 * heads * sk * dk) // dot + weighted acc
		fmt.Printf("ROOF sk=%5d  %7.2fus  %6.1f GB/s (%4.1f%% of 180)  %7.1f GFLOP/s (%4.1f%% of 6800)\n",
			sk, best*1e6, bytes/best/1e9, 100*(bytes/best/1e9)/180,
			flops/best/1e9, 100*(flops/best/1e9)/6800)
		fmt.Printf("ROOF   f16 KV would cut bytes to %.1f MB -> saves at most %.2fus of %.2fus (%.1f%%)\n",
			bytes/2/1e6, (bytes/2)/180e9*1e6, best*1e6, 100*((bytes/2)/180e9)/best)
		q.Release()
		k.Release()
		v.Release()
		o.Release()
	}
}
