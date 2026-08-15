//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestDQGemmCostSplit splits the prefill dequant+GEMM cost into its two halves, because which half
// dominates decides what to optimize next — and the intuitive answer is wrong.
//
// The expansion LOOKS like the problem: it is a fixed cost independent of M, so at small M it
// cannot amortize, and short-prompt prefill (pp64 ~0.46x of llama.cpp) is much weaker than long
// (pp1024 ~0.92x). Measured on an M2 Pro, K=2048 N=5632:
//
//	dequant-only            280.8us   (writes 46.1 MB f32)
//	M=  64  gemm-only       464.5us   dequant share 37.7%
//	M= 256  gemm-only      1143.8us   dequant share 19.7%
//	M=1024  gemm-only      4442.0us   dequant share  5.9%
//
// The expansion is the MINORITY of the cost at every M, 37.7% even at M=64. Halving it (expanding
// to f16 rather than f32) is therefore worth at most ~1.2x there, not the ~1.5x the fixed-cost
// story suggests.
//
// The GEMM is where the small-M weakness lives, and the reason is the same 46 MB:
//
//	M=  64   1.48 GFLOP / 464.5us = 3.18 TFLOP/s (47% of peak); reads 46 MB at  99 GB/s
//	M=1024  23.6 GFLOP / 4442.0us = 5.31 TFLOP/s (78% of peak)
//
// At M=64 the GEMM is throttled on the weight read — ~99 GB/s against the ~180 GB/s this machine
// sustains — with only 64 rows of arithmetic to hide it behind. By M=1024 there is enough work per
// weight byte that it becomes compute-bound and reaches 78% of peak.
//
// So f16 is worth trying, but for the GEMM's READ rather than the expansion's WRITE: it halves both,
// and the benefit concentrates at exactly the small-M shapes that are weak, while being close to
// neutral at M=1024 where the path is already near the FLOP ceiling. Recorded here so the next
// attempt targets the right half.
//
// Reported, not asserted on absolute timings; machine-dependent.
func TestDQGemmCostSplit(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const K, N = 2048, 5632
	nb := K / 256
	raw := make([]byte, N*nb*144)
	for i := range raw {
		raw[i] = byte(i*31 + 7)
	}
	rw, err := Backend{}.UploadQuant(raw, 12, N, K)
	if err != nil {
		t.Skip(err)
	}
	rq := rw.(*ResidentQWeight)
	defer rq.Close()
	wf, _ := NewDeviceBufferF32(make([]float32, K*N))
	defer wf.Release()

	timeIt := func(f func(*Recorder)) float64 {
		best := 1e18
		for range 9 {
			r, _ := NewRecorder()
			for range 3 {
				f(r)
			}
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds() / 3; d < best {
				best = d
			}
			r.Free()
		}
		return best * 1e6
	}
	dq := timeIt(func(r *Recorder) { _ = r.DequantQ4K(rq, wf) })
	fmt.Printf("SPLIT dequant-only = %.1fus (writes %.1f MB f32)\n", dq, float64(K*N*4)/1e6)
	for _, M := range []int{64, 256, 1024} {
		x, _ := NewDeviceBufferF32(make([]float32, M*K))
		o, _ := NewDeviceBufferF32(make([]float32, M*N))
		g := timeIt(func(r *Recorder) { _ = r.MatMul(x, wf, o, M, K, N) })
		fmt.Printf("SPLIT M=%4d gemm-only = %8.1fus  dequant share = %4.1f%%\n", M, g, 100*dq/(dq+g))
		x.Release()
		o.Release()
	}
}
