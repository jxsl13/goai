//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestDecodeMatmulIsAtMemoryCeiling establishes that the decode-path quantized matmul has no
// bandwidth headroom left, so effort spent tuning that kernel is wasted.
//
// Decode is weight-streaming: at M=1 the kernel reads the whole quantized weight and writes almost
// nothing. The question is whether it reaches what this machine can actually sustain. Measured on an
// M2 Pro, K=2048 N=5632 (cooled between runs; see the thermal note below):
//
//	pure-read   6.5 MB   369.1 GB/s   <- CACHE, not memory: above the hardware's DRAM peak
//	pure-read  64.0 MB   181.5 GB/s
//	pure-read 256.0 MB   178.8 GB/s
//	pure-read 600.0 MB   180.1 GB/s
//	dequant     52.6 MB   183.7 GB/s
//	qmatmul M=1  6.5 MB   179.4 GB/s   (98% of dequant)
//
// The DRAM ceiling is ~180-190 GB/s and the M=1 matmul runs at ~178, i.e. 94-98% of it. There is
// nothing left to win in this kernel.
//
// TWO MEASUREMENT TRAPS this test exists to keep documented, both of which produced confident wrong
// readings before being caught:
//
//  1. A 6.5 MB read reports 369 GB/s — roughly TWICE the hardware peak — because it fits in the
//     system-level cache. Any single-tensor microbenchmark at this size measures cache, not memory,
//     and flatters every kernel that touches it. The size sweep is in the test precisely so the
//     cache-resident point is visible next to the DRAM-resident ones. Real decode streams ~599 MB
//     per token and gets no such help.
//  2. Under thermal load the SAME unchanged code measured 315 -> 445 -> 613 us on the dequant
//     control and 50.6 -> 110.3 us on the matmul, a 2.2x swing. Cross-run comparison is worthless
//     here; only min-of-repeated-runs after cooling is meaningful, and an unchanged control has to
//     be measured alongside to detect it.
//
// The M>1 rows show the other half of the picture — the cooperative kernel re-reads its weight per
// row-group, so its effective bandwidth collapses as M grows (M=2 43%, M=4 24%, M=8 13% of the
// ceiling in a clean run). That is why prefill routes through expand-then-GEMM instead, and why
// narrow projections were moved onto that path at depth.
//
// Consequence for decode: whole-token decode averages ~104 GB/s against a ~180 GB/s ceiling, but the
// gap is NOT in the weight matmuls. It is the rest of the token — attention over the KV cache,
// norms, RoPE, elementwise, sampling and the gaps between ~330 dispatches. llama.cpp sits about the
// same distance above the streaming floor (~2.3 ms/token vs our ~2.5), so that is where any decode
// win has to come from.
//
// Reported, not asserted on absolute rates; the assertion is only that the matmul stays close to the
// measured streaming reference.
func TestDecodeMatmulIsAtMemoryCeiling(t *testing.T) {
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
	qBytes := float64(len(raw))

	timeIt := func(f func(*Recorder), reps int) float64 {
		best := 1e18
		for range 9 {
			r, _ := NewRecorder()
			for range reps {
				f(r)
			}
			r.Commit()
			r.Wait()
			if d := LastGPUSeconds() / float64(reps); d < best {
				best = d
			}
			r.Free()
		}
		return best
	}
	// reference: dequant reads the quantized weight and writes f32 — a streaming pattern
	dq := timeIt(func(r *Recorder) { _ = r.DequantQ4K(rq, wf) }, 3)
	dqBytes := qBytes + float64(K*N*4)
	fmt.Printf("BW dequant      %8.1fus  moves %6.1f MB  %6.1f GB/s\n", dq*1e6, dqBytes/1e6, dqBytes/dq/1e9)

	// Sweep the read size: anything that fits in the system-level cache reports a rate above the
	// hardware's DRAM peak and is measuring cache, not memory.
	for _, mb := range []float64{6.5, 64, 256, 600} {
		b := mb * 1e6
		if rb := ProbeReadBandwidth(b, 3); rb > 0 {
			fmt.Printf("BW pure-read %6.1f MB  %8.1fus  %6.1f GB/s\n", mb, rb*1e6, b/rb/1e9)
		}
	}
	for _, M := range []int{1, 2, 4, 8} {
		x, _ := NewDeviceBufferF32(make([]float32, M*K))
		o, _ := NewDeviceBufferF32(make([]float32, M*N))
		d := timeIt(func(r *Recorder) { _ = r.QMatMulResident(x, rq, o, M) }, 3)
		frac := 100 * (qBytes / d) / (dqBytes / dq)
		fmt.Printf("BW qmatmul M=%d  %8.1fus  reads %6.1f MB  %6.1f GB/s  (%.0f%% of dequant)\n",
			M, d*1e6, qBytes/1e6, qBytes/d/1e9, frac)
		// Loose floor: M=1 must stay in the same band as the streaming reference. If this ever
		// drops far below, the decode kernel has regressed off the memory ceiling.
		if M == 1 && frac < 60 {
			t.Errorf("M=1 quantized matmul at %.0f%% of the dequant streaming rate (%.1f GB/s) — "+
				"decode is no longer bandwidth-bound; investigate before tuning anything else",
				frac, qBytes/d/1e9)
		}
		x.Release()
		o.Release()
	}
}
