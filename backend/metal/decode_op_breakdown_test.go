//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
)

// TestDecodeNonMatmulBreakdown splits the ~2.5 ms/token of decode time that is NOT weight streaming
// into op classes, each timed at decode shapes with its real per-token count.
//
// TestDecodeMatmulIsAtMemoryCeiling showed the weight matmuls run at ~178 GB/s against a ~180 GB/s
// ceiling, so decode's remaining headroom is entirely in everything else. Measured on an M2 Pro at
// ctx=512 (TinyLlama geometry: 22 layers, dim 2048, 32 heads over 4 KV heads, dk 64):
//
//	rmsnorm (1 row)   44/tok   14.96us each    658.3 us/token
//	attention sq=1    22/tok   67.65us each   1488.4 us/token
//	binary dim        44/tok    1.60us each     70.4 us/token
//	binary hidden     22/tok    1.64us each     36.1 us/token
//	silu hidden       22/tok    1.73us each     38.1 us/token
//	SUM                                        2291.3 us/token
//
// CORRECTION (see TestIsolatedOpCostIsNotAdditive): these per-op numbers are INFLATED, by ~8x for
// rmsnorm and probably similarly for the others. Timing an op as N identical dispatches into one
// command buffer serialises them on write-after-write hazards, so each pays a pipeline drain it
// never pays in a real chain. Measured against the realistic sequence, rmsnorm adds 1.62us to a
// qmatmul, not the 14.96us below.
//
// The sum landing on the ~2.5 ms budget was therefore coincidence rather than corroboration. What
// still holds is the ORDERING — attention above rmsnorm above elementwise — which is what this test
// is now good for. Every absolute figure and every share-of-token claim below is unreliable and must
// not be used to size an optimisation; measure the sequence with and without the op instead.
//
// GETTING THE SHAPES WRONG: the first run of this test reported 3858 us/token — MORE than the whole
// budget — because the buffers were sized ctx*hidden and Unary takes no length argument, so "silu
// hidden" measured 2.88M elements instead of 5632 (72us instead of 1.73us). Components that do not
// sum to the whole mean a wrong parameter, not noisy timings. The buffers here are one row for
// exactly that reason.
//
// A REJECTED FIX, recorded so it is not retried: attention scales linearly with sk and plateaus at
// ~15.5 GB/s of useful traffic (13.93us at sk=64 to 272.78us at sk=2048). With 32 query heads over 4
// KV heads, each K/V element is read by 8 threadgroups, and 15.5 x 8 = 124 GB/s looked like a kernel
// already near the ceiling but reading everything eight times. A GQA-cooperative kernel — one
// threadgroup per KV head, rep simdgroups inside it, K/V staged once in threadgroup memory — was
// implemented and measured 0.53x-0.75x, i.e. clearly WORSE:
//
//	sk= 256  off  66.86us  on  88.76us  0.75x
//	sk= 512  off  73.42us  on 129.44us  0.57x
//	sk=2048  off 280.30us  on 504.81us  0.56x
//
// Output was bit-identical (the per-lane key assignment is unchanged), and the timings differ, so
// the new kernel definitely ran — this is a real result, not a dead gate. The redundant reads were
// never DRAM traffic: one KV head's K/V is ~131 KB at sk=512, so the eight threadgroups reading it
// concurrently hit cache. Removing "redundancy" that cache already absorbed cost an 8x drop in
// parallelism (32 threadgroups to 4) for nothing. The kernel is occupancy- and latency-bound, not
// bandwidth-bound, so the 124 GB/s figure was a coincidence rather than a diagnosis.
//
// Reported, not asserted on absolute timings.
func TestDecodeNonMatmulBreakdown(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	const (
		layers = 22
		dim    = 2048
		hidden = 5632
		heads  = 32
		kvh    = 4
		dk     = 64
		ctx    = 512
		vocab  = 32000
	)
	kvDim := kvh * dk

	// min-of-N with an unchanged control measured alongside; this host swings 2x under load.
	timeIt := func(reps int, f func(*Recorder)) float64 {
		best := 1e18
		for range 11 {
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

	// Decode-sized: ONE row. Sizing these ctx*dim made Unary (which has no length argument and
	// walks the whole buffer) measure 2.88M elements instead of 5632 — 72us instead of ~1us.
	x, _ := NewDeviceBufferF32(make([]float32, dim))
	g, _ := NewDeviceBufferF32(make([]float32, dim))
	o, _ := NewDeviceBufferF32(make([]float32, dim))
	h1, _ := NewDeviceBufferF32(make([]float32, hidden))
	h2, _ := NewDeviceBufferF32(make([]float32, hidden))
	kb, _ := NewDeviceBufferF32(make([]float32, ctx*kvDim))
	vb, _ := NewDeviceBufferF32(make([]float32, ctx*kvDim))
	defer func() {
		for _, b := range []*DeviceBuffer{x, g, o, h1, h2, kb, vb} {
			b.Release()
		}
	}()

	type row struct {
		name string
		per  int // dispatches per token
		us   float64
	}
	var rows []row
	add := func(name string, per int, us float64) {
		rows = append(rows, row{name, per, us})
	}

	add("rmsnorm (1 row)", 2*layers, timeIt(20, func(r *Recorder) {
		_ = r.RMSNorm(x, g, o, 1, dim, 1e-5)
	})*1e6)
	q1, _ := NewDeviceBufferF32(make([]float32, dim))
	o1, _ := NewDeviceBufferF32(make([]float32, dim))
	defer q1.Release()
	defer o1.Release()
	add("attention sq=1", layers, timeIt(20, func(r *Recorder) {
		_ = r.MHAAt(q1, kb, vb, o1, 0, 1, ctx, dim, heads, kvh, dk, 1, 0, 0.125)
	})*1e6)
	add("binary dim", 2*layers, timeIt(20, func(r *Recorder) {
		_ = r.BinaryN(x, o, o, 0, dim)
	})*1e6)
	add("binary hidden", layers, timeIt(20, func(r *Recorder) {
		_ = r.BinaryN(h1, h2, h1, 2, hidden)
	})*1e6)
	add("silu hidden", layers, timeIt(20, func(r *Recorder) {
		_ = r.Unary(h1, h2, 6)
	})*1e6)

	fmt.Printf("DEC --- attention vs context length (sq=1) ---\n")
	for _, sk := range []int{64, 128, 256, 512, 1024, 2048} {
		kk, _ := NewDeviceBufferF32(make([]float32, sk*kvDim))
		vv, _ := NewDeviceBufferF32(make([]float32, sk*kvDim))
		d := timeIt(20, func(r *Recorder) {
			_ = r.MHAAt(q1, kk, vv, o1, 0, 1, sk, dim, heads, kvh, dk, 1, 0, 0.125)
		}) * 1e6
		mb := float64(2*sk*kvDim*4) / 1e6
		fmt.Printf("DEC   sk=%5d  %8.2fus  reads %6.2f MB  %6.1f GB/s\n", sk, d, mb, mb/d*1e3)
		kk.Release()
		vv.Release()
	}

	total := 0.0
	fmt.Printf("DEC %-18s %5s %10s %10s\n", "op", "n/tok", "us/each", "us/token")
	for _, rr := range rows {
		c := rr.us * float64(rr.per)
		total += c
		fmt.Printf("DEC %-18s %5d %10.2f %10.1f\n", rr.name, rr.per, rr.us, c)
	}
	fmt.Printf("DEC %-18s %5s %10s %10.1f us/token (non-matmul, measured pieces)\n", "SUM", "", "", total)
}
