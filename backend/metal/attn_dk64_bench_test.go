//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
	"time"
)

// TestAttnDK64Specializations reports the two non-decode attention kernels that carried the same
// defect as mha_decode_f32 (see TestMHADecodeCost): per-thread q[128]/acc[128] walked with a loop
// bound taken from a KERNEL ARGUMENT. A runtime trip count makes those arrays dynamically indexed,
// so they cannot live in registers and every element touch becomes a memory access.
//
// Measured, 3 alternations each, near-zero variance:
//
//	flashattn_f32  seq=128  2612 -> 481us  (5.4x)   seq=384  8061 -> 1716us  (4.7x)
//	mha_f32        seq=128  3937 -> 620us  (6.4x)   seq=384 17803 -> 3887us  (4.6x)
//	                        window seq=512 25727 -> 5553us (4.6x)
//
// A caveat worth carrying: backend.Execute(OpMHA) does NOT show this. That path uploads Q,K,V and
// downloads O per call (~1.2 MB at seq=384), and the transfer dominates so thoroughly that the A/B
// there came out bimodal and uncorrelated with the variant. The win is real and lands for callers
// that keep tensors device-resident (the Recorder path); the host-slice op API is transfer-bound
// and a kernel change cannot show up in it. Measuring the kernel through that API would have
// reported "no effect" — the same regime error as timing a cache-resident weight and concluding a
// decode kernel is instruction-bound.
//
// Reported, not asserted, apart from a loose ceiling that catches a regression back onto the
// dynamically-indexed kernel without pinning machine-dependent absolute timings.
func TestAttnDK64Specializations(t *testing.T) {
	// Thresholds here are calibrated on the M2 this project targets. GitHub's macOS runners share a
	// much weaker GPU, where these numbers do not merely drift — they invert: the crossover guard
	// reports mmunit AHEAD at M=32/48/64 there, the opposite of every local reading. A timing
	// assertion that flips sign on other hardware is a dev-box tool, which is exactly the split
	// ci.yml already documents ("-short on runners, full suites are for dev boxes").
	if testing.Short() {
		t.Skip("timing guard calibrated for this project's M2; runner GPUs invert it")
	}
	if !Available() {
		t.Skip("no metal")
	}
	const heads, dk = 12, 64
	dm := heads * dk
	for _, seq := range []int{128, 384} {
		q, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		k, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		v, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		o, _ := NewDeviceBufferF32(make([]float32, seq*dm))
		slope := func(name string, rec func(r *Recorder) error, ceiling float64) {
			meas := func(n int) float64 {
				best := 1e18
				for range 15 {
					r, err := NewRecorder()
					if err != nil {
						t.Fatal(err)
					}
					for range n {
						if err := rec(r); err != nil {
							t.Fatal(err)
						}
					}
					st := time.Now()
					r.Commit()
					r.Wait()
					if d := time.Since(st).Seconds(); d < best {
						best = d
					}
					r.Free()
				}
				return best
			}
			lo, hi := meas(4), meas(32)
			us := (hi - lo) / (32 - 4) * 1e6
			fmt.Printf("ATTN %-10s seq=%3d  %.1f us/op\n", name, seq, us)
			if us > ceiling {
				t.Errorf("%s seq=%d: %.1f us/op > %.0f — attention has regressed onto the dynamically-indexed kernel", name, seq, us, ceiling)
			}
		}
		// Pre-fix costs were 2612/8061 (flash) and 3937/17803 (two-pass); the ceilings sit well
		// above the fixed numbers and well below the broken ones.
		ceilFlash, ceilTwoPass := 1200.0, 1600.0
		if seq == 384 {
			ceilFlash, ceilTwoPass = 4000.0, 9000.0
		}
		slope("flash", func(r *Recorder) error {
			return r.flashattn(q, k, v, o, seq, dm, heads, dk, 0, heads, 0.125)
		}, ceilFlash)
		slope("two-pass", func(r *Recorder) error {
			return r.MHA(q, k, v, o, seq, seq, dm, heads, heads, dk, 0, 0, 0.125)
		}, ceilTwoPass)
		q.Release()
		k.Release()
		v.Release()
		o.Release()
	}
}
