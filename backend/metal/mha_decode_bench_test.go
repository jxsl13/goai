//go:build darwin && cgo

package metal

import (
	"fmt"
	"testing"
	"time"
)

// TestMHADecodeCost measures ONE decode attention (sq=1) against a KV cache of length sk, at the
// head geometries real models use. It exists because this call had an sk-INDEPENDENT floor of ~99us
// (dk=64) / ~239us (dk=128) — a single query row against eight keys cost 99us, which no amount of
// attention work explains.
//
// The cause was a runtime loop bound. mha_decode_f32 declares per-thread q[128] and acc[128] and
// walks them with `for (d=0; d<dk; d++)` where dk is a KERNEL ARGUMENT. A runtime trip count makes
// the arrays dynamically indexed, so they cannot live in registers and spill to memory — every
// acc[d] touch in the streaming loop AND in the 5-level simdgroup merge becomes a memory access.
// The merge walks all dk accumulators at every level regardless of sk, which is precisely the
// sk-independent part of the cost.
//
// Note the failed hypothesis, because it looks identical from the outside: halving the arrays to
// q[64]/acc[64] for dk<=64 changed NOTHING (99.5 vs 99.1us, three alternations). Array SIZE was
// never the problem; dynamic indexing was. Compiling dk as a constant fixes it, and the win is
// large at every context length:
//
//	dk=64   sk=8    99.8 -> 11.5us (8.7x)   sk=1024  766.5 -> 129.6us (5.9x)
//	dk=128  sk=36  238.6 -> 64.9us (3.7x)   sk=512   891.0 -> 277.2us (3.2x)
//
// End-to-end on a 22-layer TinyLlama-shaped Q4_K decode: 44.3 -> 49.1 tok/s at a short context, and
// the gap widens with context since attention is the term that grows in sk.
//
// Reported, not asserted, apart from a loose ceiling: absolute microseconds are machine- and
// thermal-dependent, and a tight threshold would be flaky rather than a guard.
func TestMHADecodeCost(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, g := range []struct {
		name                   string
		dm, heads, kvHeads, dk int
	}{
		{"tinyllama", 2048, 32, 4, 64},
		{"llama7b", 4096, 32, 32, 128},
	} {
		const ctx = 1024
		q, _ := NewDeviceBufferF32(make([]float32, g.dm))
		o, _ := NewDeviceBufferF32(make([]float32, g.dm))
		k, _ := NewDeviceBufferF32(make([]float32, ctx*g.kvHeads*g.dk))
		v, _ := NewDeviceBufferF32(make([]float32, ctx*g.kvHeads*g.dk))
		for _, sk := range []int{36, 512} {
			meas := func(n int) float64 {
				best := 1e18
				for range 25 {
					r, err := NewRecorder()
					if err != nil {
						t.Fatal(err)
					}
					for range n {
						if err := r.MHA(q, k, v, o, 1, sk, g.dm, g.heads, g.kvHeads, g.dk, 1, 0, 0.125); err != nil {
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
			lo, hi := meas(32), meas(256)
			us := (hi - lo) / (256 - 32) * 1e6
			fmt.Printf("MHA decode %-10s dk=%3d sk=%4d  %.2f us/op\n", g.name, g.dk, sk, us)
			// The pre-fix cost was 99-891us across these points. A loose ceiling catches a
			// regression back onto the runtime-dk kernel without pinning absolute timings.
			if us > 600 {
				t.Errorf("%s dk=%d sk=%d: %.1f us/op — decode attention has regressed onto the dynamically-indexed path", g.name, g.dk, sk, us)
			}
		}
		q.Release()
		o.Release()
		k.Release()
		v.Release()
	}
}
