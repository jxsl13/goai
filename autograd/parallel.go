package autograd

import (
	"runtime"
	"sync"
)

// parallelBands runs body over disjoint bands of the range [0,d) in GOMAXPROCS chunks, and
// runs it serially in one call below a work threshold — a fan-out is only worth its
// scheduling cost once there is enough work to amortize it.
//
// The CALLER owns disjointness. The helper guarantees nothing about what body touches; it
// only guarantees that the bands it passes do not overlap. Two callers rely on that in
// different ways:
//
//   - The conv1d VJP bands CHANNELS. Every write it makes (dxs[..*D+c], dws[c*K+..], dbs[c])
//     is indexed by the channel, so channels touch disjoint memory, and keeping t-outer and
//     k-inner within each channel preserves the per-element ascending-t accumulation, which
//     is what makes the split bit-identical rather than merely race-free.
//   - The transpose VJP bands DESTINATION ROWS, which is disjoint for the trivial reason
//     that nothing is accumulated at all — see vjp_transpose.go.
//
// workPerBand is the work ONE band index carries, not the total; the gate is the product.
// Do not read the threshold as a tuning knob to lower freely: a band whose body re-streams
// a shared input pays that traffic once per band, and a finer grain has measured 126%
// SLOWER on the conv1d body for exactly that reason.
func parallelBands(d, workPerBand int, body func(lo, hi int)) {
	nw := runtime.GOMAXPROCS(0)
	if nw > d {
		nw = d
	}
	if nw <= 1 || d*workPerBand < 1<<15 {
		body(0, d)
		return
	}
	var wg sync.WaitGroup
	chunk := (d + nw - 1) / nw
	for lo := 0; lo < d; lo += chunk {
		hi := lo + chunk
		if hi > d {
			hi = d
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			body(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}
