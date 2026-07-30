package parallel

import (
	"fmt"
	"testing"
	"time"
)

// The dispatch-cost benchmarks this package never had.
//
// Every parallel threshold in this repository — and there are more than ten, most of them
// spelled 1<<15 — is a claim about what one fan-out costs. Two of those sites describe the
// constant as measured, and no artifact of that measurement survives anywhere in the tree.
// These benchmarks are that artifact: they compare the SAME body run serially and through
// Rows across the sizes the thresholds sit between, so the crossover can be read off rather
// than assumed.
//
// The body is deliberately the cheapest realistic one — a single multiply-add per element
// over one contiguous slice. That is the shape the thresholds guard (elementwise sweeps and
// row loops), and making the body cheap is what puts the fixed cost in view: a body heavy
// enough to dominate would show a crossover that says more about the body than about the
// pool.
//
// WARM AND COLD ARE SEPARATE BENCHMARKS ON PURPOSE. A worker that is still in the scheduler's
// spin phase is handed its task without an OS wake; one that has descended into a futex wait
// costs a full wake to restart. Back-to-back dispatches measure the first, and a dispatch
// after an idle gap measures the second. A single number that averages the two describes
// neither, and the thresholds are applied in both regimes.

func benchBody(dst []float64, lo, hi int) {
	for i := lo; i < hi; i++ {
		dst[i] = dst[i]*1.0000001 + 1
	}
}

var dispatchSizes = []int{1 << 10, 1 << 12, 1 << 14, 1 << 15, 1 << 16, 1 << 17, 1 << 18, 1 << 20}

// BenchmarkDispatchSerial is the baseline: the identical body, no fan-out.
func BenchmarkDispatchSerial(b *testing.B) {
	for _, n := range dispatchSizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			dst := make([]float64, n)
			b.ResetTimer()
			for range b.N {
				benchBody(dst, 0, n)
			}
		})
	}
}

// BenchmarkDispatchWarm fans out back to back, so workers stay hot between dispatches. This
// is the favorable regime, and the crossover it reports is the LOWEST defensible threshold.
func BenchmarkDispatchWarm(b *testing.B) {
	for _, n := range dispatchSizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			dst := make([]float64, n)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				Rows(n, func(lo, hi int) { benchBody(dst, lo, hi) })
			}
		})
	}
}

// BenchmarkDispatchCold puts an idle gap before each fan-out so workers have parked. The gap
// is charged to the benchmark, so the ns/op here is NOT comparable to the serial baseline —
// only the warm-versus-cold DIFFERENCE at the same n is meaningful, and that difference is
// the wake cost the threshold has to clear.
func BenchmarkDispatchCold(b *testing.B) {
	for _, n := range []int{1 << 12, 1 << 15, 1 << 18} {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			dst := make([]float64, n)
			b.ResetTimer()
			for range b.N {
				b.StopTimer()
				time.Sleep(200 * time.Microsecond) // long enough for workers to park
				b.StartTimer()
				Rows(n, func(lo, hi int) { benchBody(dst, lo, hi) })
			}
		})
	}
}

// BenchmarkDispatchRowsIdx reports ALLOCATIONS for the two variants. Read it for allocs/op
// and B/op only.
//
// Do NOT read a ns/op ratio between these two sub-benchmarks. They ran back to back over a
// shared slice in an earlier version and the second arm measured 18% slower than the first,
// which read exactly like a dispatch regression introduced by carrying the chunk index in the
// task struct. BenchmarkDispatchRowsIdxScale, which gives each size its own arms, puts the two
// within 1.00x at every size from 2^14 to 2^20 — the 18% was the second arm inheriting a
// different cache and scheduler state, not a property of the code. Each arm now owns its own
// slice, which removes the sharing but not the ordering, so the caution stands.
func BenchmarkDispatchRowsIdx(b *testing.B) {
	const n = 1 << 16
	b.Run("Rows", func(b *testing.B) {
		dst := make([]float64, n)
		b.ReportAllocs()
		for b.Loop() {
			Rows(n, func(lo, hi int) { benchBody(dst, lo, hi) })
		}
	})
	b.Run("RowsIdx", func(b *testing.B) {
		dst := make([]float64, n)
		b.ReportAllocs()
		for b.Loop() {
			RowsIdx(n, func(_, lo, hi int) { benchBody(dst, lo, hi) })
		}
	})
}

// BenchmarkDispatchRowsIdxScale separates fixed dispatch cost from per-chunk work: if a change
// to the task plumbing costs a constant, the gap between Rows and RowsIdx shrinks in relative
// terms as n grows. A gap that holds its RATIO instead would mean the change touched the body
// path, not the dispatch path.
func BenchmarkDispatchRowsIdxScale(b *testing.B) {
	for _, n := range []int{1 << 14, 1 << 16, 1 << 18, 1 << 20} {
		dst := make([]float64, n)
		b.Run(fmt.Sprint(n)+"/Rows", func(b *testing.B) {
			for b.Loop() {
				Rows(n, func(lo, hi int) { benchBody(dst, lo, hi) })
			}
		})
		b.Run(fmt.Sprint(n)+"/RowsIdx", func(b *testing.B) {
			for b.Loop() {
				RowsIdx(n, func(_, lo, hi int) { benchBody(dst, lo, hi) })
			}
		})
	}
}
