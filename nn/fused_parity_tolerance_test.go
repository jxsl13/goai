package nn_test

import "math"

// fusedParityClose reports whether a hand-written fused inference path and the backend dispatch
// path agree to within floating-point contraction noise.
//
// THESE COMPARISONS USED TO DEMAND BIT EQUALITY, and that is not achievable on every architecture.
// Go CONTRACTS `a*b + c` into a fused multiply-add on arm64 and does not on amd64. An FMA rounds
// once where a separate multiply and add round twice, so the two paths differ by an ulp wherever
// the compiler happens to fuse. Both sides fuse — the cpu matmul kernel alone emits 202 FMADDD —
// but in DIFFERENT positions, because one is a hand-written Go loop and the other a backend
// kernel with its own blocking. Nothing in Go lets two independently written implementations
// agree on where contraction happens, so exact equality was passing on amd64 CI and failing on
// every arm64 machine.
//
// The alternative was suppressing contraction on both sides with explicit conversions, which would
// have cost throughput in the hottest code in the repository (ADR: relax the pins instead).
//
// The tolerance is RELATIVE with an absolute floor, because these outputs span orders of
// magnitude across the test configs; a fixed epsilon would be vacuous for large values and
// impossible for small ones.
//
// IT IS 1e-9, NOT AN ULP OR TWO, and the reason is that these are RECURRENCES. A per-step FMA
// difference does not stay a rounding error: the state update S = S*gate + k*v carries it forward
// and the gate amplifies it, so the divergence compounds over the sequence. Measured on the two
// worst cases at the time of writing, both at the tail of a ~2000-element output: RGLRU 2.7e-12
// relative (about 12000 ulps) and GLA 1.5e-11 (about 68000 ulps). A dot-product path such as
// DeltaNet stays near 1e-15. 1e-9 leaves roughly two orders of headroom over the worst observed
// value while remaining many orders tighter than a genuine algorithmic divergence — a wrong
// reduction order or a dropped term shows up at 1e-3 and louder, not at 1e-11.
//
// If a future change pushes one of these past 1e-9, that is a signal to investigate rather than to
// widen the bound again: the numbers above are what "only contraction" looks like here.
func fusedParityClose(a, b float64) bool {
	if a == b {
		return true
	}
	if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
		return false // a NaN or Inf on one side only is a real defect, never contraction noise
	}
	d := math.Abs(a - b)
	return d <= 1e-9*math.Max(math.Abs(a), math.Abs(b)) || d <= 1e-300
}
