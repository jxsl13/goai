// Package fmath provides Min and Max with math.Min and math.Max semantics, at the cost of
// the min and max builtins.
//
// THE TWO ARE NOT THE SAME FUNCTION, which is why this package exists rather than a
// tree-wide substitution. math.Max documents +Inf as beating NaN and math.Min documents -Inf
// as beating NaN; the builtins propagate NaN unconditionally, as the language spec requires.
// Over every ordered pair drawn from {NaN, ±Inf, ±0, ±1, ±MaxFloat64, ±SmallestNonzero} the
// two formulations disagree on exactly four: Max(NaN, +Inf), Max(+Inf, NaN), Min(NaN, -Inf)
// and Min(-Inf, NaN). fmath_test.go pins all four.
//
// The price of the difference is a function CALL. On arm64 math.Min compiles to
// `CALL math.archMin` inside a 48-byte frame with a stack-growth check, while the builtin
// compiles to a single FMIND instruction in a leaf with no frame at all. In a loop that is
// the difference between one instruction and a non-inlinable call per element.
//
// The recovery is that the two can only ever disagree ON A NaN RESULT: whenever they differ,
// the builtin is the one returning NaN. So take the instruction, and consult math only when
// the instruction says NaN. The branch is never taken on ordinary data and predicts
// perfectly; measured on the reference PPO surrogate at batch 4096 the guarded form runs at
// 40.5 us against 98.7 for math.Min/math.Max, and against 63.6 for the comparison chain that
// PS3077 recommends when a clamp is the only thing in the way.
package fmath

import "math"

// Min returns math.Min(x, y), bit for bit, including the NaN and signed-zero contract.
func Min(x, y float64) float64 {
	m := min(x, y)
	if m != m {
		// The builtin says NaN. That is the only outcome on which it and math.Min can
		// disagree — math.Min(NaN, -Inf) is -Inf — so this is the one case worth the call.
		return math.Min(x, y)
	}
	return m
}

// Max returns math.Max(x, y), bit for bit, including the NaN and signed-zero contract.
func Max(x, y float64) float64 {
	m := max(x, y)
	if m != m {
		return math.Max(x, y)
	}
	return m
}

// A Clamp(v, lo, hi) wrapper was written and removed: composing the two bodies puts it over
// the inliner's cost budget, so it becomes exactly the per-element call this package exists
// to avoid. Callers write Max(lo, Min(hi, v)), which mirrors the math.Max/math.Min text it
// replaces line for line and inlines.
