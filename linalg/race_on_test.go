//go:build race

package linalg_test

// raceEnabled is true under -race, and TestSVDIsBitIdentical uses it to skip.
//
// THE DIGEST IS NOT STABLE ACROSS BUILD MODES on arm64. The one-sided Jacobi rotation is
// written as c*ai - sn*aj, which the compiler is free to contract into a fused multiply-add;
// whether it does differs between a normal build and a -race build, and an FMA rounds once
// where the separate multiply and subtract round twice. The 8x8 case alone digests to
// 12638833736442736411 under -race against 3416335863526090039 without it.
//
// DO NOT "FIX" THIS BY RE-FREEZING THE DIGEST under -race: the frozen value would then be
// wrong for the normal build, which is the one that ships. And do not generalize the skip —
// eight sibling digest tests across classic, nn, autograd and backend/cpu were checked and all
// pass under -race, so a blanket skip would give away gates that work.
const raceEnabled = true
