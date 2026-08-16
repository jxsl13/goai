// Package archgold selects a frozen bit-exactness golden for the architecture a test runs on.
//
// The repo's *BitIdentical tests freeze an FNV digest of a kernel's output bits. That is the right
// gate — the claim they defend is that an optimization reorders nothing, and a tolerance test would
// pass a reordering that changes the last ulp. But a single frozen constant CANNOT be portable,
// for two independent reasons, both measured on 2026-08-16 rather than inferred:
//
//  1. Go's math.Sin/math.Cos are not bit-identical across GOARCH. Sweeping the inputs these tests
//     build their fixtures from, 41 of 2048 values differ by one ulp (math.Cos(84) ends e523 on
//     arm64 and e522 on amd64). Where a fixture is built from a transcendental, the INPUTS already
//     differ and everything downstream follows. Spot-checking individual values is not enough to
//     rule this out — Sin(129) and Cos(451) agree exactly, and the sweep still diverges.
//
//  2. Floating-point contraction differs. arm64 fuses `v -= a*b` into FNMSUB; amd64 at the default
//     GOAMD64=v1 does not, and one rounding step per term is enough to change the digest. The
//     clincher: with exact dyadic fixtures, the only shape whose digest matched across arches was
//     the one small enough that the unrolled loop never ran.
//
// Cause 2 cannot be fixed by choosing better fixtures, so the goldens are per-architecture. Each
// arch still gets a full bit-exact assertion — nothing is weakened to a tolerance — and a
// reordering on either one is still caught.
//
// The amd64 values are trustworthy from an Apple-silicon dev box: `GOARCH=amd64 go test` runs under
// Rosetta 2 and reproduces CI exactly. TestQRVJPIsBitIdentical reports the same digests under
// Rosetta as on GitHub's ubuntu-latest and windows-latest runners, so one golden per arch class is
// sufficient and can be regenerated locally.
package archgold

import "runtime"

// Reason explains a skip on an architecture with no recorded goldens. Pass it to t.Skip.
const Reason = "no bit-exactness goldens recorded for GOARCH=" + runtime.GOARCH +
	" (digests are architecture-specific: FP contraction and math.Sin/Cos both differ); " +
	"record them with: GOARCH=<arch> go test -run <TestName> ./<pkg>/"

// Supported reports whether goldens exist for the current architecture. Tests that freeze digests
// must skip when it is false — asserting another architecture's constants would fail for a reason
// that says nothing about the code under test.
func Supported() bool { return runtime.GOARCH == "arm64" || runtime.GOARCH == "amd64" }

// Pick returns the golden for the current architecture. On an architecture without goldens it
// returns the arm64 value, which is harmless because Supported gates the assertion; keeping it
// total means a table literal stays a plain expression.
func Pick(arm64, amd64 uint64) uint64 {
	if runtime.GOARCH == "amd64" {
		return amd64
	}
	return arm64
}
