//go:build darwin && cgo

package metal

// TestMatMulSquareShapesVsTorch records goai's Metal matmul at the square shapes
// internal/benchcompare/python_compare.py times torch on, so the two can be read together.
//
// It exists because the recorded claim — matmul "~3.5x behind torch-mps" — is badly stale and has
// the sign wrong. Measured back-to-back in one session (torch 2.13.0, numpy 2.5.2, M2 Pro):
//
//	size   goai-metal (MPS f32)   torch-mps   torch-cpu
//	 256          655.4             525.9      1022.4
//	 512         2515.1             988.3      1330.1
//	1024         4955.5            3101.2      2171.4
//
// goai is AHEAD of torch-mps at all three sizes (1.25x, 2.54x, 1.60x).
//
// The torch-cpu column is NOT goai's CPU story — comparing our Metal path against torch's CPU path
// answers no useful question. goai's own CPU backend, measured at the same shapes, gives 973.3 /
// 1823.2 / 2133.9 GFLOP/s against torch-cpu's 1022.4 / 1330.1 / 2171.4: 0.95x, 1.37x, 0.98x, i.e.
// parity. But ONLY with GOEXPERIMENT=simd — without it the same backend measures 91 / 111 / 126
// GFLOP/s, 11-18x slower, because backend/cpu's Accelerate cblas_sgemm path is gated on
// goexperiment.simd (not on a -tags simd build tag, which silently does nothing here).
//
// That gate is DELIBERATE and should not be "fixed". backend/cpu has three tiers:
//
//	default build          plain Go        ~91-126 GFLOP/s   BIT-EXACT
//	goexperiment.simd      NEON kernel     ~795 (documented)  f32-native, ADR-0021 tolerance
//	  + cgo                Accelerate AMX  ~973-2134 measured f32-native, ADR-0021 tolerance
//
// The fast tiers accumulate f32-native (vendor SGEMM convention), so they are covered by a
// tolerance contract rather than bit-exactness, and gemm_accel_darwin.go states plainly that the
// default build "never sees this file and stays bit-exact". The 11-18x is the price of that
// guarantee, not an oversight — relaxing the constraint would silently change the numerics every
// default build currently promises.
//
// Caveat on what this compares: goai dispatches MPSMatrixMultiplication directly, while torch-mps
// goes through MPSGraph and its own kernels. So this is not "our GEMM beats Apple's" — it is that
// the direct MPS path beats torch's MPS path at these shapes, which is the comparison a user
// choosing between the two libraries actually faces.
//
// torch's own numbers move noticeably between runs on this machine (torch-mps at 1024 read 3641 and
// 3101 in two consecutive runs), so only the same-session pairing above is meaningful.
import (
	"fmt"
	"testing"
)

func TestMatMulSquareShapesVsTorch(t *testing.T) {
	if !Available() {
		t.Skip("no metal")
	}
	for _, n := range []int{256, 512, 1024} {
		d := ProbeGEMMDtype(n, n, n, false, 20)
		if d < 0 {
			t.Fatal("probe failed")
		}
		fmt.Printf("SQ %d  goai-metal(MPS f32) %8.1f GFLOP/s\n", n, float64(2*n*n*n)/d/1e9)
	}
}
