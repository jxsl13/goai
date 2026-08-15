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
// goai is AHEAD of torch-mps at all three sizes (1.25x, 2.54x, 1.60x) and ahead of torch-cpu at 512
// and 1024 (1.89x, 2.28x). It trails torch-cpu at 256, where the matrices are small enough that
// Accelerate's threaded CPU path beats a GPU dispatch.
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
