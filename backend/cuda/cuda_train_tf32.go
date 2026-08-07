//go:build cuda && cgo && (linux || windows)

package cuda

/*
#include "cuda_bridge.h"
*/
import "C"

import "fmt"

// SetGemmTF32 enables or disables TF32 tensor-core math for the shared cuBLAS handle. When enabled, the
// f32 device GEMMs (MatMul / MatMulGradW / MatMulGradX) run on Ampere+ tensor cores at much higher
// throughput with TF32 (~10-bit mantissa) — the precision PyTorch uses by default for training on
// Ampere, in-policy for the incumbent training tolerance. It is persistent until toggled: enable it
// around a GPU training loop and leave it off for inference paths that need exact f32.
func SetGemmTF32(enable bool) error {
	e := C.int(0)
	if enable {
		e = 1
	}
	if rc := C.cu_set_gemm_tf32(e); rc != 0 {
		return fmt.Errorf("cuda: SetGemmTF32 rc=%d", int(rc))
	}
	return nil
}
