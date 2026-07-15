//go:build darwin && cgo && arm64 && goexperiment.simd

package cpu

// Apple Accelerate cblas_sgemm fast path (ADR-0027, Path A). On Apple silicon
// Accelerate's BLAS runs on the AMX matrix coprocessor (~512 f32 FLOPs/cycle
// per block; the same door torch-cpu/numpy use), which pure NEON cannot
// reach: ~2650 GFLOP/s @1024³ vs ~795 for the ADR-0026 NEON kernel and 2584
// for torch-cpu. The binding itself lives in internal/accel (a cgo package
// cannot also carry the Plan9 kernels); this file just installs the hook.
// With CGO_ENABLED=0 both files drop out, gemmF32Accel stays nil, and the
// pure-Go NEON/raw-AMX paths serve everything — fully functional.
//
// Numerics: Accelerate accumulates f32-native (vendor SGEMM convention), the
// same contract as the NEON kernel → covered by the ADR-0021 tolerance policy
// and the gemmF32Tolerant parity tests (rtol 2e-3). The default (non-simd)
// build never sees this file and stays bit-exact. F64 is untouched.
//
// Accelerate self-threads across the AMX blocks, so it is called ONCE for the
// whole matrix — never from inside parallelWork.

import "github.com/jxsl13/goai/backend/cpu/internal/accel"

func init() {
	if accel.SGEMM != nil {
		gemmF32Accel = func(a, b, c []float32, m, k, n int) bool {
			accel.SGEMM(a, b, c, m, k, n)
			return true
		}
	}
}
