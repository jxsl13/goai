//go:build darwin && cgo && arm64 && !goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

// Default M2 builds retain scalar transcendental CPU kernels but have typed
// Neg/Sqrt/Abs kernels and the native arm64 ReLU leaf. These operation-specific
// ceilings are the winner zones from the isolated route campaigns.
func measuredHostUnaryMaxElements(op backend.Op) int {
	switch op {
	case backend.OpNeg, backend.OpSqrt, backend.OpAbs:
		return maxHostUnaryBroadElements
	case backend.OpReLU:
		return maxHostUnaryReLUElements
	case backend.OpExp, backend.OpLog, backend.OpTanh, backend.OpSigmoid:
		return maxHostUnaryTinyElements
	default:
		return 0
	}
}
