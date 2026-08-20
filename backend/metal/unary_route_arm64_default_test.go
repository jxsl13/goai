//go:build darwin && cgo && arm64 && !goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

func expectedMeasuredHostUnaryMaxElements(op backend.Op) int {
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
