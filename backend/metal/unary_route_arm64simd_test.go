//go:build darwin && cgo && arm64 && goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

func expectedMeasuredHostUnaryMaxElements(op backend.Op) int {
	switch op {
	case backend.OpExp, backend.OpLog, backend.OpTanh, backend.OpSigmoid, backend.OpSqrt:
		return maxHostUnaryBroadElements
	case backend.OpNeg:
		return maxHostUnaryNegElements
	case backend.OpReLU:
		return maxHostUnaryReLUElements
	case backend.OpAbs:
		return maxHostUnaryAbsElements
	default:
		return 0
	}
}
