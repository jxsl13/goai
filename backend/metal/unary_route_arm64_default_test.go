//go:build darwin && cgo && arm64 && !goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

func expectedMeasuredHostUnaryMaxElements(op backend.Op) int {
	switch op {
	case backend.OpSqrt:
		return maxHostUnaryBroadElements
	case backend.OpNeg:
		return maxHostUnaryNegElements
	case backend.OpReLU:
		return maxHostUnaryReLUElements
	case backend.OpAbs:
		return maxHostUnaryAbsElements
	case backend.OpExp, backend.OpLog, backend.OpTanh, backend.OpSigmoid:
		return maxHostUnaryTinyElements
	default:
		return 0
	}
}
