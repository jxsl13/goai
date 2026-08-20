//go:build darwin && cgo && arm64 && goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

// Typed NEON Exp/Log/Tanh/Sigmoid kernels extend their measured CPU winner
// zone to the largest frozen shape. ReLU retains its independent crossover.
func measuredHostUnaryMaxElements(op backend.Op) int {
	switch op {
	case backend.OpNeg, backend.OpExp, backend.OpLog, backend.OpTanh,
		backend.OpSigmoid, backend.OpSqrt, backend.OpAbs:
		return maxHostUnaryBroadElements
	case backend.OpReLU:
		return maxHostUnaryReLUElements
	default:
		return 0
	}
}
