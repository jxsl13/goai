//go:build darwin && cgo && arm64 && goexperiment.simd

package metal

import "github.com/jxsl13/goai/backend"

// Typed NEON Exp/Log/Tanh/Sigmoid/ReLU kernels extend their measured CPU winner
// zones to the largest frozen shape. Each operation retains its own selector
// constant so later kernel changes can invalidate one crossover independently.
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
