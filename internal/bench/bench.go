// Package bench holds deterministic data generators for benchmarks (§T10, §V5).
// Seeds are explicit so benchmark inputs are reproducible across runs and
// machines, which keeps baseline numbers comparable for the cgo gate (§C3).
package bench

import (
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// RandF64 returns a contiguous F64 tensor of the given shape filled with
// deterministic pseudo-random values in [-1, 1).
func RandF64(shape tensor.Shape, seed uint64) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	data := make([]float64, shape.Numel())
	for i := range data {
		data[i] = rng.Float64()*2 - 1
	}
	return tensor.FromFloat64(shape, data)
}

// RandF32 returns a contiguous F32 tensor of the given shape filled with
// deterministic pseudo-random values in [-1, 1).
func RandF32(shape tensor.Shape, seed uint64) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	data := make([]float32, shape.Numel())
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	return tensor.FromFloat32(shape, data)
}
