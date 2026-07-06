package nn

import (
	"math"
	"math/rand/v2"

	"github.com/jxsl13/goai/tensor"
)

// Parameter initializers (§T15). Deterministic: callers pass an explicit seed
// (§V13 reproducibility). Formulas follow the original papers.

// XavierUniform fills t with U(-a, a), a = √(6/(fanIn+fanOut)) — Glorot &
// Bengio (2010), "Understanding the difficulty of training deep feedforward
// neural networks". Keeps forward/backward variance constant for tanh-like
// activations.
func XavierUniform(t *tensor.Tensor, fanIn, fanOut int, seed uint64) {
	a := math.Sqrt(6.0 / float64(fanIn+fanOut))
	fillUniform(t, -a, a, seed)
}

// KaimingNormal fills t with N(0, √(2/fanIn)) — He et al. (2015), "Delving Deep
// into Rectifiers". The variance-preserving choice for ReLU activations.
func KaimingNormal(t *tensor.Tensor, fanIn int, seed uint64) {
	std := math.Sqrt(2.0 / float64(fanIn))
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	for i := range t.Numel() {
		t.SetF64(rng.NormFloat64()*std, tensor.Unravel(i, t.Shape())...)
	}
}

// KaimingUniform fills t with U(-a, a), a = √(6/fanIn) (He et al. 2015, uniform
// variant).
func KaimingUniform(t *tensor.Tensor, fanIn int, seed uint64) {
	a := math.Sqrt(6.0 / float64(fanIn))
	fillUniform(t, -a, a, seed)
}

// Zeros fills t with 0 (the conventional bias init).
func Zeros(t *tensor.Tensor) {
	for i := range t.Numel() {
		t.SetF64(0, tensor.Unravel(i, t.Shape())...)
	}
}

func fillUniform(t *tensor.Tensor, lo, hi float64, seed uint64) {
	rng := rand.New(rand.NewPCG(seed, 0x6b79a2c3d4e5f601))
	for i := range t.Numel() {
		t.SetF64(lo+rng.Float64()*(hi-lo), tensor.Unravel(i, t.Shape())...)
	}
}
