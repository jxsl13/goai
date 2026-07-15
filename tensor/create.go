package tensor

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Array-creation constructors in the style of numpy (np.zeros/ones/full/arange/linspace/eye/…) and
// gonum, for ergonomic tensor building. They are dtype-explicit like New; the reference for their
// semantics is numpy (definitional, no paper). All return a fresh contiguous CPU tensor.

// fill writes gen(i) into every element of a fresh contiguous tensor t (row-major flat index i) and
// returns t. Fast paths for f32/f64/f16/bf16; other dtypes go through the dtype-converting setter.
// t is always freshly created here (contiguous, offset 0), so the row-major flat index IS the
// storage index — no Unravel needed (§base-perf: the old per-element Unravel heap-allocated).
func fill(t *Tensor, gen func(i int) float64) *Tensor {
	switch t.Dtype() {
	case F64:
		s := t.storage.F64()
		for i := range s {
			s[i] = gen(i)
		}
	case F32:
		s := t.storage.F32()
		for i := range s {
			s[i] = float32(gen(i))
		}
	case F16:
		s := t.storage.U16()
		for i := range s {
			s[i] = f32ToF16(float32(gen(i)))
		}
	case BF16:
		s := t.storage.U16()
		for i := range s {
			s[i] = f32ToBF16(float32(gen(i)))
		}
	default:
		for i := range t.Numel() {
			t.storage.setF64(i, gen(i))
		}
	}
	return t
}

// Zeros returns a tensor of shape filled with 0 — the numpy np.zeros. (New already zeroes, so this is
// a numpy-familiar alias.)
func Zeros(dtype Dtype, shape Shape) *Tensor { return New(dtype, shape) }

// Ones returns a tensor of shape filled with 1 (np.ones).
func Ones(dtype Dtype, shape Shape) *Tensor {
	return fill(New(dtype, shape), func(int) float64 { return 1 })
}

// Full returns a tensor of shape filled with val (np.full).
func Full(dtype Dtype, shape Shape, val float64) *Tensor {
	return fill(New(dtype, shape), func(int) float64 { return val })
}

// ZerosLike returns a zero tensor with the same shape and dtype as t (np.zeros_like).
func ZerosLike(t *Tensor) *Tensor { return New(t.Dtype(), t.shape.Clone()) }

// OnesLike returns a ones tensor with the same shape and dtype as t (np.ones_like).
func OnesLike(t *Tensor) *Tensor { return Ones(t.Dtype(), t.shape.Clone()) }

// Arange returns the 1-D tensor [start, start+step, …) up to but excluding stop (np.arange): the
// length is max(0, ⌈(stop−start)/step⌉). step must be non-zero.
func Arange(dtype Dtype, start, stop, step float64) *Tensor {
	if step == 0 {
		panic("tensor: Arange step must be non-zero")
	}
	n := max(int(math.Ceil((stop-start)/step)), 0)
	return fill(New(dtype, Shape{n}), func(i int) float64 { return start + float64(i)*step })
}

// Linspace returns n evenly spaced samples over the closed interval [start, stop] (np.linspace,
// endpoint=True): step = (stop−start)/(n−1) for n>1; n=1 yields [start]. n must be ≥ 1.
func Linspace(dtype Dtype, start, stop float64, n int) *Tensor {
	if n < 1 {
		panic(fmt.Sprintf("tensor: Linspace n=%d must be ≥ 1", n))
	}
	if n == 1 {
		return fill(New(dtype, Shape{1}), func(int) float64 { return start })
	}
	step := (stop - start) / float64(n-1)
	return fill(New(dtype, Shape{n}), func(i int) float64 {
		if i == n-1 {
			return stop // exact endpoint, no rounding drift
		}
		return start + float64(i)*step
	})
}

// Eye returns the n×n identity matrix (np.eye / np.identity): 1 on the main diagonal, 0 elsewhere.
func Eye(dtype Dtype, n int) *Tensor {
	if n < 0 {
		panic(fmt.Sprintf("tensor: Eye n=%d must be ≥ 0", n))
	}
	t := New(dtype, Shape{n, n}) // already zeroed: only the diagonal needs writes (§base-perf:
	// O(n) sets instead of n² generator calls with two integer divisions each).
	switch dtype {
	case F64:
		s := t.storage.F64()
		for i := 0; i < n; i++ {
			s[i*(n+1)] = 1
		}
	case F32:
		s := t.storage.F32()
		for i := 0; i < n; i++ {
			s[i*(n+1)] = 1
		}
	default:
		for i := 0; i < n; i++ {
			t.storage.setF64(i*(n+1), 1)
		}
	}
	return t
}

// Identity is an alias for Eye (np.identity), the n×n identity matrix.
func Identity(dtype Dtype, n int) *Tensor { return Eye(dtype, n) }

// Rand returns a tensor of shape filled with i.i.d. uniform samples in [0,1) (np.random.rand),
// seeded deterministically (§V-reproducibility).
func Rand(dtype Dtype, seed uint64, shape Shape) *Tensor {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	return fill(New(dtype, shape), func(int) float64 { return rng.Float64() })
}

// Randn returns a tensor of shape filled with i.i.d. standard-normal samples (mean 0, variance 1;
// np.random.randn), seeded deterministically.
func Randn(dtype Dtype, seed uint64, shape Shape) *Tensor {
	rng := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	return fill(New(dtype, shape), func(int) float64 { return rng.NormFloat64() })
}
