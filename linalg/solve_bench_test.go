package linalg_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// benchMat builds a diagonally dominant [n,n] matrix — nonsingular for LU/QR and, being
// symmetric with a dominant diagonal, positive definite for Cholesky. A fixed seed keeps
// the arms of an A/B comparison on identical input.
func benchMat(n int, sym bool) *tensor.Tensor {
	rng := rand.New(rand.NewSource(7))
	d := make([]float64, n*n)
	for i := range n {
		for j := range n {
			if sym && j < i {
				d[i*n+j] = d[j*n+i]
				continue
			}
			d[i*n+j] = rng.NormFloat64()
		}
	}
	for i := range n {
		d[i*n+i] += float64(n) + 1 // dominance: |a_ii| > sum of the row's off-diagonals
	}
	return tensor.FromFloat64(tensor.Shape{n, n}, d)
}

var benchSizes = []int{64, 256, 512, 768}

// BenchmarkInverse is the GPTQ/SparseGPT-shaped call — full Factor plus Solve against the
// identity, which nn/gptq.go and nn/sparsegpt.go run once per quantized linear layer at
// widths of 768 to 4096. This is the end-to-end ratio a caller sees; BenchmarkLUSolve_*
// isolates the substitution phase by hoisting Factor out.
func BenchmarkInverse(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			a := benchMat(n, false)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				f, err := linalg.Factor(a)
				if err != nil {
					b.Fatal(err)
				}
				if _, err := f.Inverse(); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkCholSolveMat(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			a, rhs := benchMat(n, true), benchMat(n, false)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := linalg.CholSolve(a, rhs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkLstsqMat(b *testing.B) {
	for _, n := range benchSizes {
		b.Run(fmt.Sprint(n), func(b *testing.B) {
			a, rhs := benchMat(n, false), benchMat(n, false)
			b.ReportAllocs()
			b.ResetTimer()
			for range b.N {
				if _, err := linalg.Lstsq(a, rhs); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// Tolerance-0 goldens. The layout fix reads the same values in the same order into the
// same accumulator and changes only where they live, so every bit must survive — a
// relative-tolerance check would pass a genuine reassociation and is not the bar here.
// The xor of all result bits is order-independent and moves if ANY single bit moves; the
// bitwise sum catches a change that happens to cancel in the xor. Values were captured
// from the implementation before the contiguous-back-substitution change and reproduced
// unchanged after it.
func TestSolveBitStableGoldens(t *testing.T) {
	const n = 24
	digest := func(x *tensor.Tensor) (uint64, uint64) {
		var s float64
		var xr uint64
		for _, v := range x.Storage().F64() {
			s += v
			xr ^= math.Float64bits(v)
		}
		return math.Float64bits(s), xr
	}
	rhs := benchMat(n, false)
	solveLU := func() (*tensor.Tensor, error) {
		f, err := linalg.Factor(benchMat(n, false))
		if err != nil {
			return nil, err
		}
		return f.Solve(rhs)
	}
	cases := []struct {
		name     string
		run      func() (*tensor.Tensor, error)
		sum, xor uint64
	}{
		{"LU", solveLU, 0x4038000000000000, 0x3c73ef0dbab7fa6e},
		{"Chol", func() (*tensor.Tensor, error) { return linalg.CholSolve(benchMat(n, true), rhs) },
			0x4038c884da946387, 0x8107957763e4b04b},
		{"Lstsq", func() (*tensor.Tensor, error) { return linalg.Lstsq(benchMat(n, false), rhs) },
			0x4038000000000000, 0x00c1d20052620777},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			x, err := tc.run()
			if err != nil {
				t.Fatal(err)
			}
			sum, xor := digest(x)
			if sum != tc.sum || xor != tc.xor {
				t.Fatalf("%s not bit-identical: sum=%016x xor=%016x, want sum=%016x xor=%016x",
					tc.name, sum, xor, tc.sum, tc.xor)
			}
		})
	}
}
