package cpu_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// spd builds a deterministic symmetric positive-definite matrix.
func spd(n int, dt tensor.Dtype) *tensor.Tensor {
	rng := rand.New(rand.NewPCG(5, 7))
	m := make([]float64, n*n)
	for i := range m {
		m[i] = rng.NormFloat64()
	}
	a := tensor.New(dt, tensor.Shape{n, n})
	inv := 1.0 / float64(n)
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += m[i*n+k] * m[j*n+k]
			}
			v := s * inv
			if i == j {
				v += float64(n)
			}
			a.SetF64(v, i, j)
		}
	}
	return a
}

// TestCholeskyCPUMatchesRefBitExactly gates the new cpu kernel against the reference BIT for
// BIT, not to a tolerance. A tolerance comparison would pass for a blocked or reassociated
// factorization too, and the point of this kernel is that it is ref's arithmetic with the row
// update banded — if that ever stops being true, every cross-backend golden silently becomes a
// tolerance test.
//
// The sizes straddle the fan-out gate: at n=8 and n=32 the row update runs inline in both
// arms, and 129 and 200 clear it. 129 is deliberately not a multiple of the worker count.
func TestCholeskyCPUMatchesRefBitExactly(t *testing.T) {
	for _, dt := range []tensor.Dtype{tensor.F64, tensor.F32} {
		for _, n := range []int{1, 8, 32, 129, 200} {
			a := spd(n, dt)
			ref, err := backend.Execute(backend.NewContext().WithBackend(backend.Reference()),
				backend.OpCholesky, []*tensor.Tensor{a}, nil)
			if err != nil {
				t.Fatalf("%v n=%d ref: %v", dt, n, err)
			}
			got, err := backend.Execute(backend.NewContext(), backend.OpCholesky,
				[]*tensor.Tensor{a}, nil)
			if err != nil {
				t.Fatalf("%v n=%d cpu: %v", dt, n, err)
			}
			for i := range n {
				for j := range n {
					w, g := ref[0].AtF64(i, j), got[0].AtF64(i, j)
					if math.Float64bits(w) != math.Float64bits(g) {
						t.Fatalf("%v n=%d: L[%d,%d] cpu %v != ref %v", dt, n, i, j, g, w)
					}
				}
			}
		}
	}
}

// TestCholeskyCPURejectsNonPositiveDefinite pins that the failure path did not move to the new
// kernel's benefit: a matrix with a non-positive pivot must still error rather than return a
// factor full of NaNs.
func TestCholeskyCPURejectsNonPositiveDefinite(t *testing.T) {
	a := tensor.New(tensor.F64, tensor.Shape{2, 2})
	a.SetF64(1, 0, 0)
	a.SetF64(2, 0, 1)
	a.SetF64(2, 1, 0)
	a.SetF64(1, 1, 1) // eigenvalues 3 and -1
	if _, err := backend.Execute(backend.NewContext(), backend.OpCholesky,
		[]*tensor.Tensor{a}, nil); err == nil {
		t.Fatal("want an error on a non-positive-definite matrix, got nil")
	}
}
