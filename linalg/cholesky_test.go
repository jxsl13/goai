package linalg_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/linalg"
	"github.com/jxsl13/goai/tensor"
)

// spd builds a random symmetric positive-definite matrix A = B·Bᵀ + n·I.
func spd(rng *rand.Rand, n int) *tensor.Tensor {
	b := randRect(rng, n, n)
	a := tensor.New(tensor.F64, tensor.Shape{n, n})
	for i := range n {
		for j := range n {
			var s float64
			for k := range n {
				s += b.AtF64(i, k) * b.AtF64(j, k) // (B·Bᵀ)[i,j]
			}
			if i == j {
				s += float64(n)
			}
			a.SetF64(s, i, j)
		}
	}
	return a
}

// §V16 tier-1: Cholesky reconstructs A = L·Lᵀ, L is lower-triangular with positive diagonal (§R120).
func TestCholeskyReconstruct(t *testing.T) {
	rng := rand.New(rand.NewPCG(51, 99))
	for _, n := range []int{1, 2, 3, 5, 8} {
		a := spd(rng, n)
		l, err := linalg.Cholesky(a)
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		for i := range n {
			if l.AtF64(i, i) <= 0 {
				t.Errorf("n=%d L[%d,%d] = %g, want > 0", n, i, i, l.AtF64(i, i))
			}
			for j := i + 1; j < n; j++ {
				if l.AtF64(i, j) != 0 {
					t.Errorf("n=%d L[%d,%d] = %g (not lower-triangular)", n, i, j, l.AtF64(i, j))
				}
			}
		}
		// L·Lᵀ == A
		for i := range n {
			for j := range n {
				var s float64
				for k := range n {
					s += l.AtF64(i, k) * l.AtF64(j, k)
				}
				if math.Abs(s-a.AtF64(i, j)) > 1e-9 {
					t.Errorf("n=%d (L·Lᵀ−A)[%d,%d] = %g", n, i, j, s-a.AtF64(i, j))
				}
			}
		}
	}
}

// §V16 tier-1: the SPD solve gives A·x = b, matching the general LU solve.
func TestCholSolve(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 17))
	const n = 6
	a := spd(rng, n)
	bd := make([]float64, n)
	for i := range bd {
		bd[i] = rng.NormFloat64()
	}
	b := tensor.FromFloat64(tensor.Shape{n}, bd)
	x, err := linalg.CholSolve(a, b)
	if err != nil {
		t.Fatal(err)
	}
	for i := range n {
		var s float64
		for j := range n {
			s += a.AtF64(i, j) * x.AtF64(j)
		}
		if math.Abs(s-b.AtF64(i)) > 1e-9 {
			t.Errorf("residual[%d] = %g", i, s-b.AtF64(i))
		}
	}
	// matches the LU solve
	lu, _ := linalg.Solve(a, b)
	for i := range n {
		if math.Abs(x.AtF64(i)-lu.AtF64(i)) > 1e-8 {
			t.Errorf("CholSolve x[%d]=%g != LU %g", i, x.AtF64(i), lu.AtF64(i))
		}
	}
	// matrix RHS
	bm := tensor.FromFloat64(tensor.Shape{n, 2}, make([]float64, n*2))
	for i := range n {
		bm.SetF64(rng.NormFloat64(), i, 0)
		bm.SetF64(rng.NormFloat64(), i, 1)
	}
	xm, err := linalg.CholSolve(a, bm)
	if err != nil || !xm.Shape().Equal(tensor.Shape{n, 2}) {
		t.Fatalf("matrix RHS: %v shape %v", err, xm.Shape())
	}
}

// §V16 tier-1: a symmetric but indefinite/non-SPD matrix is rejected.
func TestCholeskyNonSPD(t *testing.T) {
	// [[1,2],[2,1]] is symmetric with eigenvalues 3 and −1 → not positive definite
	indef := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 2, 1})
	if _, err := linalg.Cholesky(indef); err == nil {
		t.Error("indefinite matrix must be rejected as not positive definite")
	}
	// negative diagonal
	if _, err := linalg.Cholesky(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{-1, 0, 0, 1})); err == nil {
		t.Error("non-positive-definite matrix must be rejected")
	}
}

func TestCholeskyErrors(t *testing.T) {
	if _, err := linalg.Cholesky(tensor.New(tensor.F64, tensor.Shape{2, 3})); err == nil {
		t.Error("non-square must error")
	}
	if _, err := linalg.Cholesky(tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})); err == nil {
		t.Error("non-symmetric must error")
	}
}

func ExampleCholesky() {
	// A = [[4,2],[2,3]] → L = [[2,0],[1,√2]]
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{4, 2, 2, 3})
	l, _ := linalg.Cholesky(a)
	fmt.Printf("%.4f %.4f %.4f\n", l.AtF64(0, 0), l.AtF64(1, 0), l.AtF64(1, 1))
	// Output: 2.0000 1.0000 1.4142
}

// TestCholSolveBandsAndTypedReadMatchSerial covers the two independent changes to CholSolve at
// once, because they interact: the right-hand-side columns are now solved in bands, and an F64
// right-hand side is read through its storage instead of AtF64.
//
// The band split is held bit-identical to the GOMAXPROCS(1) path — a column reads the factor and
// its own column of b and writes only its own output slots, so which worker retires it cannot
// change its arithmetic, and a tolerance here would accept exactly the reassociation the split
// claims not to do. The shape clears parallelCols' cols*n*n < 1<<14 gate, without which both arms
// would run the same serial code.
//
// The typed read is held against the AtF64 path by feeding the SAME values through inputs that
// route differently: a plain F64 matrix takes the storage path, a transposed view is not
// contiguous, and an F32 matrix has no F64 storage at all. This is the shape of defect that hides
// best — a fast path is exercised only in the dtype and layout where it happens to be right, and
// the other arms are never run.
func TestCholSolveBandsAndTypedReadMatchSerial(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 12))
	const n, cols = 96, 24
	a := spd(rng, n)
	rhs := randRect(rng, n, cols)

	solve := func(b *tensor.Tensor) []float64 {
		t.Helper()
		x, err := linalg.CholSolve(a, b)
		if err != nil {
			t.Fatal(err)
		}
		out := make([]float64, x.Numel())
		for i := range out {
			out[i] = x.AtF64(i/cols, i%cols)
		}
		return out
	}
	prev := runtime.GOMAXPROCS(1)
	serial := solve(rhs)
	runtime.GOMAXPROCS(prev)

	// A [cols,n] matrix holding the TRANSPOSE of rhs, viewed back as [n,cols]: the same values in
	// the same logical positions, but the underlying storage is in the other order. Taking that
	// storage directly yields a different matrix, so this arm fails loudly if the fast path
	// forgets to materialize a contiguous copy first. Transposing rhs twice does NOT work — the
	// strides come back to where they started and the view is contiguous again, which is how the
	// first version of this fixture passed with the contiguity check removed.
	tr := tensor.New(tensor.F64, tensor.Shape{cols, n})
	for i := range n {
		for j := range cols {
			tr.SetF64(rhs.AtF64(i, j), j, i)
		}
	}
	view, err := tr.Transpose(1, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		name string
		b    *tensor.Tensor
	}{
		{"banded F64", rhs},
		{"non-contiguous transposed view", view},
		{"F32 right-hand side", rhs.Cast(tensor.F32)},
	} {
		got := solve(c.b)
		if len(got) != len(serial) {
			t.Fatalf("%s: %d values, want %d", c.name, len(got), len(serial))
		}
		for i := range serial {
			if c.name == "F32 right-hand side" {
				// F32 storage rounds the input itself, so this arm is held to the value, not
				// the bits — what it proves is that the path RUNS and stays correct.
				if math.Abs(got[i]-serial[i]) > 1e-4*(1+math.Abs(serial[i])) {
					t.Fatalf("%s element %d: %v, serial %v", c.name, i, got[i], serial[i])
				}
				continue
			}
			if math.Float64bits(got[i]) != math.Float64bits(serial[i]) {
				t.Fatalf("%s element %d: %v, serial %v — the change altered an accumulation",
					c.name, i, got[i], serial[i])
			}
		}
	}
}
