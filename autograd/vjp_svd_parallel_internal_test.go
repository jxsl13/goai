package autograd

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestSVDVJPStripedRowsMatchSerial pins the three row splits added to this rule — the mid·Vᵀ
// product, the U·T product, and the tall correction — to the serial path they replaced.
//
// It gates the SPLIT, not the arithmetic, and that is the right scope here: no expression changed,
// only which goroutine evaluates it. Each row reads shared read-only operands and writes its own
// output row, so the two arms must agree bit for bit; a tolerance would accept exactly the
// reassociation the split claims not to do. The arithmetic of the pieces is held separately by
// TestMatTmulRectInterchangeIsBitIdentical and TestSVDVJPProjectionHoistIsBitIdentical.
//
// The shape clears every gate involved (n³ and m·n² against 1<<15) and keeps m > n so the tall
// correction runs at all — it is the largest of the three loops and is skipped entirely when the
// input is square.
func TestSVDVJPStripedRowsMatchSerial(t *testing.T) {
	if runtime.GOMAXPROCS(0) < 2 {
		t.Skip("needs more than one processor to exercise the split")
	}
	vjp := vjpsMulti[backend.OpSVD]
	if vjp == nil {
		t.Fatal("no multi-output VJP registered for OpSVD")
	}
	rng := rand.New(rand.NewSource(20260805))
	const m, n = 96, 48
	mk := func(rows, cols int) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range rows {
			for j := range cols {
				x.SetF64(rng.NormFloat64(), i, j)
			}
		}
		return x
	}
	// Orthonormal-ish U and V and a well-separated spectrum: the rule divides by (s_i²−s_j²), so
	// near-degenerate singular values would put both arms on a numerically different path.
	u, v := mk(m, n), mk(n, n)
	s := tensor.New(tensor.F64, tensor.Shape{n})
	for i := range n {
		s.SetF64(2+0.7*float64(i), i)
	}
	ub, sb, vb := mk(m, n), tensor.New(tensor.F64, tensor.Shape{n}), mk(n, n)
	for i := range n {
		sb.SetF64(math.Sin(float64(i)*0.3)*0.2, i)
	}
	outs := []*tensor.Tensor{u, s, v}
	gouts := []*tensor.Tensor{ub, sb, vb}
	run := func() []*tensor.Tensor {
		got, err := vjp(nil, []*tensor.Tensor{mk(m, n)}, outs, nil, gouts)
		if err != nil {
			t.Fatal(err)
		}
		return got
	}
	prev := runtime.GOMAXPROCS(1)
	serial := run()
	runtime.GOMAXPROCS(prev)
	par := run()

	// Every row must have been filled. Both arms run the same code, so a split that skips a row
	// skips it identically and the comparison below cannot see it — the mutations proved that: a
	// row dropped from either of the two products left this test green until this check existed.
	// With random inputs every entry of Ā is nonzero with probability one, so an all-zero row means
	// nobody wrote it.
	ad := par[0].Storage().F64()
	for i := range m {
		allZero := true
		for j := range n {
			if ad[i*n+j] != 0 {
				allZero = false
				break
			}
		}
		if allZero {
			t.Fatalf("row %d of the gradient is entirely zero — a split skipped it", i)
		}
	}
	for g := range serial {
		sd, pd := serial[g].Storage().F64(), par[g].Storage().F64()
		if len(sd) != len(pd) {
			t.Fatalf("gradient %d: %d values serial, %d striped", g, len(sd), len(pd))
		}
		for i := range sd {
			if math.Float64bits(sd[i]) != math.Float64bits(pd[i]) {
				t.Fatalf("gradient %d element %d: serial %v, striped %v", g, i, sd[i], pd[i])
			}
		}
	}
}
