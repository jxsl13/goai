package autograd

import (
	"math"
	"runtime"
	"testing"

	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"
)

// TestQRVJPInterchangeIsBitIdentical locks the interchanged M and B loops to the arithmetic they
// replaced. The reference is the original nest written out verbatim, with both operands of the
// dominant term read down a column.
//
// The claim is that only the loop nesting changed — each M[i][j] still takes the R·R̄ᵀ sum first and
// then subtracts m terms in ascending k, and each B[i][j] still starts from Q̄ and adds n terms in
// ascending k — so the assertion is exact bits.
func TestQRVJPInterchangeIsBitIdentical(t *testing.T) {
	vjp := vjpsMulti[backend.OpQR]
	if vjp == nil {
		t.Fatal("no multi-output VJP registered for OpQR")
	}
	for _, sz := range [][2]int{{1, 1}, {3, 2}, {6, 6}, {17, 5}, {33, 16}} {
		m, n := sz[0], sz[1]
		mk := func(rows, cols int, f func(i, j int) float64) (*tensor.Tensor, [][]float64) {
			x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
			raw := make([][]float64, rows)
			for i := range rows {
				raw[i] = make([]float64, cols)
				for j := range cols {
					v := f(i, j)
					raw[i][j] = v
					x.SetF64(v, i, j)
				}
			}
			return x, raw
		}
		q, qd := mk(m, n, func(i, j int) float64 {
			return math.Sqrt(2/float64(m)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(m))
		})
		r, rd := mk(n, n, func(i, j int) float64 {
			if i > j {
				return 0
			}
			if i == j {
				return 2 + 0.5*float64(i%7) // diagonal away from zero: the rule inverts R
			}
			return math.Sin(float64(i*3+j*5)) * 0.4
		})
		qbar, qb := mk(m, n, func(i, j int) float64 { return math.Cos(float64(i*7+j*3)) * 0.3 })
		rbar, rb := mk(n, n, func(i, j int) float64 {
			if i > j {
				return 0
			}
			return math.Sin(float64(i*5+j*11)) * 0.25
		})
		got, err := vjp(nil, nil, []*tensor.Tensor{q, r}, nil, []*tensor.Tensor{qbar, rbar})
		if err != nil {
			t.Fatalf("%dx%d: %v", m, n, err)
		}
		want := qrVJPColumnWalk(m, n, qd, rd, qb, rb)
		for i := range m {
			for j := range n {
				g := got[0].AtF64(i, j)
				if math.Float64bits(g) != math.Float64bits(want[i][j]) {
					t.Fatalf("%dx%d [%d,%d]: interchanged %v, column-walk %v — not bit-identical",
						m, n, i, j, g, want[i][j])
				}
			}
		}
	}
}

// qrVJPColumnWalk is the pre-interchange rule, kept verbatim.
func qrVJPColumnWalk(m, n int, qd, rd, qb, rb [][]float64) [][]float64 {
	mm := make([][]float64, n)
	for i := range n {
		mm[i] = make([]float64, n)
		for j := range n {
			var s float64
			for k := range n {
				s += rd[i][k] * rb[j][k]
			}
			for k := range m {
				s -= qb[k][i] * qd[k][j]
			}
			mm[i][j] = s
		}
	}
	c := make([][]float64, n)
	for i := range n {
		c[i] = make([]float64, n)
	}
	for i := range n {
		for j := range n {
			if i >= j {
				c[i][j] = mm[i][j]
			} else {
				c[i][j] = mm[j][i]
			}
		}
	}
	b := make([][]float64, m)
	for i := range m {
		b[i] = make([]float64, n)
		for j := range n {
			s := qb[i][j]
			for k := range n {
				s += qd[i][k] * c[k][j]
			}
			b[i][j] = s
		}
	}
	rinv := make([][]float64, n)
	for i := range n {
		rinv[i] = make([]float64, n)
	}
	for col := range n {
		rinv[col][col] = 1 / rd[col][col]
		for i := col - 1; i >= 0; i-- {
			var s float64
			for k := i + 1; k <= col; k++ {
				s += rd[i][k] * rinv[k][col]
			}
			rinv[i][col] = -s / rd[i][i]
		}
	}
	out := make([][]float64, m)
	for i := range m {
		out[i] = make([]float64, n)
		for j := range n {
			var s float64
			for k := j; k < n; k++ {
				s += b[i][k] * rinv[j][k]
			}
			out[i][j] = s
		}
	}
	return out
}

// TestQRVJPStripedRowsMatchSerial locks the striped B loop to the serial one, bit for bit.
//
// Row i of B reads only qb[i], qd[i] and all of c, and writes only b[i], so which worker retires a
// row cannot change its arithmetic — each B[i][j] still adds its n terms in ascending k. That is
// why the assertion is exact rather than a tolerance: a tolerance here would accept precisely the
// reassociation the split claims not to do.
//
// GOMAXPROCS(1) is the reference because logdetParallelIdx falls back to the plain loop when it
// resolves one worker. The shape must clear that helper's work < 1<<15 gate — 64*32*32 = 65536 —
// or both arms would run the same serial code and the test would compare it against itself. 64
// rows also does not divide 12 workers evenly, so the stride's tail is exercised.
func TestQRVJPStripedRowsMatchSerial(t *testing.T) {
	vjp := vjpsMulti[backend.OpQR]
	if vjp == nil {
		t.Fatal("no multi-output VJP registered for OpQR")
	}
	const m, n = 64, 32
	mk := func(rows, cols int, f func(i, j int) float64) *tensor.Tensor {
		x := tensor.New(tensor.F64, tensor.Shape{rows, cols})
		for i := range rows {
			for j := range cols {
				x.SetF64(f(i, j), i, j)
			}
		}
		return x
	}
	q := mk(m, n, func(i, j int) float64 {
		return math.Sqrt(2/float64(m)) * math.Cos(math.Pi*float64(i)*(float64(j)+0.5)/float64(m))
	})
	r := mk(n, n, func(i, j int) float64 {
		if i > j {
			return 0
		}
		if i == j {
			return 1 + 0.25*float64(i)
		}
		return 0.1 * math.Sin(float64(i*3+j))
	})
	qb := mk(m, n, func(i, j int) float64 { return math.Cos(float64(i*7+j*3)) * 0.3 })
	rb := mk(n, n, func(i, j int) float64 { return math.Sin(float64(i*5+j*2)) * 0.2 })
	run := func() []*tensor.Tensor {
		out, err := vjp(nil, nil, []*tensor.Tensor{q, r}, nil, []*tensor.Tensor{qb, rb})
		if err != nil {
			t.Fatal(err)
		}
		return out
	}
	prev := runtime.GOMAXPROCS(1)
	serial := run()
	runtime.GOMAXPROCS(prev)
	striped := run()
	if len(serial) != len(striped) {
		t.Fatalf("%d gradients serial, %d striped", len(serial), len(striped))
	}
	for g := range serial {
		sd, pd := serial[g].Storage().F64(), striped[g].Storage().F64()
		if len(sd) != len(pd) {
			t.Fatalf("gradient %d: %d values serial, %d striped", g, len(sd), len(pd))
		}
		for i := range sd {
			if math.Float64bits(sd[i]) != math.Float64bits(pd[i]) {
				t.Fatalf("gradient %d element %d: serial %v, striped %v — the split changed an"+
					" accumulation", g, i, sd[i], pd[i])
			}
		}
	}
}
