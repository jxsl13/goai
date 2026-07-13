package autograd_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

func randT(rng *rand.Rand, shape tensor.Shape) *tensor.Tensor {
	t := tensor.New(tensor.F64, shape)
	for i := range t.Numel() {
		t.SetF64(rng.NormFloat64(), tensor.Unravel(i, shape)...)
	}
	return t
}

// §V2: the einsum-swap-rule gradient matches central finite differences for a range
// of pure contractions (matmul, batched matmul, transpose, outer, elementwise, dot,
// and a general contraction), w.r.t. every operand.
func TestEinsumGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 22))
	cases := []struct {
		spec   string
		shapes []tensor.Shape
	}{
		{"ij,jk->ik", []tensor.Shape{{2, 3}, {3, 2}}},
		{"bij,bjk->bik", []tensor.Shape{{2, 2, 3}, {2, 3, 2}}},
		{"ij->ji", []tensor.Shape{{2, 3}}},
		{"i,j->ij", []tensor.Shape{{3}, {2}}},
		{"ij,ij->ij", []tensor.Shape{{2, 3}, {2, 3}}},
		{"i,i->", []tensor.Shape{{4}, {4}}},
		{"ij,ik->jk", []tensor.Shape{{2, 3}, {2, 2}}},
		{"ij->i", []tensor.Shape{{2, 3}}},
		{"ij->j", []tensor.Shape{{2, 3}}},
		{"ij->", []tensor.Shape{{2, 3}}},
		{"ijk->ik", []tensor.Shape{{2, 3, 4}}},
		{"ij,jk->i", []tensor.Shape{{2, 3}, {3, 4}}},
		// single-operand diagonals/traces (gradient scatters onto the diagonal)
		{"ii->i", []tensor.Shape{{3, 3}}},
		{"ii->", []tensor.Shape{{3, 3}}},
		{"iij->j", []tensor.Shape{{2, 2, 3}}},
	}
	const h = 1e-6
	for _, c := range cases {
		operands := make([]*tensor.Tensor, len(c.shapes))
		for i, s := range c.shapes {
			operands[i] = randT(rng, s)
		}
		attrs := backend.EinsumAttrs{Spec: c.spec}
		// fixed weights over the output, so the loss depends non-trivially on the output
		fwd, err := backend.Execute(backend.NewContext(), backend.OpEinsum, operands, attrs)
		if err != nil {
			t.Fatalf("%s forward: %v", c.spec, err)
		}
		w := randT(rng, fwd[0].Shape())
		lossOf := func(ctx *backend.Context) float64 {
			o, err := backend.Execute(ctx, backend.OpEinsum, operands, attrs)
			if err != nil {
				t.Fatalf("%s: %v", c.spec, err)
			}
			wc, _ := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{o[0], w}, nil)
			s, _ := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
			return s[0].AtF64()
		}
		tape := autograd.NewTape()
		o, _ := backend.Execute(tape.Context(), backend.OpEinsum, operands, attrs)
		wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{o[0], w}, nil)
		loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatalf("%s backward: %v", c.spec, err)
		}
		for k, op := range operands {
			g := tape.Grad(op)
			if g == nil {
				t.Fatalf("%s operand %d: nil gradient", c.spec, k)
			}
			for i := range op.Numel() {
				idx := tensor.Unravel(i, op.Shape())
				o := op.AtF64(idx...)
				op.SetF64(o+h, idx...)
				lp := lossOf(backend.NewContext())
				op.SetF64(o-h, idx...)
				lm := lossOf(backend.NewContext())
				op.SetF64(o, idx...)
				if num, ana := (lp-lm)/(2*h), g.AtF64(idx...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
					t.Errorf("%s op%d grad[%v]: numeric %.8g vs analytic %.8g", c.spec, k, idx, num, ana)
				}
			}
		}
	}
}

// §V2: the matmul-via-einsum gradient matches g·Bᵀ / Aᵀ·g exactly (a closed-form
// cross-check of the swap rule).
func TestEinsumMatmulGradClosedForm(t *testing.T) {
	a := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4})
	b := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{5, 6, 7, 8})
	tape := autograd.NewTape()
	o, _ := backend.Execute(tape.Context(), backend.OpEinsum, []*tensor.Tensor{a, b}, backend.EinsumAttrs{Spec: "ij,jk->ik"})
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{o[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	// dA = ones · Bᵀ → each dA[i,j] = Σ_k B[j,k]
	ga := tape.Grad(a)
	if ga.AtF64(0, 0) != 11 || ga.AtF64(0, 1) != 15 { // row sums of B: 5+6=11, 7+8=15
		t.Errorf("dA row = [%g,%g], want [11,15]", ga.AtF64(0, 0), ga.AtF64(0, 1))
	}
	// dB = Aᵀ · ones → dB[j,k] = Σ_i A[i,j]
	gb := tape.Grad(b)
	if gb.AtF64(0, 0) != 4 || gb.AtF64(1, 0) != 6 { // col sums of A: 1+3=4, 2+4=6
		t.Errorf("dB col = [%g,%g], want [4,6]", gb.AtF64(0, 0), gb.AtF64(1, 0))
	}
}

// §V2 closed form: for a reduction "ijk->ik" the gradient broadcasts the upstream g
// along the summed index j (dA[i,j,k]=g[i,k] for every j).
func TestEinsumReductionGradBroadcast(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 9))
	a := randT(rng, tensor.Shape{2, 4, 3})
	w := randT(rng, tensor.Shape{2, 3}) // fixed weights over the [i,k] output
	tape := autograd.NewTape()
	o, _ := backend.Execute(tape.Context(), backend.OpEinsum, []*tensor.Tensor{a}, backend.EinsumAttrs{Spec: "ijk->ik"})
	wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{o[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	ga := tape.Grad(a)
	for i := range 2 {
		for j := range 4 {
			for k := range 3 {
				if got := ga.AtF64(i, j, k); got != w.AtF64(i, k) { // g[i,k]=w[i,k], broadcast over j
					t.Errorf("dA[%d,%d,%d]=%g, want w[%d,%d]=%g", i, j, k, got, i, k, w.AtF64(i, k))
				}
			}
		}
	}
}

// §V2 closed form: the diagonal-extraction "ii->i" scatters the upstream gradient
// onto the diagonal — dA[i,j] = g_i if i==j else 0.
func TestEinsumDiagonalGradScatter(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 8))
	a := randT(rng, tensor.Shape{4, 4})
	w := randT(rng, tensor.Shape{4}) // fixed weights over the diagonal output
	tape := autograd.NewTape()
	o, _ := backend.Execute(tape.Context(), backend.OpEinsum, []*tensor.Tensor{a}, backend.EinsumAttrs{Spec: "ii->i"})
	wc, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{o[0], w}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{wc[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	ga := tape.Grad(a)
	for i := range 4 {
		for j := range 4 {
			want := 0.0
			if i == j {
				want = w.AtF64(i) // g_i on the diagonal
			}
			if got := ga.AtF64(i, j); got != want {
				t.Errorf("dA[%d,%d]=%g, want %g", i, j, got, want)
			}
		}
	}
}

// Single-operand reductions and diagonals/traces ("ij->i", "ii->i", "ii->") are now
// differentiable (covered by TestEinsumGradcheck). What still errors on backward while
// the forward works: a repeated index shared across MULTIPLE operands, and a repeated
// OUTPUT index.
func TestEinsumUnsupportedGradient(t *testing.T) {
	cases := []struct {
		spec     string
		operands []*tensor.Tensor
	}{
		{"ii,ij->j", []*tensor.Tensor{ // repeat "ii" alongside a second operand
			tensor.FromFloat64(tensor.Shape{3, 3}, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}),
			tensor.FromFloat64(tensor.Shape{3, 2}, []float64{1, 2, 3, 4, 5, 6}),
		}},
		{"i->ii", []*tensor.Tensor{ // repeated OUTPUT index (diagonalizes the output)
			tensor.FromFloat64(tensor.Shape{3}, []float64{1, 2, 3}),
		}},
	}
	for _, c := range cases {
		tape := autograd.NewTape()
		o, err := backend.Execute(tape.Context(), backend.OpEinsum, c.operands, backend.EinsumAttrs{Spec: c.spec})
		if err != nil {
			t.Fatalf("%s forward should work: %v", c.spec, err)
		}
		loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{o[0]}, nil)
		if err := tape.Backward(loss[0]); err == nil {
			t.Errorf("%s: backward should error (unsupported gradient)", c.spec)
		}
	}
}
