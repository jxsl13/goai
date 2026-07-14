package nn_test

import (
	"fmt"
	"math"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"
)

// §V16 tier-1 (§T618): Hyper-Connections / mHC. Structure verified against arXiv:2409.19606
// (HC) + arXiv:2512.24880 (mHC). Tests: residual-equivalence at n=1, exact mixing parity vs
// hand-computed, the doubly-stochastic property of the mHC-projected Ar, and gradcheck through
// the Sinkhorn unroll.

// n=1 with the residual-equivalent init reduces EXACTLY to a plain residual connection:
// LayerInput(H)=H and Update(H,y)=y+H.
func TestHyperConnectionResidualEquivalence(t *testing.T) {
	hc, err := nn.NewHyperConnection(tensor.F64, 1, nn.HCNone)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	h := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{2, -1, 0.5})
	h0, err := hc.LayerInput(ctx, h)
	if err != nil {
		t.Fatal(err)
	}
	for j := range 3 {
		if got, want := h0.AtF64(0, j), h.AtF64(0, j); math.Abs(got-want) > 1e-12 {
			t.Fatalf("LayerInput n=1: col %d got %v want %v", j, got, want)
		}
	}
	y := tensor.FromFloat64(tensor.Shape{1, 3}, []float64{0.3, 0.4, -0.2})
	hn, err := hc.Update(ctx, h, y)
	if err != nil {
		t.Fatal(err)
	}
	for j := range 3 {
		if got, want := hn.AtF64(0, j), y.AtF64(0, j)+h.AtF64(0, j); math.Abs(got-want) > 1e-12 {
			t.Fatalf("Update n=1 (residual): col %d got %v want %v", j, got, want)
		}
	}
}

// Exact mixing parity for n=2 with explicit Am/B/Ar against the paper's equations
// h₀ = Amᵀ·H and Ĥ = Bᵀ·y + Arᵀ·H.
func TestHyperConnectionMixingParity(t *testing.T) {
	hc, err := nn.NewHyperConnection(tensor.F64, 2, nn.HCNone)
	if err != nil {
		t.Fatal(err)
	}
	hc.Am = tensor.FromFloat64(tensor.Shape{2, 1}, []float64{0.6, 0.4})
	hc.B = tensor.FromFloat64(tensor.Shape{1, 2}, []float64{0.7, 0.3})
	hc.Ar = tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0.9, 0.1, 0.2, 0.8})
	ctx := backend.NewContext()
	h := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, 2, 3, 4}) // rows = streams, cols = F

	h0, err := hc.LayerInput(ctx, h)
	if err != nil {
		t.Fatal(err)
	}
	// h₀ = Amᵀ·H = [0.6·[1,2] + 0.4·[3,4]] = [1.8, 3.2]
	for j, want := range []float64{0.6*1 + 0.4*3, 0.6*2 + 0.4*4} {
		if got := h0.AtF64(0, j); math.Abs(got-want) > 1e-12 {
			t.Fatalf("LayerInput parity: col %d got %v want %v", j, got, want)
		}
	}

	y := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{5, 6})
	hn, err := hc.Update(ctx, h, y)
	if err != nil {
		t.Fatal(err)
	}
	// Ĥ[i,:] = B[i]·y + Σ_j Ar[j,i]·H[j,:]
	//   i=0: 0.7·[5,6] + (0.9·[1,2] + 0.2·[3,4]) = [3.5,4.2]+[1.5,2.6] = [5.0,6.8]
	//   i=1: 0.3·[5,6] + (0.1·[1,2] + 0.8·[3,4]) = [1.5,1.8]+[2.5,3.4] = [4.0,5.2]
	want := [][]float64{{5.0, 6.8}, {4.0, 5.2}}
	for i := range 2 {
		for j := range 2 {
			if got := hn.AtF64(i, j); math.Abs(got-want[i][j]) > 1e-12 {
				t.Fatalf("Update parity: [%d,%d] got %v want %v", i, j, got, want[i][j])
			}
		}
	}
}

// The mHC projection makes Ar doubly-stochastic: every row AND every column sums to 1.
func TestHyperConnectionDoublyStochastic(t *testing.T) {
	hc, err := nn.NewHyperConnection(tensor.F64, 3, nn.HCDoublyStochastic, nn.WithSinkhornIters(60))
	if err != nil {
		t.Fatal(err)
	}
	// an arbitrary, deliberately unbalanced Ar
	hc.Ar = tensor.FromFloat64(tensor.Shape{3, 3}, []float64{
		2.0, -1.0, 0.3,
		0.1, 1.5, -0.7,
		-0.4, 0.2, 1.1,
	})
	ar, err := hc.EffectiveAr(backend.NewContext())
	if err != nil {
		t.Fatal(err)
	}
	for i := range 3 {
		var rs, cs float64
		for j := range 3 {
			rs += ar.AtF64(i, j)
			cs += ar.AtF64(j, i)
			if ar.AtF64(i, j) < 0 {
				t.Fatalf("projected Ar has a negative entry at [%d,%d]: %v", i, j, ar.AtF64(i, j))
			}
		}
		if math.Abs(rs-1) > 1e-6 {
			t.Fatalf("row %d sums to %v, want 1 (not doubly-stochastic)", i, rs)
		}
		if math.Abs(cs-1) > 1e-6 {
			t.Fatalf("col %d sums to %v, want 1 (not doubly-stochastic)", i, cs)
		}
	}
}

// §V2: autograd gradients of Am, B and Ar — including through the mHC Sinkhorn unroll — match
// central finite differences.
func TestHyperConnectionGradcheck(t *testing.T) {
	newHC := func() *nn.HyperConnection {
		hc, err := nn.NewHyperConnection(tensor.F64, 2, nn.HCDoublyStochastic, nn.WithSinkhornIters(8))
		if err != nil {
			t.Fatal(err)
		}
		hc.Am = tensor.FromFloat64(tensor.Shape{2, 1}, []float64{0.6, -0.3})
		hc.B = tensor.FromFloat64(tensor.Shape{1, 2}, []float64{0.7, 0.2})
		hc.Ar = tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0.5, -0.2, 0.1, 0.4})
		return hc
	}
	h := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1, -1, 0.5, 2})
	mask := tensor.FromFloat64(tensor.Shape{2, 2}, []float64{0.7, -1.2, 0.4, 0.9})

	// End-to-end cell path with an identity layer 𝒯: h₀ = LayerInput(H) (exercises Am), then
	// Ĥ = Update(H, h₀) (exercises B and Ar, and Am again via the depth term). loss = Σ mask⊙Ĥ.
	forward := func(hc *nn.HyperConnection, ctx *backend.Context) (*tensor.Tensor, error) {
		h0, err := hc.LayerInput(ctx, h)
		if err != nil {
			return nil, err
		}
		return hc.Update(ctx, h, h0)
	}

	// analytic gradients via the tape
	hc := newHC()
	tape := autograd.NewTape()
	hn, err := forward(hc, tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{hn, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{prod[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}

	lossOf := func(hc *nn.HyperConnection) float64 {
		out, err := forward(hc, backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range 2 {
			for j := range 2 {
				acc += mask.AtF64(i, j) * out.AtF64(i, j)
			}
		}
		return acc
	}
	const eps = 1e-6
	check := func(name string, param *tensor.Tensor, get func(*nn.HyperConnection) *tensor.Tensor) {
		g := tape.Grad(param)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		sh := param.Shape()
		rows, cols := sh[0], sh[1]
		for i := range rows {
			for j := range cols {
				hcp, hcm := newHC(), newHC()
				orig := param.AtF64(i, j)
				get(hcp).SetF64(orig+eps, i, j)
				get(hcm).SetF64(orig-eps, i, j)
				fd := (lossOf(hcp) - lossOf(hcm)) / (2 * eps)
				if an := g.AtF64(i, j); math.Abs(an-fd) > 1e-5 {
					t.Fatalf("%s grad[%d,%d]: analytic %v vs finite-diff %v", name, i, j, an, fd)
				}
			}
		}
	}
	check("Am", hc.Am, func(h *nn.HyperConnection) *tensor.Tensor { return h.Am })
	check("B", hc.B, func(h *nn.HyperConnection) *tensor.Tensor { return h.B })
	check("Ar", hc.Ar, func(h *nn.HyperConnection) *tensor.Tensor { return h.Ar })
}

// ExampleHyperConnection runs one plain (HCNone) cell over a 2-stream expansion with an
// identity layer, showing the Expand → LayerInput → Update → Collapse flow. With the
// residual-equivalent init the arithmetic is exact.
func ExampleHyperConnection() {
	hc, _ := nn.NewHyperConnection(tensor.F64, 2, nn.HCNone) // Ar=I, B=Am=1
	ctx := backend.NewContext()

	x := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{1, 2})
	h, _ := hc.Expand(ctx, x)      // H = [[1,2],[1,2]]
	h0, _ := hc.LayerInput(ctx, h) // Amᵀ·H = [2,4]
	// identity layer: 𝒯(h0) = h0
	h, _ = hc.Update(ctx, h, h0)  // Bᵀ·h0 + Arᵀ·H = [[3,6],[3,6]]
	out, _ := hc.Collapse(ctx, h) // sum streams = [6,12]

	fmt.Printf("%.0f %.0f\n", out.AtF64(0, 0), out.AtF64(0, 1))
	// Output: 6 12
}

// ExampleHyperConnection_mHC shows the mHC mode: the width-mixing matrix Ar is projected onto
// the doubly-stochastic manifold, so each of its rows sums to 1 (a convex combination).
func ExampleHyperConnection_mHC() {
	hc, _ := nn.NewHyperConnection(tensor.F64, 2, nn.HCDoublyStochastic, nn.WithSinkhornIters(40))
	hc.Ar = tensor.FromFloat64(tensor.Shape{2, 2}, []float64{1.5, -0.5, 0.2, 0.8})
	ar, _ := hc.EffectiveAr(backend.NewContext())

	fmt.Printf("row0 sum = %.0f\n", ar.AtF64(0, 0)+ar.AtF64(0, 1))
	// Output: row0 sum = 1
}

// DHC with zero-init scales is EXACTLY the static SHC cell (the dynamic correction vanishes),
// so training "warms in" from the stable static baseline.
func TestDynamicHyperConnectionReducesToStatic(t *testing.T) {
	const n, d = 2, 4
	dyn, err := nn.NewDynamicHyperConnection(tensor.F64, n, d, 7)
	if err != nil {
		t.Fatal(err)
	}
	stat, err := nn.NewHyperConnection(tensor.F64, n, nn.HCNone)
	if err != nil {
		t.Fatal(err)
	}
	ctx := backend.NewContext()
	h := tensor.FromFloat64(tensor.Shape{n, d}, []float64{1, -1, 0.5, 2, 0.3, 1, -2, 0.7})
	y := tensor.FromFloat64(tensor.Shape{1, d}, []float64{0.2, -0.4, 0.6, 0.1})

	dynH0, _ := dyn.LayerInput(ctx, h)
	statH0, _ := stat.LayerInput(ctx, h)
	dynU, _ := dyn.Update(ctx, h, y)
	statU, _ := stat.Update(ctx, h, y)
	for j := range d {
		if math.Abs(dynH0.AtF64(0, j)-statH0.AtF64(0, j)) > 1e-12 {
			t.Fatalf("DHC LayerInput != SHC at col %d: %v vs %v", j, dynH0.AtF64(0, j), statH0.AtF64(0, j))
		}
	}
	for i := range n {
		for j := range d {
			if math.Abs(dynU.AtF64(i, j)-statU.AtF64(i, j)) > 1e-12 {
				t.Fatalf("DHC Update != SHC at [%d,%d]: %v vs %v", i, j, dynU.AtF64(i, j), statU.AtF64(i, j))
			}
		}
	}
}

// §V2: DHC gradients through the RMSNorm→proj→tanh→scale dynamic path match finite differences.
func TestDynamicHyperConnectionGradcheck(t *testing.T) {
	const n, d = 2, 3
	newHC := func() *nn.HyperConnection {
		hc, err := nn.NewDynamicHyperConnection(tensor.F64, n, d, 5)
		if err != nil {
			t.Fatal(err)
		}
		// activate the dynamic path with non-trivial scales
		hc.Dyn.Sm = tensor.FromFloat64(tensor.Shape{1, 1}, []float64{0.5})
		hc.Dyn.Sr = tensor.FromFloat64(tensor.Shape{1, 1}, []float64{0.3})
		hc.Dyn.Sb = tensor.FromFloat64(tensor.Shape{1, 1}, []float64{0.4})
		return hc
	}
	h := tensor.FromFloat64(tensor.Shape{n, d}, []float64{1, -1, 0.5, 0.3, 2, -0.7})
	mask := tensor.FromFloat64(tensor.Shape{n, d}, []float64{0.7, -1.2, 0.4, 0.9, -0.6, 1.1})
	forward := func(hc *nn.HyperConnection, ctx *backend.Context) (*tensor.Tensor, error) {
		h0, err := hc.LayerInput(ctx, h)
		if err != nil {
			return nil, err
		}
		return hc.Update(ctx, h, h0)
	}
	hc := newHC()
	tape := autograd.NewTape()
	hn, err := forward(hc, tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{hn, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{prod[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	lossOf := func(hc *nn.HyperConnection) float64 {
		out, err := forward(hc, backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		var acc float64
		for i := range n {
			for j := range d {
				acc += mask.AtF64(i, j) * out.AtF64(i, j)
			}
		}
		return acc
	}
	const eps = 1e-6
	check := func(name string, param *tensor.Tensor, get func(*nn.HyperConnection) *tensor.Tensor) {
		g := tape.Grad(param)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		sh := param.Shape()
		rows, cols := sh[0], sh[1]
		for i := range rows {
			for j := range cols {
				hcp, hcm := newHC(), newHC()
				orig := param.AtF64(i, j)
				get(hcp).SetF64(orig+eps, i, j)
				get(hcm).SetF64(orig-eps, i, j)
				fd := (lossOf(hcp) - lossOf(hcm)) / (2 * eps)
				if an := g.AtF64(i, j); math.Abs(an-fd) > 1e-5 {
					t.Fatalf("%s grad[%d,%d]: analytic %v vs finite-diff %v", name, i, j, an, fd)
				}
			}
		}
	}
	check("Wm", hc.Dyn.Wm, func(h *nn.HyperConnection) *tensor.Tensor { return h.Dyn.Wm })
	check("Wr", hc.Dyn.Wr, func(h *nn.HyperConnection) *tensor.Tensor { return h.Dyn.Wr })
	check("Wb", hc.Dyn.Wb, func(h *nn.HyperConnection) *tensor.Tensor { return h.Dyn.Wb })
	check("Sr", hc.Dyn.Sr, func(h *nn.HyperConnection) *tensor.Tensor { return h.Dyn.Sr })
}

// HCOrthogonal projects Ar so that Ar·Arᵀ ≈ I (an orthogonal matrix): the Newton-Schulz
// orthogonalization converges to the orthogonal polar factor.
func TestHyperConnectionOrthogonal(t *testing.T) {
	const n = 3
	hc, err := nn.NewHyperConnection(tensor.F64, n, nn.HCOrthogonal, nn.WithSinkhornIters(30))
	if err != nil {
		t.Fatal(err)
	}
	hc.Ar = tensor.FromFloat64(tensor.Shape{n, n}, []float64{
		1.2, 0.3, -0.5,
		0.1, 0.9, 0.4,
		-0.6, 0.2, 1.1,
	})
	ar, err := hc.EffectiveAr(backend.NewContext())
	if err != nil {
		t.Fatal(err)
	}
	// Q·Qᵀ should be the identity
	for i := range n {
		for j := range n {
			var dot float64
			for k := range n {
				dot += ar.AtF64(i, k) * ar.AtF64(j, k)
			}
			want := 0.0
			if i == j {
				want = 1.0
			}
			if math.Abs(dot-want) > 1e-5 {
				t.Fatalf("Q·Qᵀ[%d,%d] = %v, want %v (not orthogonal)", i, j, dot, want)
			}
		}
	}
}

// §V2: HCOrthogonal gradients of Ar flow through the Newton-Schulz unroll (differentiability).
func TestHyperConnectionOrthogonalGradcheck(t *testing.T) {
	const n = 2
	newHC := func() *nn.HyperConnection {
		hc, err := nn.NewHyperConnection(tensor.F64, n, nn.HCOrthogonal, nn.WithSinkhornIters(6))
		if err != nil {
			t.Fatal(err)
		}
		hc.Ar = tensor.FromFloat64(tensor.Shape{n, n}, []float64{0.8, 0.3, -0.2, 1.1})
		hc.Am = tensor.FromFloat64(tensor.Shape{n, 1}, []float64{0.6, 0.4})
		hc.B = tensor.FromFloat64(tensor.Shape{1, n}, []float64{0.7, 0.3})
		return hc
	}
	h := tensor.FromFloat64(tensor.Shape{n, 2}, []float64{1, -1, 0.5, 2})
	y := tensor.FromFloat64(tensor.Shape{1, 2}, []float64{0.3, 0.8})
	mask := tensor.FromFloat64(tensor.Shape{n, 2}, []float64{0.7, -1.2, 0.4, 0.9})

	hc := newHC()
	tape := autograd.NewTape()
	hn, err := hc.Update(tape.Context(), h, y)
	if err != nil {
		t.Fatal(err)
	}
	prod, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{hn, mask}, nil)
	loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{prod[0]}, nil)
	if err := tape.Backward(loss[0]); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(hc.Ar)
	if g == nil {
		t.Fatal("nil Ar gradient through Newton-Schulz")
	}
	lossOf := func(hc *nn.HyperConnection) float64 {
		out, _ := hc.Update(backend.NewContext(), h, y)
		var acc float64
		for i := range n {
			for j := range 2 {
				acc += mask.AtF64(i, j) * out.AtF64(i, j)
			}
		}
		return acc
	}
	const eps = 1e-6
	for i := range n {
		for j := range n {
			hcp, hcm := newHC(), newHC()
			orig := hc.Ar.AtF64(i, j)
			hcp.Ar.SetF64(orig+eps, i, j)
			hcm.Ar.SetF64(orig-eps, i, j)
			fd := (lossOf(hcp) - lossOf(hcm)) / (2 * eps)
			if an := g.AtF64(i, j); math.Abs(an-fd) > 1e-5 {
				t.Fatalf("Ar grad[%d,%d] thru Newton-Schulz: analytic %v vs fd %v", i, j, an, fd)
			}
		}
	}
}
