package nn_test

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"github.com/jxsl13/goai/autograd"
	"github.com/jxsl13/goai/backend"
	"github.com/jxsl13/goai/nn"
	"github.com/jxsl13/goai/tensor"

	_ "github.com/jxsl13/goai/backend/cpu"
	_ "github.com/jxsl13/goai/backend/ref"
)

// peerGELU is the exact GELU (matches backend.OpGELU) for hand-computed oracles.
func peerGELU(z float64) float64 { return 0.5 * z * (1 + math.Erf(z/math.Sqrt2)) }

// peerDot is a plain dot product for the single-neuron expert pre-activation.
func peerDot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func peerRow(t *tensor.Tensor, r int) []float64 {
	c := t.Shape()[1]
	out := make([]float64, c)
	for j := range c {
		out[j] = t.AtF64(r, j)
	}
	return out
}

// §V16.2 (COLLAPSE, single expert): N=1, topK=1 ⇒ the pool has one expert and one
// product key, the softmax gate over a single score is exactly 1, so PEER(x) must
// equal g·v₀·GELU(u₀ᵀx) = v₀·GELU(u₀ᵀx) hand-computed to 1e-10 — pinning the
// single-neuron expert math independent of any retrieval choice.
func TestPEERSingleExpertCollapse(t *testing.T) {
	const d = 4
	p := nn.NewPEER(tensor.F64, d, 1, 1, 42) // n=1 ⇒ N=1
	if p.NumExperts() != 1 {
		t.Fatalf("expected N=1, got %d", p.NumExperts())
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, 3, d)

	y, idx, gates, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	u0 := peerRow(p.U, 0)
	v0 := peerRow(p.V, 0)
	for r := range 3 {
		if len(idx[r]) != 1 || idx[r][0] != 0 {
			t.Fatalf("row %d: expected only expert 0, got %v", r, idx[r])
		}
		if g := gates.AtF64(r, 0); math.Abs(g-1) > 1e-12 {
			t.Fatalf("row %d: single-expert gate must be 1, got %g", r, g)
		}
		a := peerGELU(peerDot(peerRow(x, r), u0))
		for c := range d {
			want := v0[c] * a
			if got := y.AtF64(r, c); math.Abs(got-want) > 1e-10 {
				t.Fatalf("row %d col %d: PEER %g != hand %g", r, c, got, want)
			}
		}
	}
}

// §V16.2 (COLLAPSE, dense): with all product-key scores tied (Wq forced to 0 ⇒ q=0
// ⇒ s1=s2=0), topK=N retrieves ALL experts and the softmax gate is uniform 1/N.
// PEER then equals (1/N)·Σ_e v_e·GELU(u_eᵀx), a direct dense two-matrix GELU-MLP,
// to 1e-10 — pinning the expert-sum math independent of retrieval.
func TestPEERDenseCollapse(t *testing.T) {
	const d, n = 3, 3 // N = 9
	N := n * n
	p := nn.NewPEER(tensor.F64, d, n, N, 42, nn.WithPEERSubKeyTopK(n)) // topK=N, retrieve all
	// Force every query to zero ⇒ all product-key scores equal ⇒ uniform gate.
	for i := range p.Wq.Numel() {
		p.Wq.SetF64(0, tensor.Unravel(i, p.Wq.Shape())...)
	}
	rng := rand.New(rand.NewPCG(4, 8))
	x := randMat(rng, 4, d)

	y, idx, gates, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for r := range 4 {
		if len(idx[r]) != N {
			t.Fatalf("row %d: topK=N must retrieve all %d experts, got %d", r, N, len(idx[r]))
		}
		for c := range N {
			if g := gates.AtF64(r, c); math.Abs(g-1.0/float64(N)) > 1e-12 {
				t.Fatalf("row %d: gate %d = %g, want uniform %g", r, c, g, 1.0/float64(N))
			}
		}
		xr := peerRow(x, r)
		for col := range d {
			var dense float64
			for e := range N {
				dense += peerRow(p.V, e)[col] * peerGELU(peerDot(xr, peerRow(p.U, e)))
			}
			want := dense / float64(N) // uniform 1/N gate
			if got := y.AtF64(r, col); math.Abs(got-want) > 1e-10 {
				t.Fatalf("row %d col %d: PEER %g != dense/N %g", r, col, got, want)
			}
		}
	}
}

// §V16.3 (GRADCHECK): central finite differences of the (masked) output w.r.t. the
// sub-keys K1,K2, the query Wq (gradient flows ONLY through the softmax gate — the
// routing indices are detached) and the SELECTED rows of the expert pool U,V match
// the tape gradients. topK=2 makes the gate a non-trivial softmax over 2 scores so
// Wq/K1/K2 receive real gradient. lossOf asserts the routing never flips under the
// perturbation, keeping the finite difference valid.
func TestPEERGradcheck(t *testing.T) {
	const tks, d, n, topK = 3, 4, 4, 2 // N = 16
	rng := rand.New(rand.NewPCG(17, 19))
	p := nn.NewPEER(tensor.F64, d, n, topK, 5)
	x := randMat(rng, tks, d)
	mask := randMat(rng, tks, d)

	baseIdx := func() [][]int {
		_, idx, _, err := p.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		return idx
	}
	base := baseIdx()
	sameRouting := func(idx [][]int) bool {
		for r := range base {
			if len(idx[r]) != len(base[r]) {
				return false
			}
			for c := range base[r] {
				if idx[r][c] != base[r][c] {
					return false
				}
			}
		}
		return true
	}

	lossOf := func() float64 {
		y, idx, _, err := p.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		if !sameRouting(idx) {
			t.Fatalf("routing flipped under FD perturbation — pick another seed (FD invalid at a selection boundary)")
		}
		var s float64
		for i := range tks {
			for j := range d {
				s += mask.AtF64(i, j) * y.AtF64(i, j)
			}
		}
		return s
	}

	tape := autograd.NewTape()
	y, _, _, err := p.Forward(tape.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	pr, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, mask}, nil)
	sum, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{pr[0]}, nil)
	if err := tape.Backward(sum[0]); err != nil {
		t.Fatal(err)
	}

	const h = 1e-6
	entries, maxRel := 0, 0.0
	// selected expert rows (the only rows of U,V with non-zero gradient)
	selected := map[int]bool{}
	for _, row := range base {
		for _, e := range row {
			selected[e] = true
		}
	}
	checkEntry := func(name string, param *tensor.Tensor, idx []int) {
		g := tape.Grad(param)
		if g == nil {
			t.Fatalf("%s: nil gradient", name)
		}
		o := param.AtF64(idx...)
		param.SetF64(o+h, idx...)
		lp := lossOf()
		param.SetF64(o-h, idx...)
		lm := lossOf()
		param.SetF64(o, idx...)
		num, ana := (lp-lm)/(2*h), g.AtF64(idx...)
		rel := math.Abs(num-ana) / math.Max(1, math.Abs(num))
		maxRel = math.Max(maxRel, rel)
		entries++
		if rel > 1e-4 {
			t.Errorf("%s%v: FD %g vs tape %g (rel %g)", name, idx, num, ana, rel)
		}
	}

	// K1, K2, Wq: full sweep (small; gradient flows through the gate softmax).
	nonzeroGate := false
	for _, pr := range []struct {
		name string
		t    *tensor.Tensor
	}{{"K1", p.K1}, {"K2", p.K2}, {"Wq", p.Wq}} {
		g := tape.Grad(pr.t)
		for i := range pr.t.Numel() {
			idx := tensor.Unravel(i, pr.t.Shape())
			if math.Abs(g.AtF64(idx...)) > 1e-9 {
				nonzeroGate = true
			}
			checkEntry(pr.name, pr.t, idx)
		}
	}
	if !nonzeroGate {
		t.Fatal("gate params K1/K2/Wq received no gradient — routing gate not differentiable")
	}

	// U, V: only the selected rows (all d columns) — the retrieval gather path.
	nonzeroExpert := false
	for e := range selected {
		for c := range d {
			idx := []int{e, c}
			if g := tape.Grad(p.U); math.Abs(g.AtF64(idx...)) > 1e-9 {
				nonzeroExpert = true
			}
			checkEntry("U", p.U, idx)
			checkEntry("V", p.V, idx)
		}
	}
	if !nonzeroExpert {
		t.Fatal("selected expert rows of U received no gradient — gather path not differentiable")
	}
	// Sanity: an UN-selected expert row must carry exactly zero gradient.
	unsel := -1
	for e := range p.NumExperts() {
		if !selected[e] {
			unsel = e
			break
		}
	}
	if unsel >= 0 {
		gU := tape.Grad(p.U)
		for c := range d {
			if v := gU.AtF64(unsel, c); v != 0 {
				t.Fatalf("un-retrieved expert %d got non-zero grad %g (should be untouched)", unsel, v)
			}
		}
	}
	t.Logf("gradcheck OK: %d entries (K1,K2,Wq via gate + selected U,V rows via gather), max rel err %.2e", entries, maxRel)
}

// §V16.4a (THE VALUE — efficiency): retrieval gathers only H·topK rows regardless
// of N. We assert the retrieved-index count per token equals H·topK (not N) across
// heads, and report the compute/param arithmetic: 2·N·d pool params, per-token
// FLOPs ∝ H·(k·d + 2n·(dk/2)) not N·d.
func TestPEEREfficiency(t *testing.T) {
	const d, n, topK, heads = 6, 16, 4, 2 // N = 256
	p := nn.NewPEER(tensor.F64, d, n, topK, 3, nn.WithPEERHeads(heads))
	N := p.NumExperts()
	rng := rand.New(rand.NewPCG(2, 3))
	x := randMat(rng, 5, d)

	_, idx, gates, err := p.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for r := range 5 {
		if len(idx[r]) != heads*topK {
			t.Fatalf("row %d: gathered %d experts, want H·topK=%d (NOT N=%d)", r, len(idx[r]), heads*topK, N)
		}
	}
	if gates.Shape()[1] != heads*topK {
		t.Fatalf("gates cols %d != H·topK %d", gates.Shape()[1], heads*topK)
	}
	poolParams := 2 * N * d
	gotPool := p.U.Numel() + p.V.Numel()
	if gotPool != poolParams {
		t.Fatalf("pool params %d != 2·N·d %d", gotPool, poolParams)
	}
	half := p.QueryDim() / 2
	perTokenExpertFLOPs := heads * topK * d    // gather+MLP touches only k experts
	perTokenRouteFLOPs := heads * 2 * n * half // 2n sub-key dot products of width dk/2
	denseFLOPs := N * d                        // a dense million-expert pass would touch all N
	t.Logf("N=%d experts, pool=2·N·d=%d params; per-token retrieval touches H·topK=%d experts (expert FLOPs∝%d, route FLOPs∝%d) vs dense N·d∝%d (%.0f× fewer expert touches)",
		N, gotPool, heads*topK, perTokenExpertFLOPs, perTokenRouteFLOPs, denseFLOPs, float64(N)/float64(heads*topK))
}

// §V16.4b (THE VALUE — learns & specializes): a small PEER fits a nonlinear target
// (MSE drops sharply over a few hundred Adam steps) and different inputs route to
// different experts (retrieval specializes). Reports the loss trajectory and the
// number of distinct experts used.
func TestPEERLearnsAndSpecializes(t *testing.T) {
	const tks, d, n, topK = 16, 6, 8, 4 // N = 64
	rng := rand.New(rand.NewPCG(21, 23))
	p := nn.NewPEER(tensor.F64, d, n, topK, 99)
	x := randMat(rng, tks, d)

	// Nonlinear teacher: target = GELU(x·A)·B, a fixed random 2-layer net.
	A := randMat(rng, d, d)
	B := randMat(rng, d, d)
	ctx0 := backend.NewContext()
	ta, _ := backend.Execute(ctx0, backend.OpMatMul, []*tensor.Tensor{x, A}, nil)
	tg, _ := backend.Execute(ctx0, backend.OpGELU, []*tensor.Tensor{ta[0]}, nil)
	target, _ := backend.Execute(ctx0, backend.OpMatMul, []*tensor.Tensor{tg[0], B}, nil)
	tgt := target[0]

	opt := nn.NewAdam(p.Params(), 5e-3)
	mse := func() (float64, [][]int) {
		tape := autograd.NewTape()
		y, idx, _, err := p.Forward(tape.Context(), x)
		if err != nil {
			t.Fatal(err)
		}
		diff, _ := backend.Execute(tape.Context(), backend.OpSub, []*tensor.Tensor{y, tgt}, nil)
		sq, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{diff[0], diff[0]}, nil)
		loss, _ := backend.Execute(tape.Context(), backend.OpSum, []*tensor.Tensor{sq[0]}, nil)
		if err := tape.Backward(loss[0]); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
		return loss[0].AtF64() / float64(tks*d), idx
	}

	first, _ := mse()
	var last float64
	var idx [][]int
	for range 400 {
		last, idx = mse()
	}
	if last >= first {
		t.Fatalf("PEER did not learn: MSE %g -> %g", first, last)
	}
	if last > 0.5*first {
		t.Fatalf("PEER learned too little: MSE %g -> %g (want < 50%% of start)", first, last)
	}

	// Specialization: distinct experts across tokens, and not every token routes
	// to the same set.
	used := map[int]bool{}
	firstSet := fmt.Sprint(idx[0])
	allSame := true
	for r := range tks {
		for _, e := range idx[r] {
			used[e] = true
		}
		if fmt.Sprint(idx[r]) != firstSet {
			allSame = false
		}
	}
	if allSame {
		t.Fatal("all tokens routed to the identical expert set — no specialization")
	}
	t.Logf("PEER learned: MSE %.4f -> %.4f (%.0f%% drop) over 400 steps; %d/%d distinct experts used across %d tokens",
		first, last, 100*(1-last/first), len(used), p.NumExperts(), tks)
}

// §V16.5 (determinism): same seed ⇒ identical params and identical routing for the
// same input.
func TestPEERDeterminism(t *testing.T) {
	const d, n, topK = 4, 4, 2
	a := nn.NewPEER(tensor.F64, d, n, topK, 123)
	b := nn.NewPEER(tensor.F64, d, n, topK, 123)
	for pi := range a.Params() {
		pa, pb := a.Params()[pi], b.Params()[pi]
		for i := range pa.Numel() {
			idx := tensor.Unravel(i, pa.Shape())
			if pa.AtF64(idx...) != pb.AtF64(idx...) {
				t.Fatalf("param %d differs at %v — construction not deterministic", pi, idx)
			}
		}
	}
	rng := rand.New(rand.NewPCG(1, 1))
	x := randMat(rng, 3, d)
	_, ia, _, _ := a.Forward(backend.NewContext(), x)
	_, ib, _, _ := b.Forward(backend.NewContext(), x)
	for r := range ia {
		if fmt.Sprint(ia[r]) != fmt.Sprint(ib[r]) {
			t.Fatalf("routing differs at row %d: %v vs %v", r, ia[r], ib[r])
		}
	}
}

// §V16.5 (shape errors): a wrong input width is rejected.
func TestPEERShapeErrors(t *testing.T) {
	p := nn.NewPEER(tensor.F64, 4, 4, 2, 1)
	if _, _, _, err := p.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{3, 5})); err == nil {
		t.Fatal("expected error for x with wrong feature width")
	}
	if _, _, _, err := p.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4})); err == nil {
		t.Fatal("expected error for rank-1 x")
	}
}

func ExamplePEER() {
	// A pool of N = n² = 64 single-neuron experts over model dim 4; each token
	// retrieves its top-2 via product-key memory. Retrieval touches only 2
	// experts, never all 64. The layer returns the output, the retrieved expert
	// indices per token, and the softmax gate weights.
	p := nn.NewPEER(tensor.F64, 4, 8, 2, 42)
	x := tensor.FromFloat64(tensor.Shape{2, 4}, []float64{1, 0, -1, 0.5, 0.3, 2, -0.5, 1})
	y, idx, gates, _ := p.Forward(backend.NewContext(), x)
	fmt.Println("y:", y.Shape(), "gates:", gates.Shape())
	fmt.Printf("N=%d topK=%d k'=%d heads=%d dk=%d\n",
		p.NumExperts(), p.TopK(), p.SubKeyTopK(), p.Heads(), p.QueryDim())
	fmt.Println("retrieved per token:", len(idx[0]))
	fmt.Println("gate row sums to 1:", math.Abs(gates.AtF64(0, 0)+gates.AtF64(0, 1)-1) < 1e-12)
	fmt.Println("params:", len(p.Params()))
	// Output:
	// y: (2, 4) gates: (2, 2)
	// N=64 topK=2 k'=2 heads=1 dk=4
	// retrieved per token: 2
	// gate row sums to 1: true
	// params: 5
}
