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

	_ "github.com/jxsl13/goai/backend/ref"
)

// stPermuteRows returns a copy of x whose row i is x[perm[i]].
func stPermuteRows(x *tensor.Tensor, perm []int) *tensor.Tensor {
	r, c := x.Shape()[0], x.Shape()[1]
	out := tensor.New(x.Dtype(), tensor.Shape{r, c})
	for i := range r {
		for j := range c {
			out.SetF64(x.AtF64(perm[i], j), i, j)
		}
	}
	return out
}

// stMaxAbsDiff is the max |a−b| over two equal-shaped rank-2 tensors.
func stMaxAbsDiff(a, b *tensor.Tensor) float64 {
	var m float64
	for i := range a.Shape()[0] {
		for j := range a.Shape()[1] {
			if d := math.Abs(a.AtF64(i, j) - b.AtF64(i, j)); d > m {
				m = d
			}
		}
	}
	return m
}

// stScalar makes a [1,1] F64 constant (a regression target that broadcasts
// against a [1,1] prediction).
func stScalar(v float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{1, 1})
	t.SetF64(v, 0, 0)
	return t
}

// stShuffle returns a random permutation of 0..n−1 (not the identity for n>1).
func stShuffle(rng *rand.Rand, n int) []int {
	p := make([]int, n)
	for i := range p {
		p[i] = i
	}
	rng.Shuffle(n, func(i, j int) { p[i], p[j] = p[j], p[i] })
	return p
}

// stGradcheck compares every analytic parameter gradient of a block against
// central finite differences of a sum-of-SQUARES objective (F64; §V16 (3), max
// rel err ~1e-4). A plain sum is degenerate here — the blocks end in LayerNorm,
// whose row sums depend only on β — so the objective squares the output first.
// fwd rebuilds the forward graph through the given context.
func stGradcheck(t *testing.T, params []*tensor.Tensor, names []string, fwd func(ctx *backend.Context) (*tensor.Tensor, error)) {
	t.Helper()
	const h = 1e-6
	tape := autograd.NewTape()
	out, err := fwd(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	sq, err := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{out, out}, nil)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), sq[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	sumLoss := func() float64 {
		o, err := fwd(backend.NewContext())
		if err != nil {
			t.Fatal(err)
		}
		var s float64
		for i := range o.Numel() {
			v := o.AtF64(tensor.Unravel(i, o.Shape())...)
			s += v * v
		}
		return s
	}
	for pi, p := range params {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — no gradient reached this parameter", names[pi])
		}
		var anaNZ, numNZ bool
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := sumLoss()
			p.SetF64(o-h, coord...)
			lm := sumLoss()
			p.SetF64(o, coord...)
			num, ana := (lp-lm)/(2*h), g.AtF64(coord...)
			if math.Abs(ana) > 1e-9 {
				anaNZ = true
			}
			if math.Abs(num) > 1e-7 {
				numNZ = true
			}
			if math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", names[pi], coord, num, ana)
			}
		}
		// A parameter whose gradient is genuinely zero everywhere is fine ONLY if
		// the numeric gradient agrees (a true null direction — e.g. the key-
		// projection bias, which a softmax over keys cancels). Analytic-zero while
		// numeric-nonzero means the parameter fell off the tape.
		if numNZ && !anaNZ {
			t.Errorf("%s: analytic grads all zero but numeric nonzero — parameter not on the tape", names[pi])
		}
	}
}

// §V16 (1) THE DEFINING PROPERTY — PERMUTATION INVARIANCE: the PMA-pooled output
// of a full Set Transformer is unchanged (to 1e-10) when the input set's rows are
// permuted. This is what makes the encoder a set function rather than a sequence
// function.
func TestSetTransformerPermutationInvariance(t *testing.T) {
	const dim, heads, n = 8, 2, 6
	st, err := nn.NewSetTransformer(tensor.F64, dim, heads, 7, nn.WithSetTransformerDepth(2))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(1, 2))
	x := randMat(rng, n, dim)
	out, err := st.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{1, dim}) {
		t.Fatalf("pooled shape %v, want [1 %d]", out.Shape(), dim)
	}
	for trial := range 4 {
		perm := stShuffle(rng, n)
		outP, err := st.Forward(backend.NewContext(), stPermuteRows(x, perm))
		if err != nil {
			t.Fatal(err)
		}
		if d := stMaxAbsDiff(out, outP); d > 1e-10 {
			t.Errorf("trial %d: PMA output not permutation-invariant, maxdiff %.3g > 1e-10", trial, d)
		}
	}
}

// §V16 (1) PERMUTATION EQUIVARIANCE: permuting the input rows of a SAB permutes
// its output rows the same way (to 1e-10) — SAB(Px) = P·SAB(X).
func TestSABPermutationEquivariance(t *testing.T) {
	const dim, heads, n = 8, 2, 6
	sab, err := nn.NewSAB(tensor.F64, dim, heads, 11)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(3, 4))
	x := randMat(rng, n, dim)
	out, err := sab.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{n, dim}) {
		t.Fatalf("SAB shape %v, want [%d %d]", out.Shape(), n, dim)
	}
	for trial := range 4 {
		perm := stShuffle(rng, n)
		outP, err := sab.Forward(backend.NewContext(), stPermuteRows(x, perm))
		if err != nil {
			t.Fatal(err)
		}
		want := stPermuteRows(out, perm) // P·SAB(X)
		if d := stMaxAbsDiff(want, outP); d > 1e-10 {
			t.Errorf("trial %d: SAB not equivariant, maxdiff %.3g > 1e-10", trial, d)
		}
	}
}

// §V16 (2) ISAB — shape/mechanism + equivariance: ISAB maps [n,dim]→[n,dim] via
// two O(n·m) MABs through m inducing points, is permutation-EQUIVARIANT like a
// full SAB, and with m ≥ n it stands in for a full SAB (both are equivariant
// [n,dim]→[n,dim] maps here); the inducing-point count m materially changes the
// map (mechanism), confirming the inducing points participate.
func TestISABShapeEquivarianceMechanism(t *testing.T) {
	const dim, heads, n = 8, 2, 6
	rng := rand.New(rand.NewPCG(5, 6))
	x := randMat(rng, n, dim)
	for _, m := range []int{2, n, n + 3} { // m<n, m==n, m>n (m≥n ≈ full SAB capacity)
		isab, err := nn.NewISAB(tensor.F64, dim, heads, m, 13)
		if err != nil {
			t.Fatal(err)
		}
		out, err := isab.Forward(backend.NewContext(), x)
		if err != nil {
			t.Fatal(err)
		}
		if !out.Shape().Equal(tensor.Shape{n, dim}) {
			t.Fatalf("m=%d: ISAB shape %v, want [%d %d]", m, out.Shape(), n, dim)
		}
		perm := stShuffle(rng, n)
		outP, err := isab.Forward(backend.NewContext(), stPermuteRows(x, perm))
		if err != nil {
			t.Fatal(err)
		}
		if d := stMaxAbsDiff(stPermuteRows(out, perm), outP); d > 1e-10 {
			t.Errorf("m=%d: ISAB not equivariant, maxdiff %.3g > 1e-10", m, d)
		}
	}
	// Mechanism: changing m (with everything else fixed) changes the output — the
	// inducing points are not a no-op bottleneck.
	iA, _ := nn.NewISAB(tensor.F64, dim, heads, 2, 13)
	iB, _ := nn.NewISAB(tensor.F64, dim, heads, 5, 13)
	oA, _ := iA.Forward(backend.NewContext(), x)
	oB, _ := iB.Forward(backend.NewContext(), x)
	if stMaxAbsDiff(oA, oB) < 1e-9 {
		t.Error("ISAB output independent of m — inducing points not participating")
	}
}

// §V16 (1) PMA invariance + shape: PMA pools [n,dim]→[k,dim] and is invariant to
// permutations of the input set (the pooled summary does not depend on order).
func TestPMAInvariance(t *testing.T) {
	const dim, heads, n, k = 8, 2, 7, 3
	pma, err := nn.NewPMA(tensor.F64, dim, heads, k, 17)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(7, 8))
	x := randMat(rng, n, dim)
	out, err := pma.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{k, dim}) {
		t.Fatalf("PMA shape %v, want [%d %d]", out.Shape(), k, dim)
	}
	for trial := range 4 {
		perm := stShuffle(rng, n)
		outP, err := pma.Forward(backend.NewContext(), stPermuteRows(x, perm))
		if err != nil {
			t.Fatal(err)
		}
		if d := stMaxAbsDiff(out, outP); d > 1e-10 {
			t.Errorf("trial %d: PMA not permutation-invariant, maxdiff %.3g > 1e-10", trial, d)
		}
	}
}

// §V16 (3) GRADCHECK through MAB: central finite differences confirm gradients
// reach every parameter of the cross-attention block.
func TestMABGradcheck(t *testing.T) {
	const dim, heads, sq, sk = 4, 2, 3, 4
	mab, err := nn.NewMAB(tensor.F64, dim, heads, 3)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(9, 10))
	x := randMat(rng, sq, dim)
	y := randMat(rng, sk, dim)
	params := mab.Params()
	names := make([]string, len(params))
	for i := range names {
		names[i] = fmt.Sprintf("MAB.p%d", i)
	}
	stGradcheck(t, params, names, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return mab.Forward(ctx, x, y)
	})
}

// §V16 (3) GRADCHECK through ISAB — INCLUDING the learnable inducing points:
// gradients reach both MABs and the inducing-point tensor.
func TestISABGradcheck(t *testing.T) {
	const dim, heads, m, n = 4, 2, 3, 5
	isab, err := nn.NewISAB(tensor.F64, dim, heads, m, 19)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(11, 12))
	x := randMat(rng, n, dim)
	params := isab.Params()
	names := make([]string, len(params))
	for i := range names {
		names[i] = fmt.Sprintf("ISAB.p%d", i)
	}
	names[len(names)-1] = "ISAB.Inducer" // last param is the inducing points
	stGradcheck(t, params, names, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return isab.Forward(ctx, x)
	})
}

// §V16 (3) GRADCHECK through PMA — INCLUDING the learnable seed vectors.
func TestPMAGradcheck(t *testing.T) {
	const dim, heads, k, n = 4, 2, 2, 5
	pma, err := nn.NewPMA(tensor.F64, dim, heads, k, 23)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(13, 14))
	x := randMat(rng, n, dim)
	params := pma.Params()
	names := make([]string, len(params))
	for i := range names {
		names[i] = fmt.Sprintf("PMA.p%d", i)
	}
	names[len(names)-1] = "PMA.Seeds"
	stGradcheck(t, params, names, func(ctx *backend.Context) (*tensor.Tensor, error) {
		return pma.Forward(ctx, x)
	})
}

// §V16 (4) THE VALUE — a permutation-invariant set-regression task: predict the
// MEAN of the sets' first coordinate. The Set Transformer pools each set to a
// single vector, a linear head reads out the scalar, and MSE is minimised over a
// small batch. The loss must fall sharply, and the trained model's prediction
// must stay permutation-invariant.
func TestSetTransformerValue(t *testing.T) {
	const dim, heads, n, batch, steps = 16, 2, 5, 8, 200
	st, err := nn.NewSetTransformer(tensor.F64, dim, heads, 101, nn.WithSetTransformerDepth(2))
	if err != nil {
		t.Fatal(err)
	}
	head := nn.NewLinear(tensor.F64, dim, 1, 202) // pooled [1,dim] → scalar
	rng := rand.New(rand.NewPCG(31, 41))
	sets := make([]*tensor.Tensor, batch)
	targets := make([]float64, batch)
	for b := range batch {
		sets[b] = randMat(rng, n, dim)
		var s float64
		for i := range n {
			s += sets[b].AtF64(i, 0)
		}
		targets[b] = s / float64(n) // mean of the first coordinate — permutation-invariant
	}
	params := append(st.Params(), head.Params()...)
	opt := nn.NewAdam(params, 5e-3)

	predict := func(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
		pooled, err := st.Forward(ctx, x) // [1,dim]
		if err != nil {
			return nil, err
		}
		return head.Forward(ctx, pooled) // [1,1]
	}
	batchLoss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		var total *tensor.Tensor
		for b := range batch {
			p, err := predict(ctx, sets[b])
			if err != nil {
				return nil, err
			}
			tgt := stScalar(targets[b])
			diffOut, err := backend.Execute(ctx, backend.OpSub, []*tensor.Tensor{p, tgt}, nil)
			if err != nil {
				return nil, err
			}
			sqOut, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{diffOut[0], diffOut[0]}, nil)
			if err != nil {
				return nil, err
			}
			if total == nil {
				total = sqOut[0]
			} else {
				addOut, err := backend.Execute(ctx, backend.OpAdd, []*tensor.Tensor{total, sqOut[0]}, nil)
				if err != nil {
					return nil, err
				}
				total = addOut[0]
			}
		}
		return total, nil
	}
	var first, last float64
	for step := range steps {
		tape := autograd.NewTape()
		loss, err := batchLoss(tape.Context())
		if err != nil {
			t.Fatal(err)
		}
		lv := loss.AtF64(0, 0)
		if step == 0 {
			first = lv
		}
		last = lv
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		if err := opt.Step(tape.Grad); err != nil {
			t.Fatal(err)
		}
	}
	t.Logf("SetTransformer set-mean task: MSE %.5f → %.5f", first, last)
	if last >= first*0.2 {
		t.Fatalf("set-regression did not train: MSE %.5f → %.5f", first, last)
	}
	// The trained predictor is still permutation-invariant.
	perm := stShuffle(rng, n)
	p0, err := predict(backend.NewContext(), sets[0])
	if err != nil {
		t.Fatal(err)
	}
	pP, err := predict(backend.NewContext(), stPermuteRows(sets[0], perm))
	if err != nil {
		t.Fatal(err)
	}
	if d := math.Abs(p0.AtF64(0, 0) - pP.AtF64(0, 0)); d > 1e-9 {
		t.Errorf("trained prediction not permutation-invariant, diff %.3g", d)
	}
}

// §V16 (5) SANITY: the constructors reject bad shapes and Forward rejects wrong
// input widths/ranks.
func TestSetTransformerErrors(t *testing.T) {
	if _, err := nn.NewMAB(tensor.F64, 7, 2, 1); err == nil {
		t.Error("MAB dim%heads != 0: want error")
	}
	if _, err := nn.NewISAB(tensor.F64, 8, 2, 0, 1); err == nil {
		t.Error("ISAB m=0: want error")
	}
	if _, err := nn.NewPMA(tensor.F64, 8, 2, 0, 1); err == nil {
		t.Error("PMA k=0: want error")
	}
	if _, err := nn.NewSetTransformer(tensor.F64, 8, 3, 1); err == nil {
		t.Error("SetTransformer dim%heads != 0: want error")
	}
	st, _ := nn.NewSetTransformer(tensor.F64, 8, 2, 1)
	if _, err := st.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 5})); err == nil {
		t.Error("wrong input width: want error")
	}
	if _, err := st.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 3, 8})); err == nil {
		t.Error("rank-3 input: want error")
	}
	pma, _ := nn.NewPMA(tensor.F64, 8, 2, 1, 1)
	if _, err := pma.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{4, 5})); err == nil {
		t.Error("PMA wrong width: want error")
	}
}

// §V16 (2) ISAB efficiency variant wired into the full encoder: switching the
// encoder to ISAB blocks preserves permutation invariance of the pooled output.
func TestSetTransformerISABInvariance(t *testing.T) {
	const dim, heads, n = 8, 2, 9
	st, err := nn.NewSetTransformer(tensor.F64, dim, heads, 55,
		nn.WithSetTransformerInducing(3), nn.WithSetTransformerDepth(2), nn.WithSetTransformerSeeds(2))
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewPCG(15, 16))
	x := randMat(rng, n, dim)
	out, err := st.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	if !out.Shape().Equal(tensor.Shape{2, dim}) {
		t.Fatalf("ISAB-encoder pooled shape %v, want [2 %d]", out.Shape(), dim)
	}
	perm := stShuffle(rng, n)
	outP, err := st.Forward(backend.NewContext(), stPermuteRows(x, perm))
	if err != nil {
		t.Fatal(err)
	}
	if d := stMaxAbsDiff(out, outP); d > 1e-10 {
		t.Errorf("ISAB-encoder pooled output not permutation-invariant, maxdiff %.3g", d)
	}
}

func ExampleMAB() {
	// A Multihead Attention Block: query set X (2 rows) attends to key/value set
	// Y (3 rows), producing one output row per query row.
	mab, _ := nn.NewMAB(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{2, 4})
	y := tensor.New(tensor.F64, tensor.Shape{3, 4})
	out, _ := mab.Forward(backend.NewContext(), x, y)
	fmt.Printf("%v params=%d\n", out.Shape(), len(mab.Params()))
	// Output: (2, 4) params=16
}

func ExampleSAB() {
	// A Set Attention Block is self-attention over a set: SAB(X)=MAB(X,X).
	sab, _ := nn.NewSAB(tensor.F64, 4, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{3, 4})
	out, _ := sab.Forward(backend.NewContext(), x)
	fmt.Printf("%v params=%d\n", out.Shape(), len(sab.Params()))
	// Output: (3, 4) params=16
}

func ExampleISAB() {
	// An Induced Set Attention Block with 2 inducing points: O(n·m) attention
	// over a set of 5 elements, still permutation-equivariant.
	isab, _ := nn.NewISAB(tensor.F64, 4, 2, 2, 1)
	x := tensor.New(tensor.F64, tensor.Shape{5, 4})
	out, _ := isab.Forward(backend.NewContext(), x)
	fmt.Printf("%v\n", out.Shape())
	// Output: (5, 4)
}

func ExamplePMA() {
	// Pooling by Multihead Attention with 1 seed: a permutation-invariant pooled
	// summary of a 6-element set.
	pma, _ := nn.NewPMA(tensor.F64, 4, 2, 1, 1)
	x := tensor.New(tensor.F64, tensor.Shape{6, 4})
	out, _ := pma.Forward(backend.NewContext(), x)
	fmt.Printf("%v\n", out.Shape())
	// Output: (1, 4)
}

func ExampleSetTransformer() {
	// A Set Transformer encoder: 2 SAB blocks + a single-seed PMA head map a set
	// of 5 vectors to one permutation-invariant summary vector.
	st, _ := nn.NewSetTransformer(tensor.F64, 8, 2, 1, nn.WithSetTransformerDepth(2))
	x := tensor.New(tensor.F64, tensor.Shape{5, 8})
	out, _ := st.Forward(backend.NewContext(), x)
	fmt.Printf("%v\n", out.Shape())
	// Output: (1, 8)
}
