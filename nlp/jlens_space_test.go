package nlp_test

import (
	"math"
	"testing"

	"github.com/jxsl13/goai/backend"
	_ "github.com/jxsl13/goai/backend/ref"
	"github.com/jxsl13/goai/nlp"
	"github.com/jxsl13/goai/tensor"
)

// jspaceLens builds a lens whose J matrices are all the identity, so the
// J-space dictionary directions reduce to the (normalised) columns of the
// model's unembedding matrix — the synthetic setting where recovery is exact.
func jspaceLens(dim, layers int) *nlp.JLens {
	jl := &nlp.JLens{Dim: dim, Layers: layers, Weight: 1}
	for range layers + 1 {
		eye := tensor.New(tensor.F64, tensor.Shape{dim, dim})
		for i := range dim {
			eye.SetF64(1, i, i)
		}
		jl.J = append(jl.J, eye)
	}
	return jl
}

// jspaceModel builds a GPT whose Head is the given [dim, vocab] matrix — the
// only part of the model JSpaceDecompose consults.
func jspaceModel(t *testing.T, head *tensor.Tensor) *nlp.GPT {
	t.Helper()
	s := head.Shape()
	g := jlensGPT(t, s[1], 8, s[0], 2, 1)
	g.Head = head
	return g
}

// §T813 (a), synthetic exactness: an activation constructed as a nonnegative
// combination of m < k known orthonormal dictionary directions is recovered
// exactly — the greedy pursuit finds the support in descending-coefficient
// order, the coefficients to f64 round-off, and a ≈ 0 residual; the
// random-direction control then stops it at occupancy m.
func TestJSpaceDecomposeSyntheticExact(t *testing.T) {
	const dim, vocab = 8, 8
	head := tensor.New(tensor.F64, tensor.Shape{dim, vocab})
	for v := range vocab {
		head.SetF64(3, v, v) // scaled identity: d_v = e_v after normalisation
	}
	model := jspaceModel(t, head)
	lens := jspaceLens(dim, 1)

	h := tensor.New(tensor.F64, tensor.Shape{dim})
	h.SetF64(2.5, 1)
	h.SetF64(1.5, 4)
	h.SetF64(0.5, 6)

	dec, err := nlp.JSpaceDecompose(lens, model, h, 0, 25)
	if err != nil {
		t.Fatal(err)
	}
	want := []struct {
		tok  int
		coef float64
	}{{1, 2.5}, {4, 1.5}, {6, 0.5}}
	if len(dec.Components) != len(want) {
		t.Fatalf("recovered %d components, want %d: %+v", len(dec.Components), len(want), dec.Components)
	}
	for i, w := range want {
		c := dec.Components[i]
		if c.Token != w.tok {
			t.Fatalf("component %d token %d, want %d (greedy order = descending coefficient)", i, c.Token, w.tok)
		}
		if math.Abs(c.Coef-w.coef) > 1e-12 {
			t.Fatalf("component %d coefficient %g, want %g", i, c.Coef, w.coef)
		}
	}
	if dec.Occupancy != 3 {
		t.Fatalf("occupancy %d, want 3", dec.Occupancy)
	}
	if dec.ResidualNorm > 1e-12 {
		t.Fatalf("residual %g after exact recovery, want ≈ 0", dec.ResidualNorm)
	}
	for b := range dim {
		if math.Abs(dec.Reconstruction[b]-h.AtF64(b)) > 1e-12 {
			t.Fatalf("reconstruction[%d] = %g, want %g", b, dec.Reconstruction[b], h.AtF64(b))
		}
	}
	// A [1, dim] row (as sliced from a capture) decomposes identically.
	row := tensor.New(tensor.F64, tensor.Shape{1, dim})
	for b := range dim {
		row.SetF64(h.AtF64(b), 0, b)
	}
	dec2, err := nlp.JSpaceDecompose(lens, model, row, 0, 25)
	if err != nil {
		t.Fatal(err)
	}
	if dec2.Occupancy != dec.Occupancy || dec2.ResidualNorm != dec.ResidualNorm {
		t.Fatalf("[1,dim] row decomposition differs from [dim] vector")
	}
}

// §T813 (b), occupancy control: with a small dictionary (8 directions) in a
// larger space (dim 32), isotropic noise has near-zero occupancy — the best
// dictionary gain does not beat the best of 64 random unit directions — while
// a planted sparse combination of m = 3 dictionary directions has occupancy
// exactly m: its components tower over the control, and once they are removed
// the pursuit stops immediately.
func TestJSpaceOccupancyControl(t *testing.T) {
	const dim, vocab = 32, 8
	head := tensor.New(tensor.F64, tensor.Shape{dim, vocab})
	for v := range vocab {
		head.SetF64(2, 4*v, v) // d_v = e_{4v}: orthonormal, sparse in the big space
	}
	model := jspaceModel(t, head)
	lens := jspaceLens(dim, 1)
	strict := nlp.WithJSpaceControlDirections(64)

	// Planted combination: occupancy == m and exact support.
	h := tensor.New(tensor.F64, tensor.Shape{dim})
	h.SetF64(3, 0)  // 3·d_0
	h.SetF64(2, 12) // 2·d_3
	h.SetF64(1, 20) // 1·d_5
	dec, err := nlp.JSpaceDecompose(lens, model, h, 0, 25, strict)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Occupancy != 3 {
		t.Fatalf("planted m=3 combo: occupancy %d, want 3 (%+v)", dec.Occupancy, dec.Components)
	}
	gotSupport := map[int]bool{}
	for _, c := range dec.Components {
		gotSupport[c.Token] = true
	}
	for _, tok := range []int{0, 3, 5} {
		if !gotSupport[tok] {
			t.Fatalf("planted token %d missing from support %+v", tok, dec.Components)
		}
	}

	// Isotropic noise: occupancy collapses to ≈ 0.
	state := uint64(2)
	gauss := func() float64 { // deterministic Box-Muller over an LCG
		state = state*6364136223846793005 + 1442695040888963407
		u1 := float64(state>>11)/9007199254740992 + 1e-12
		state = state*6364136223846793005 + 1442695040888963407
		u2 := float64(state>>11) / 9007199254740992
		return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
	}
	total := 0
	for trial := range 10 {
		noise := tensor.New(tensor.F64, tensor.Shape{dim})
		for b := range dim {
			noise.SetF64(gauss(), b)
		}
		nd, err := nlp.JSpaceDecompose(lens, model, noise, 0, 25, strict, nlp.WithJSpaceSeed(uint64(trial+1)))
		if err != nil {
			t.Fatal(err)
		}
		if nd.Occupancy > 2 {
			t.Fatalf("noise trial %d: occupancy %d, want ≤ 2 (control not biting)", trial, nd.Occupancy)
		}
		total += nd.Occupancy
	}
	if total > 5 {
		t.Fatalf("mean noise occupancy %.1f over 10 trials, want ≤ 0.5 (planted m=3 must stay separable)", float64(total)/10)
	}
	t.Logf("occupancy: planted m=3 → %d, noise mean → %.2f", dec.Occupancy, float64(total)/10)
}

// §T813 (c), real activations: decomposing a captured residual of the §T812
// golden model (reference-fitted lens from testdata/jlens_ref.pt) must give a
// strictly decreasing residual curve, and the cosine between reconstruction
// and activation must improve as components are added (k=25 vs k=1). Token
// identities are not asserted (tier-2 — the reference implementation ships no
// decomposition to golden against), only reconstruction properties.
func TestJSpaceDecomposeGoldenModel(t *testing.T) {
	model := jlensGoldenModel(t)
	lens, err := nlp.JLensFromPT("testdata/jlens_ref.pt")
	if err != nil {
		t.Fatalf("JLensFromPT (run scripts/golden_jlens.py): %v", err)
	}
	probe := jlensGoldenTokens(t, "testdata/jlens_probe.npy")[0]

	const layer = 2 // mid-stack fitted layer (GoAI convention)
	var hl *tensor.Tensor
	if _, err := model.ForwardResiduals(backend.NewContext(), probe, func(l int, h *tensor.Tensor) {
		if l == layer {
			hl = h
		}
	}); err != nil {
		t.Fatal(err)
	}
	pos := len(probe) - 1
	h := tensor.New(tensor.F64, tensor.Shape{lens.Dim})
	for b := range lens.Dim {
		h.SetF64(hl.AtF64(pos, b), b)
	}

	dec, err := nlp.JSpaceDecompose(lens, model, h, layer, 25)
	if err != nil {
		t.Fatal(err)
	}
	if len(dec.Components) == 0 {
		t.Fatal("no components accepted on a real residual")
	}
	// Deep pursuit: relax the occupancy control (4 random directions instead
	// of the matched vocabulary count) so the pursuit keeps adding components
	// and the monotonicity/cosine properties are exercised over many steps.
	deep, err := nlp.JSpaceDecompose(lens, model, h, layer, 25, nlp.WithJSpaceControlDirections(4))
	if err != nil {
		t.Fatal(err)
	}
	if len(deep.ResidualCurve) < 3 {
		t.Fatalf("deep pursuit accepted only %d steps — cannot exercise monotonicity", len(deep.ResidualCurve))
	}
	top := dec.Components[0]
	if top.Token < 0 || top.Token >= 32 || top.Coef <= 0 {
		t.Fatalf("top component implausible: %+v", top)
	}
	for _, d := range []*nlp.JSpaceDecomposition{dec, deep} {
		prev := d.InputNorm
		for i, rn := range d.ResidualCurve {
			if rn >= prev {
				t.Fatalf("residual curve not strictly decreasing at step %d: %g → %g", i, prev, rn)
			}
			prev = rn
		}
		if d.ResidualNorm >= d.InputNorm {
			t.Fatalf("residual %g did not shrink below input norm %g", d.ResidualNorm, d.InputNorm)
		}
	}

	cos := func(rec []float64, h *tensor.Tensor) float64 {
		var dot, n2 float64
		for b, v := range rec {
			dot += v * h.AtF64(b)
			n2 += v * v
		}
		if n2 == 0 {
			return 0
		}
		var h2 float64
		for b := range rec {
			h2 += h.AtF64(b) * h.AtF64(b)
		}
		return dot / math.Sqrt(n2*h2)
	}
	dec1, err := nlp.JSpaceDecompose(lens, model, h, layer, 1, nlp.WithJSpaceControlDirections(4))
	if err != nil {
		t.Fatal(err)
	}
	c1, cDeep := cos(dec1.Reconstruction, h), cos(deep.Reconstruction, h)
	if cDeep <= c1 {
		t.Fatalf("cosine did not improve with more components: k=1 → %.4f, deep → %.4f", c1, cDeep)
	}
	t.Logf("golden residual: occupancy %d (deep pursuit %d steps), relative residual %.3f (deep %.3f), cosine k=1 %.4f → deep %.4f",
		dec.Occupancy, len(deep.ResidualCurve), dec.ResidualNorm/dec.InputNorm, deep.ResidualNorm/deep.InputNorm, c1, cDeep)
}
