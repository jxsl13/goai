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

// maxsimRef independently computes Σ_i max_j (q̂_i·d̂_j) with L2-normalized tokens.
// §V16 tier-1 parity oracle.
func maxsimRef(q, doc [][]float64) float64 {
	l2 := func(v []float64) []float64 {
		var ss float64
		for _, x := range v {
			ss += x * x
		}
		n := math.Sqrt(ss + 1e-12)
		o := make([]float64, len(v))
		for i, x := range v {
			o[i] = x / n
		}
		return o
	}
	var score float64
	for i := range q {
		qi := l2(q[i])
		best := math.Inf(-1)
		for j := range doc {
			dj := l2(doc[j])
			var dp float64
			for c := range qi {
				dp += qi[c] * dj[c]
			}
			if dp > best {
				best = dp
			}
		}
		score += best
	}
	return score
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2004.12832 §3): MaxSim == the independent
// sum-over-query-tokens of max-over-document-tokens cosine.
func TestMaxSimParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 60))
	q, doc := randMat(rng, 4, 5), randMat(rng, 7, 5)
	got, err := nn.MaxSim(backend.NewContext(), q, doc)
	if err != nil {
		t.Fatal(err)
	}
	want := maxsimRef(toRows(q), toRows(doc))
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("MaxSim=%.10g want %.10g", got.AtF64(), want)
	}
}

// §V2 (defining property): when the document contains the query tokens, every query
// token matches itself with cosine ≈ 1, so the score ≈ Nq (the max achievable).
func TestMaxSimExactMatchScoresNq(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 61))
	q := randMat(rng, 5, 4)
	// doc = the query tokens (plus an extra distractor token) ⇒ each query token has
	// an exact match in the document.
	doc := tensor.New(tensor.F64, tensor.Shape{6, 4})
	for i := 0; i < 5; i++ {
		for c := 0; c < 4; c++ {
			doc.SetF64(q.AtF64(i, c), i, c)
		}
	}
	for c := 0; c < 4; c++ {
		doc.SetF64(rng.NormFloat64(), 5, c) // distractor
	}
	score, err := nn.MaxSim(backend.NewContext(), q, doc)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(score.AtF64()-5) > 1e-6 { // Nq = 5 self-matches at cosine ≈ 1
		t.Errorf("exact-match score=%g, want ≈5 (Nq)", score.AtF64())
	}
}

// §V2: a document containing the query tokens ranks above a random document.
func TestMaxSimRanking(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 62))
	q := randMat(rng, 4, 6)
	pos := tensor.New(tensor.F64, q.Shape()) // pos = query tokens
	for i := range q.Numel() {
		c := tensor.Unravel(i, q.Shape())
		pos.SetF64(q.AtF64(c...), c...)
	}
	neg := randMat(rng, 4, 6)
	sPos, _ := nn.MaxSim(backend.NewContext(), q, pos)
	sNeg, _ := nn.MaxSim(backend.NewContext(), q, neg)
	if sPos.AtF64() <= sNeg.AtF64() {
		t.Errorf("positive doc score %g not > negative %g", sPos.AtF64(), sNeg.AtF64())
	}
}

// §V2: differentiable w.r.t. q and doc (the max routes each query token's gradient to
// its best document token). Finite-diff is valid here — no stop-gradient.
func TestMaxSimGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 63))
	q, doc := randMat(rng, 3, 4), randMat(rng, 5, 4)
	tape := autograd.NewTape()
	score, err := nn.MaxSim(tape.Context(), q, doc)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(score); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		s, _ := nn.MaxSim(backend.NewContext(), q, doc)
		return s.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp := fwd()
			p.SetF64(o-h, coord...)
			lm := fwd()
			p.SetF64(o, coord...)
			if num, ana := (lp-lm)/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("q", q)
	check("doc", doc)
}

func TestMaxSimErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.MaxSim(backend.NewContext(), r(3, 4), r(5, 6)); err == nil {
		t.Error("embedding-dim mismatch: want error")
	}
	if _, err := nn.MaxSim(backend.NewContext(), r(3, 4, 1), r(5, 4)); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleMaxSim() {
	// a 2-token query; the document contains a near-copy of each query token, so both
	// query tokens match at cosine ≈ 1 and the score ≈ 2.
	q := toT([][]float64{{1, 0}, {0, 1}})
	doc := toT([][]float64{{1, 0}, {0, 1}, {-1, -1}})
	s, _ := nn.MaxSim(backend.NewContext(), q, doc)
	fmt.Printf("%.2f\n", s.AtF64())
	// Output:
	// 2.00
}
