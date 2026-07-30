package classic

import (
	"math"
	"testing"
)

// TestSoftmaxPredictProbaMatchesReference and TestSoftmaxPredictProbaBitStable gate
// SoftmaxRegression.PredictProba, which had NO output-value test.
//
// The absence was total: perturbing either the input fill or the copied-out probabilities by one
// ulp left the whole classic package green. That mattered because both of those loops were rewritten
// from per-element accessors to contiguous copies, a 2x change to a function nothing checked.
//
// Two tests, because one oracle cannot do both jobs.
//
// The reference test states what the function MEANS, independently of how it computes it: softmax
// of x*W + B, evaluated here in plain Go from the model's own exported weights rather than through
// the backend's matmul/add-bias/softmax chain. That makes it a real second implementation, so it
// catches an index or ordering mistake in either the fill or the copy-out. It CANNOT be bit-exact —
// a hand-rolled dot product associates the sum differently than the backend kernel, and the
// reference subtracts its own row max for stability — so it runs at a tolerance.
//
// The golden pins the exact bits, which is what a tolerance cannot see. It is a stability gate, not
// a correctness one; the reference test above is what makes the frozen value trustworthy. The split
// is not theoretical: perturbing the input fill by one ulp reddens ONLY the golden, because the
// difference reaching the output is ~1e-16 and the tolerance absorbs it.
//
// One measured caveat on the reference test, so its sensitivity is not overread. Flipping the low
// mantissa bit of a copied-out probability does redden it, but via the `got > 1` range check rather
// than the tolerance: the k=7 geometry saturates 30 of its entries to exactly 1.0, and a flip there
// leaves 1.0000000000000002. That is a property of the fixture, not evidence that the tolerance sees
// single bits.
func TestSoftmaxPredictProbaMatchesReference(t *testing.T) {
	for _, g := range []struct{ n, d, k int }{{200, 8, 4}, {64, 3, 2}, {97, 12, 7}} {
		x, y := softmaxSynthetic(g.n, g.d, g.k, int64(g.k)*7+1)
		m := &SoftmaxRegression{}
		if err := m.Fit(x, y, g.k, 40, 0.1); err != nil {
			t.Fatal(err)
		}
		p, err := m.PredictProba(x)
		if err != nil {
			t.Fatal(err)
		}
		if len(p) != g.n {
			t.Fatalf("%+v: %d rows, want %d", g, len(p), g.n)
		}
		for i := range p {
			if len(p[i]) != g.k {
				t.Fatalf("%+v: row %d width %d, want %d", g, i, len(p[i]), g.k)
			}
			// Independent reference: logits from W and B, then a max-shifted softmax.
			logit := make([]float64, g.k)
			hi := math.Inf(-1)
			for c := range g.k {
				s := m.B.AtF64(c)
				for j := range g.d {
					s += x[i][j] * m.W.AtF64(j, c)
				}
				logit[c] = s
				hi = max(hi, s)
			}
			var tot float64
			for c := range g.k {
				logit[c] = math.Exp(logit[c] - hi)
				tot += logit[c]
			}
			var sum float64
			for c := range g.k {
				want := logit[c] / tot
				got := p[i][c]
				if got < 0 || got > 1 || math.IsNaN(got) {
					t.Fatalf("%+v: [%d][%d] = %v is not a probability", g, i, c, got)
				}
				if math.Abs(got-want) > 1e-12 {
					t.Fatalf("%+v: [%d][%d] = %v, reference says %v (diff %g) — the fill or the "+
						"copy-out moved a value", g, i, c, got, want, got-want)
				}
				sum += got
			}
			if math.Abs(sum-1) > 1e-12 {
				t.Fatalf("%+v: row %d sums to %v, want 1", g, i, sum)
			}
		}
	}
}

// TestSoftmaxPredictProbaBitStable freezes the exact output bits for a fixed fixture. Any change to
// the input fill or the copy-out — including one ulp, which the tolerance above cannot see — moves
// the hash.
func TestSoftmaxPredictProbaBitStable(t *testing.T) {
	const n, d, k = 64, 6, 3
	const wantHash uint64 = 0x241257d779b1f31f
	x, y := softmaxSynthetic(n, d, k, 5)
	m := &SoftmaxRegression{}
	if err := m.Fit(x, y, k, 20, 0.1); err != nil {
		t.Fatal(err)
	}
	p, err := m.PredictProba(x)
	if err != nil {
		t.Fatal(err)
	}
	var h uint64 = 14695981039346656037
	for _, row := range p {
		for _, v := range row {
			h = (h ^ math.Float64bits(v)) * 1099511628211
		}
	}
	if wantHash == 0 {
		t.Fatalf("CAPTURE: set wantHash to %#x", h)
	}
	if h != wantHash {
		t.Fatalf("PredictProba output hash %#x, want %#x — the fill or the copy-out changed a bit",
			h, wantHash)
	}
}
