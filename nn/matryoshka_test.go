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

// matryoshkaRef independently sums batch-mean softmax cross-entropy over nested
// prefixes z[:, :m]·w[:m, :]. §V16 tier-1 parity oracle.
func matryoshkaRef(z, w [][]float64, y []int, dims []int) float64 {
	batch, C := len(z), len(w[0])
	var total float64
	for _, m := range dims {
		var ce float64
		for b := 0; b < batch; b++ {
			logit := make([]float64, C)
			for c := 0; c < C; c++ {
				for j := 0; j < m; j++ {
					logit[c] += z[b][j] * w[j][c]
				}
			}
			mx := math.Inf(-1)
			for _, l := range logit {
				if l > mx {
					mx = l
				}
			}
			var sum float64
			for _, l := range logit {
				sum += math.Exp(l - mx)
			}
			ce += -(logit[y[b]] - mx - math.Log(sum))
		}
		total += ce / float64(batch) // OpCrossEntropy means over the batch
	}
	return total
}

func targetsVec(idx []int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(idx)})
	for i, c := range idx {
		t.SetF64(float64(c), i)
	}
	return t
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2205.13147 Eq.1): MatryoshkaLoss equals the
// independent sum of nested-prefix cross-entropies (uniform weights, shared head).
func TestMatryoshkaParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 25))
	const batch, d, C = 4, 8, 3
	z, w := randMat(rng, batch, d), randMat(rng, d, C)
	y := []int{0, 2, 1, 2}
	dims := []int{2, 4, 8}
	got, err := nn.MatryoshkaLoss(backend.NewContext(), z, w, targetsVec(y), dims)
	if err != nil {
		t.Fatal(err)
	}
	want := matryoshkaRef(toRows(z), toRows(w), y, dims)
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.10g want %.10g", got.AtF64(), want)
	}
}

// §V2: a single full-width granularity {d} reduces to plain cross-entropy z·w.
func TestMatryoshkaSingleGranularityIsCE(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 26))
	const batch, d, C = 5, 6, 4
	z, w := randMat(rng, batch, d), randMat(rng, d, C)
	tgt := targetsVec([]int{0, 1, 2, 3, 0})
	mrl, err := nn.MatryoshkaLoss(backend.NewContext(), z, w, tgt, []int{d})
	if err != nil {
		t.Fatal(err)
	}
	logits := matmul(t, z, w)
	ce, err := nn.CrossEntropy(backend.NewContext(), logits, tgt)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(mrl.AtF64()-ce.AtF64()) > 1e-12 {
		t.Errorf("MRL({d})=%g != CE=%g", mrl.AtF64(), ce.AtF64())
	}
}

// §V2 (defining nesting property, exact): adding a smaller granularity m to the
// nesting set changes the gradient ONLY on the first m embedding coordinates — the
// coordinates ≥ m are untouched by the m-granularity term. This is why early
// coordinates accumulate more gradient (coarse-to-fine nesting).
func TestMatryoshkaGradientNesting(t *testing.T) {
	rng := rand.New(rand.NewPCG(9, 27))
	const batch, d, C, m = 4, 8, 3, 4
	z, w := randMat(rng, batch, d), randMat(rng, d, C)
	tgt := targetsVec([]int{0, 1, 2, 0})
	gradZ := func(dims []int) *tensor.Tensor {
		tape := autograd.NewTape()
		loss, err := nn.MatryoshkaLoss(tape.Context(), z, w, tgt, dims)
		if err != nil {
			t.Fatal(err)
		}
		if err := tape.Backward(loss); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(z)
	}
	gFull := gradZ([]int{d})    // only the full granularity
	gBoth := gradZ([]int{m, d}) // adds the m-granularity term
	var earlyDiff bool
	for b := 0; b < batch; b++ {
		for j := 0; j < d; j++ {
			diff := math.Abs(gBoth.AtF64(b, j) - gFull.AtF64(b, j))
			if j >= m { // coordinates ≥ m must be identical (m-term doesn't touch them)
				if diff > 1e-12 {
					t.Errorf("grad[%d,%d] (col≥m) changed by %g — m-granularity leaked past m", b, j, diff)
				}
			} else if diff > 1e-9 {
				earlyDiff = true
			}
		}
	}
	if !earlyDiff {
		t.Error("adding the m-granularity did not change any early-coordinate gradient")
	}
}

// §V2: differentiable w.r.t. z and w.
func TestMatryoshkaGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 28))
	const batch, d, C = 3, 6, 3
	z, w := randMat(rng, batch, d), randMat(rng, d, C)
	tgt := targetsVec([]int{0, 2, 1})
	dims := []int{2, 4, 6}
	tape := autograd.NewTape()
	loss, err := nn.MatryoshkaLoss(tape.Context(), z, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.MatryoshkaLoss(backend.NewContext(), z, w, tgt, dims)
		return l.AtF64()
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
	check("z", z)
	check("w", w)
}

func TestMatryoshkaErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	tgt := targetsVec([]int{0, 1})
	if _, err := nn.MatryoshkaLoss(backend.NewContext(), r(2, 8), r(8, 3), tgt, nil); err == nil {
		t.Error("empty nesting: want error")
	}
	if _, err := nn.MatryoshkaLoss(backend.NewContext(), r(2, 8), r(8, 3), tgt, []int{4, 4}); err == nil {
		t.Error("non-increasing nesting: want error")
	}
	if _, err := nn.MatryoshkaLoss(backend.NewContext(), r(2, 8), r(8, 3), tgt, []int{4, 9}); err == nil {
		t.Error("nesting dim > d: want error")
	}
	if _, err := nn.MatryoshkaLoss(backend.NewContext(), r(2, 8), r(6, 3), tgt, []int{4}); err == nil {
		t.Error("z/w dim mismatch: want error")
	}
}

func ExampleMatryoshkaLoss() {
	// one nesting set {2,4}: the loss is CE on the first 2 dims plus CE on the first 4.
	z := toT([][]float64{{2, 0, 0, 0}, {0, 0, 2, 0}})
	w := toT([][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}})
	loss, _ := nn.MatryoshkaLoss(backend.NewContext(), z, w, targetsVec([]int{0, 1}), []int{2, 4})
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.5370
}
