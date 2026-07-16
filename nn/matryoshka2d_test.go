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

// §V16 anchor 1 (COLLAPSE): Matryoshka2DLoss with a SINGLE layer, the full dims
// list and default (uniform-1) weights is identical to the 1D MatryoshkaLoss on
// that layer's embedding — 2D adds ONLY the layer axis. @1e-10.
func TestMatryoshka2DCollapsesTo1D(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 29))
	const batch, d, C = 4, 8, 3
	z, w := randMat(rng, batch, d), randMat(rng, d, C)
	tgt := targetsVec([]int{0, 2, 1, 2})
	dims := []int{2, 4, 8}
	got, err := nn.Matryoshka2DLoss(backend.NewContext(), []*tensor.Tensor{z}, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	want, err := nn.MatryoshkaLoss(backend.NewContext(), z, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.AtF64()-want.AtF64()) > 1e-10 {
		t.Errorf("2D single-layer=%.12g != 1D MatryoshkaLoss=%.12g", got.AtF64(), want.AtF64())
	}
}

// matryoshka2DRef independently sums w_l·w_d·CE over the (layer × dim) grid. §V16
// tier-1 parity oracle for the structure invariant.
func matryoshka2DRef(reps [][][]float64, w [][]float64, y, dims []int, lw, dw []float64) float64 {
	var total float64
	for l, z := range reps {
		for di, m := range dims {
			total += lw[l] * dw[di] * ceAtDim(z, w, y, m)
		}
	}
	return total
}

// ceAtDim is batch-mean softmax CE of z[:, :m]·w[:m, :] against class targets y.
func ceAtDim(z, w [][]float64, y []int, m int) float64 {
	batch, C := len(z), len(w[0])
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
	return ce / float64(batch)
}

// §V16 anchor 3 (STRUCTURE): the loss equals Σ over the (layer × dim) grid of the
// weighted base losses — verified against a hand-computed sum @1e-10, with
// non-uniform layer AND dim weights.
func TestMatryoshka2DGridStructure(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 30))
	const batch, d, C = 3, 6, 4
	dims := []int{2, 4, 6}
	reps := []*tensor.Tensor{randMat(rng, batch, d), randMat(rng, batch, d), randMat(rng, batch, d)}
	w := randMat(rng, d, C)
	y := []int{0, 3, 1}
	lw := []float64{0.5, 1.0, 2.0}
	dw := []float64{0.25, 1.5, 3.0}
	got, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, targetsVec(y), dims,
		nn.WithMatryoshka2DLayerWeights(lw...), nn.WithMatryoshka2DDimWeights(dw...))
	if err != nil {
		t.Fatal(err)
	}
	rrows := make([][][]float64, len(reps))
	for i, r := range reps {
		rrows[i] = toRows(r)
	}
	want := matryoshka2DRef(rrows, toRows(w), y, dims, lw, dw)
	if math.Abs(got.AtF64()-want) > 1e-10 {
		t.Errorf("2D loss=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V16 anchor 3 (STRUCTURE, zero-weighting): zero-weighting a layer drops its
// contribution EXACTLY — the loss equals the same run with that layer removed.
func TestMatryoshka2DZeroLayerWeightDropsExactly(t *testing.T) {
	rng := rand.New(rand.NewPCG(13, 31))
	const batch, d, C = 3, 6, 3
	dims := []int{3, 6}
	reps := []*tensor.Tensor{randMat(rng, batch, d), randMat(rng, batch, d), randMat(rng, batch, d)}
	w := randMat(rng, d, C)
	tgt := targetsVec([]int{0, 1, 2})
	// Zero out the middle layer.
	withZero, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, dims,
		nn.WithMatryoshka2DLayerWeights(1, 0, 1))
	if err != nil {
		t.Fatal(err)
	}
	// Same two surviving layers, dropped from the list entirely.
	dropped, err := nn.Matryoshka2DLoss(backend.NewContext(), []*tensor.Tensor{reps[0], reps[2]}, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(withZero.AtF64()-dropped.AtF64()) > 1e-10 {
		t.Errorf("zero-weighted layer=%.12g != dropped layer=%.12g", withZero.AtF64(), dropped.AtF64())
	}
	// Symmetrically, a zero DIM weight drops that truncation width at every layer.
	withZeroDim, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{3, 6},
		nn.WithMatryoshka2DDimWeights(0, 1))
	if err != nil {
		t.Fatal(err)
	}
	droppedDim, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{6})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(withZeroDim.AtF64()-droppedDim.AtF64()) > 1e-10 {
		t.Errorf("zero-weighted dim=%.12g != dropped dim=%.12g", withZeroDim.AtF64(), droppedDim.AtF64())
	}
}

// §V16 anchor 2 (GRADCHECK): gradient flows to EVERY provided layer rep (not just
// the last) and to w; finite-difference gradcheck confirms the analytic grads.
func TestMatryoshka2DGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(14, 32))
	const batch, d, C = 3, 6, 3
	dims := []int{2, 4, 6}
	reps := []*tensor.Tensor{randMat(rng, batch, d), randMat(rng, batch, d), randMat(rng, batch, d)}
	w := randMat(rng, d, C)
	tgt := targetsVec([]int{0, 2, 1})
	lw := []float64{1.5, 0.75, 2.0}
	opt := nn.WithMatryoshka2DLayerWeights(lw...)

	tape := autograd.NewTape()
	loss, err := nn.Matryoshka2DLoss(tape.Context(), reps, w, tgt, dims, opt)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, dims, opt)
		return l.AtF64()
	}
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad — gradient did not reach this parameter", name)
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
	// Assert grads reach ALL layers, not only the last.
	for l, rep := range reps {
		check(fmt.Sprintf("layerReps[%d]", l), rep)
	}
	check("w", w)
}

// §V16 anchor 4 (determinism): identical inputs give a bit-identical loss.
func TestMatryoshka2DDeterminism(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 33))
	const batch, d, C = 4, 8, 3
	reps := []*tensor.Tensor{randMat(rng, batch, d), randMat(rng, batch, d)}
	w := randMat(rng, d, C)
	tgt := targetsVec([]int{0, 1, 2, 0})
	dims := []int{4, 8}
	a, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	b, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, dims)
	if err != nil {
		t.Fatal(err)
	}
	if a.AtF64() != b.AtF64() {
		t.Errorf("non-deterministic: %g vs %g", a.AtF64(), b.AtF64())
	}
}

// §V16 anchor 4 (sanity errors): empty layers, empty dims, non-increasing dims,
// out-of-range dims, mismatched weight lengths.
func TestMatryoshka2DErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	w := r(8, 3)
	tgt := targetsVec([]int{0, 1})
	reps := []*tensor.Tensor{r(2, 8), r(2, 8)}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), nil, w, tgt, []int{4}); err == nil {
		t.Error("empty layers: want error")
	}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, nil); err == nil {
		t.Error("empty dims: want error")
	}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{4, 4}); err == nil {
		t.Error("non-increasing dims: want error")
	}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{4, 9}); err == nil {
		t.Error("dim > D: want error")
	}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{4},
		nn.WithMatryoshka2DLayerWeights(1)); err == nil {
		t.Error("layer weight length mismatch: want error")
	}
	if _, err := nn.Matryoshka2DLoss(backend.NewContext(), reps, w, tgt, []int{4},
		nn.WithMatryoshka2DDimWeights(1, 1)); err == nil {
		t.Error("dim weight length mismatch: want error")
	}
}

func ExampleMatryoshka2DLoss() {
	// Two layers (a shallow and the final embedding) read out from a model, one
	// shared classifier, nesting dims {2,4}. The loss sums CE on each layer's
	// first-2 and first-4 coordinates over the 2×2 (layer × dim) grid.
	shallow := toT([][]float64{{2, 0, 0, 0}, {0, 0, 2, 0}})
	final := toT([][]float64{{1, 1, 0, 0}, {0, 0, 1, 1}})
	w := toT([][]float64{{1, 0}, {0, 1}, {0, 1}, {1, 0}})
	loss, _ := nn.Matryoshka2DLoss(backend.NewContext(),
		[]*tensor.Tensor{shallow, final}, w, targetsVec([]int{0, 1}), []int{2, 4})
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 1.9233
}
