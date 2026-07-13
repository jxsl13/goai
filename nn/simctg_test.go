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

// simctgRef independently computes SimCTG's L_CL (Eq. 2, arXiv:2202.06417):
// (1/(n(n−1)))·Σ_{i≠j} max(0, ρ − s(h_i,h_i) + s(h_i,h_j)) with s = cosine.
// §V16 tier-1 parity oracle.
func simctgRef(h [][]float64, margin float64) float64 {
	n := len(h)
	cos := func(a, b []float64) float64 {
		var dot, na, nb float64
		for k := range a {
			dot += a[k] * b[k]
			na += a[k] * a[k]
			nb += b[k] * b[k]
		}
		return dot / (math.Sqrt(na) * math.Sqrt(nb))
	}
	var sum float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if i == j {
				continue
			}
			sum += math.Max(0, margin-cos(h[i], h[i])+cos(h[i], h[j]))
		}
	}
	return sum / float64(n*(n-1))
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2202.06417 §3.1 Eq. 2): the loss matches an
// independent cosine-hinge reference.
func TestSimCTGFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 208))
	rows := [][]float64{}
	for i := 0; i < 5; i++ {
		rows = append(rows, []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()})
	}
	h := toT(rows)
	got, err := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if want := simctgRef(rows, 0.5); math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.12g want %.12g", got.AtF64(), want)
	}
}

// §V2 (defining property + diagonal exclusion): mutually orthogonal representations have
// pairwise cosine 0 < 1−ρ, so every off-diagonal hinge is inactive and the loss is 0.
// This also proves the self-pairs (i=i, cosine 1 ⇒ hinge ρ) are correctly excluded — if
// they leaked in the loss would be ρ/(n−1) > 0, not 0.
func TestSimCTGOrthogonalIsZero(t *testing.T) {
	h := toT([][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}})
	loss, err := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if loss.AtF64() > 1e-12 {
		t.Errorf("orthogonal reps loss=%g, want 0", loss.AtF64())
	}
}

// §V2: two parallel (cosine-1) representations sit fully inside the hinge, each pair
// contributing max(0, ρ−1+1)=ρ, so the mean loss equals the margin ρ exactly.
func TestSimCTGIdenticalPairsGiveMargin(t *testing.T) {
	h := toT([][]float64{{1, 2}, {2, 4}}) // both point along [1,2] ⇒ cosine 1
	loss, err := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(loss.AtF64()-0.5) > 1e-9 {
		t.Errorf("cosine-1 pair loss=%g, want ρ=0.5", loss.AtF64())
	}
}

// §V16 tier-1 (ρ=0 ⇒ degenerates to MLE): with margin 0 no pair can be penalized
// (cosine ≤ 1 ⇒ 0−1+cos ≤ 0), so the contrastive term vanishes for ANY representations.
func TestSimCTGZeroMarginIsZero(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 211))
	rows := [][]float64{}
	for i := 0; i < 6; i++ {
		rows = append(rows, []float64{rng.NormFloat64(), rng.NormFloat64(), rng.NormFloat64()})
	}
	loss, err := nn.SimCTGContrastiveLoss(backend.NewContext(), toT(rows), 0)
	if err != nil {
		t.Fatal(err)
	}
	if loss.AtF64() > 1e-12 {
		t.Errorf("ρ=0 loss=%g, want 0", loss.AtF64())
	}
}

// §V2: a larger margin pushes representations apart more strongly, so the loss is
// non-decreasing in ρ (strictly increasing while pairs stay in the active hinge).
func TestSimCTGMarginIncreasesLoss(t *testing.T) {
	h := toT([][]float64{{3, 3, 2.9}, {3.1, 2.9, 3}, {2.9, 3, 3.1}}) // clustered ⇒ cosine ≈ 1
	lo, _ := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.3)
	hi, _ := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.8)
	if !(hi.AtF64() > lo.AtF64()) {
		t.Errorf("loss(ρ=0.8)=%g not > loss(ρ=0.3)=%g", hi.AtF64(), lo.AtF64())
	}
}

// §V2: differentiable w.r.t. the token representations. Reps are clustered so every
// off-diagonal pair sits strictly inside the hinge (cosine ≈ 1 > 1−ρ), keeping the ReLU
// locally linear where central finite differences are valid.
func TestSimCTGGradcheck(t *testing.T) {
	h := toT([][]float64{
		{3.0, 3.1, 2.9},
		{3.2, 2.8, 3.0},
		{2.9, 3.0, 3.1},
		{3.1, 3.0, 2.8},
	})
	tape := autograd.NewTape()
	loss, err := nn.SimCTGContrastiveLoss(tape.Context(), h, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	g := tape.Grad(h)
	if g == nil {
		t.Fatal("nil grad")
	}
	const eps = 1e-6
	fwd := func() float64 {
		l, _ := nn.SimCTGContrastiveLoss(backend.NewContext(), h, 0.5)
		return l.AtF64()
	}
	for idx := range h.Numel() {
		coord := tensor.Unravel(idx, h.Shape())
		o := h.AtF64(coord...)
		h.SetF64(o+eps, coord...)
		lp := fwd()
		h.SetF64(o-eps, coord...)
		lm := fwd()
		h.SetF64(o, coord...)
		if num, ana := (lp-lm)/(2*eps), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", coord, num, ana)
		}
	}
}

func TestSimCTGErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.SimCTGContrastiveLoss(backend.NewContext(), r(4), 0.5); err == nil {
		t.Error("rank-1: want error")
	}
	if _, err := nn.SimCTGContrastiveLoss(backend.NewContext(), r(4, 3, 1), 0.5); err == nil {
		t.Error("rank-3: want error")
	}
	if _, err := nn.SimCTGContrastiveLoss(backend.NewContext(), r(1, 3), 0.5); err == nil {
		t.Error("seq<2: want error")
	}
}

func ExampleSimCTGContrastiveLoss() {
	// Two near-identical token representations are exactly what SimCTG penalizes: their
	// cosine ≈ 1 sits fully inside the margin, so the loss is the margin ρ=0.5. Well-
	// separated (orthogonal) representations would give 0.
	h := toT([][]float64{{1, 2}, {2, 4}})
	loss, _ := nn.SimCTGContrastiveLoss(backend.NewContext(), h, nn.SimCTGMargin)
	fmt.Printf("%.1f\n", loss.AtF64())
	// Output:
	// 0.5
}
