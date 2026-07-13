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

// barlowRef independently computes the Barlow Twins loss: per-feature batch
// standardization, cross-correlation, invariance + λ·redundancy. §V16 tier-1 oracle.
func barlowRef(za, zb [][]float64, lambda float64) float64 {
	const eps = 1e-5
	n, d := len(za), len(za[0])
	std := func(z [][]float64) [][]float64 {
		out := make([][]float64, n)
		for i := range out {
			out[i] = make([]float64, d)
		}
		for j := 0; j < d; j++ {
			var mean float64
			for i := 0; i < n; i++ {
				mean += z[i][j]
			}
			mean /= float64(n)
			var v float64
			for i := 0; i < n; i++ {
				c := z[i][j] - mean
				v += c * c
			}
			v /= float64(n)
			s := math.Sqrt(v + eps)
			for i := 0; i < n; i++ {
				out[i][j] = (z[i][j] - mean) / s
			}
		}
		return out
	}
	na, nb := std(za), std(zb)
	var loss float64
	for i := 0; i < d; i++ {
		for j := 0; j < d; j++ {
			var c float64
			for b := 0; b < n; b++ {
				c += na[b][i] * nb[b][j]
			}
			c /= float64(n)
			if i == j {
				loss += (1 - c) * (1 - c)
			} else {
				loss += lambda * c * c
			}
		}
	}
	return loss
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2103.03230 Eq.1): BarlowTwinsLoss matches the
// independent standardize→cross-correlation→invariance+redundancy reference.
func TestBarlowParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 80))
	za, zb := randMat(rng, 8, 4), randMat(rng, 8, 4)
	got, err := nn.BarlowTwinsLoss(backend.NewContext(), za, zb, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	want := barlowRef(toRows(za), toRows(zb), 0.005)
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.10g want %.10g", got.AtF64(), want)
	}
}

// §V2 (defining minimum): when the cross-correlation is the identity — two identical
// views with decorrelated features — the loss is ≈0.
func TestBarlowIdentityCorrelationIsZero(t *testing.T) {
	// columns are orthogonal and mean-0 ⇒ standardized C = I.
	z := toT([][]float64{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}})
	loss, err := nn.BarlowTwinsLoss(backend.NewContext(), z, z, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	if loss.AtF64() > 1e-6 { // only the ε-standardization residual remains
		t.Errorf("C=I loss=%g, want ≈0", loss.AtF64())
	}
}

// §V2 (redundancy reduction): perfectly correlated (redundant) features incur only
// the off-diagonal penalty, so the loss ≈ 2λ and scales linearly with λ.
func TestBarlowRedundancyPenalty(t *testing.T) {
	// both features identical ⇒ diagonal ≈1 (invariance≈0), off-diagonal ≈1.
	z := toT([][]float64{{1, 1}, {-1, -1}, {1, 1}, {-1, -1}})
	l1, _ := nn.BarlowTwinsLoss(backend.NewContext(), z, z, 0.005)
	l2, _ := nn.BarlowTwinsLoss(backend.NewContext(), z, z, 0.010)
	if math.Abs(l1.AtF64()-2*0.005) > 1e-4 {
		t.Errorf("redundant loss=%g, want ≈2λ=%g", l1.AtF64(), 2*0.005)
	}
	if math.Abs(l2.AtF64()/l1.AtF64()-2) > 1e-3 { // doubling λ doubles the loss
		t.Errorf("loss ratio %g, want ≈2 (redundancy scales with λ)", l2.AtF64()/l1.AtF64())
	}
}

// §V2: differentiable w.r.t. both views.
func TestBarlowGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 81))
	za, zb := randMat(rng, 6, 3), randMat(rng, 6, 3)
	tape := autograd.NewTape()
	loss, err := nn.BarlowTwinsLoss(tape.Context(), za, zb, 0.005)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.BarlowTwinsLoss(backend.NewContext(), za, zb, 0.005)
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
	check("za", za)
	check("zb", zb)
}

func TestBarlowErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.BarlowTwinsLoss(backend.NewContext(), r(4, 3), r(4, 5), 0.005); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.BarlowTwinsLoss(backend.NewContext(), r(4, 3, 1), r(4, 3, 1), 0.005); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleBarlowTwinsLoss() {
	// two identical views with orthogonal, decorrelated features ⇒ C = I ⇒ loss ≈ 0.
	z := toT([][]float64{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}})
	loss, _ := nn.BarlowTwinsLoss(backend.NewContext(), z, z, 0.005)
	fmt.Printf("%.4f\n", loss.AtF64())
	// Output:
	// 0.0000
}
