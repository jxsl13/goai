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

// vicregRef independently computes the VICReg loss (paper form: invariance (1/N)Σ‖·‖²,
// unbiased variance, off-diagonal covariance). §V16 tier-1 parity oracle.
func vicregRef(za, zb [][]float64, lambda, mu, nu float64) float64 {
	n, d := len(za), len(za[0])
	var s float64
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			df := za[i][j] - zb[i][j]
			s += df * df
		}
	}
	s /= float64(n)
	varTerm := func(z [][]float64) float64 {
		var v float64
		for j := 0; j < d; j++ {
			var mean float64
			for i := range z {
				mean += z[i][j]
			}
			mean /= float64(n)
			var vr float64
			for i := range z {
				c := z[i][j] - mean
				vr += c * c
			}
			vr /= float64(n - 1)
			v += math.Max(0, 1-math.Sqrt(vr+1e-4))
		}
		return v / float64(d)
	}
	covTerm := func(z [][]float64) float64 {
		means := make([]float64, d)
		for j := range means {
			for i := range z {
				means[j] += z[i][j]
			}
			means[j] /= float64(n)
		}
		var c float64
		for a := 0; a < d; a++ {
			for b := 0; b < d; b++ {
				if a == b {
					continue
				}
				var cov float64
				for i := range z {
					cov += (z[i][a] - means[a]) * (z[i][b] - means[b])
				}
				cov /= float64(n - 1)
				c += cov * cov
			}
		}
		return c / float64(d)
	}
	return lambda*s + mu*(varTerm(za)+varTerm(zb)) + nu*(covTerm(za)+covTerm(zb))
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2105.04906 Eq.1-6): VICRegLoss matches the
// independent three-term reference.
func TestVICRegParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 130))
	za, zb := randMat(rng, 8, 5), randMat(rng, 8, 5)
	got, err := nn.VICRegLoss(backend.NewContext(), za, zb, 25, 25, 1)
	if err != nil {
		t.Fatal(err)
	}
	want := vicregRef(toRows(za), toRows(zb), 25, 25, 1)
	if math.Abs(got.AtF64()-want) > 1e-9 {
		t.Errorf("loss=%.10g want %.10g", got.AtF64(), want)
	}
}

// §V2 (the defining anti-collapse mechanism): a COLLAPSED representation (all samples
// identical ⇒ per-dim std ≈ 0) is heavily penalized by the variance hinge — each dim
// contributes ≈ γ=1, so the variance term dominates. This is what VICReg adds over a
// plain invariance loss.
func TestVICRegVarianceHingePenalizesCollapse(t *testing.T) {
	const n, d = 6, 4
	// every row identical ⇒ zero variance in every dimension.
	collapsed := tensor.New(tensor.F64, tensor.Shape{n, d})
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			collapsed.SetF64(float64(j), i, j) // same across rows
		}
	}
	loss, err := nn.VICRegLoss(backend.NewContext(), collapsed, collapsed, 25, 25, 1)
	if err != nil {
		t.Fatal(err)
	}
	// invariance=0 (identical views), covariance=0 (zero variance), so loss ≈
	// μ·[v+v] = 25·2·1 = 50 (each dim's hinge ≈ 1 − √(0+1e-4) ≈ 0.99).
	if math.Abs(loss.AtF64()-50) > 1 {
		t.Errorf("collapsed loss=%g, want ≈50 (variance hinge dominates)", loss.AtF64())
	}
}

// §V2: with well-spread, decorrelated features the variance and covariance terms are
// small, so the loss is driven by the invariance (view-mismatch) term.
func TestVICRegSpreadDecorrelated(t *testing.T) {
	// orthogonal, high-variance columns ⇒ variance hinge 0 and covariance 0.
	z := toT([][]float64{{2, 2}, {2, -2}, {-2, 2}, {-2, -2}})
	loss, err := nn.VICRegLoss(backend.NewContext(), z, z, 25, 25, 1)
	if err != nil {
		t.Fatal(err)
	}
	// identical views (invariance 0), std=√(16/3)≈2.3>1 (variance 0), orthogonal
	// columns (covariance 0) ⇒ loss ≈ 0.
	if loss.AtF64() > 1e-6 {
		t.Errorf("spread/decorrelated loss=%g, want ≈0", loss.AtF64())
	}
}

// §V2: differentiable w.r.t. both views.
func TestVICRegGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 131))
	za, zb := randMat(rng, 6, 3), randMat(rng, 6, 3)
	tape := autograd.NewTape()
	loss, err := nn.VICRegLoss(tape.Context(), za, zb, 25, 25, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		l, _ := nn.VICRegLoss(backend.NewContext(), za, zb, 25, 25, 1)
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

func TestVICRegErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.VICRegLoss(backend.NewContext(), r(4, 3), r(4, 5), 25, 25, 1); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.VICRegLoss(backend.NewContext(), r(1, 3), r(1, 3), 25, 25, 1); err == nil {
		t.Error("N<2: want error")
	}
	if _, err := nn.VICRegLoss(backend.NewContext(), r(4, 3, 1), r(4, 3, 1), 25, 25, 1); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleVICRegLoss() {
	// a collapsed representation (all rows equal) ⇒ the variance hinge fires, giving
	// the maximal μ·2·γ = 50 penalty that prevents collapse.
	z := toT([][]float64{{1, 2}, {1, 2}, {1, 2}, {1, 2}})
	loss, _ := nn.VICRegLoss(backend.NewContext(), z, z, 25, 25, 1)
	fmt.Printf("%.1f\n", loss.AtF64()) // ≈ μ·2·(1−√1e-4) = 25·2·0.99
	// Output:
	// 49.5
}
