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

// groupNormRef independently computes GroupNorm: per-sample per-group biased-variance
// normalization + per-channel affine. §V16 tier-1 parity oracle.
func groupNormRef(x [][]float64, groups int, gamma, beta []float64, eps float64) [][]float64 {
	c := len(x[0])
	gs := c / groups
	out := make([][]float64, len(x))
	for n := range x {
		out[n] = make([]float64, c)
		for g := 0; g < groups; g++ {
			var mean float64
			for k := 0; k < gs; k++ {
				mean += x[n][g*gs+k]
			}
			mean /= float64(gs)
			var v float64
			for k := 0; k < gs; k++ {
				d := x[n][g*gs+k] - mean
				v += d * d
			}
			v /= float64(gs) // biased
			std := math.Sqrt(v + eps)
			for k := 0; k < gs; k++ {
				ch := g*gs + k
				out[n][ch] = gamma[ch]*(x[n][ch]-mean)/std + beta[ch]
			}
		}
	}
	return out
}

func setVec(t *tensor.Tensor, v []float64) {
	for i, x := range v {
		t.SetF64(x, i)
	}
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1803.08494): GroupNorm matches the independent
// per-group biased-variance reference with a per-channel affine.
func TestGroupNormParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 70))
	const batch, c, groups = 4, 8, 2
	x := randMat(rng, batch, c)
	gn := nn.NewGroupNorm(tensor.F64, groups, c)
	gamma := make([]float64, c)
	beta := make([]float64, c)
	for i := range c {
		gamma[i] = rng.NormFloat64()
		beta[i] = rng.NormFloat64()
	}
	setVec(gn.Gamma, gamma)
	setVec(gn.Beta, beta)
	got, err := gn.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	want := groupNormRef(toRows(x), groups, gamma, beta, gn.Eps)
	for n := 0; n < batch; n++ {
		for ch := 0; ch < c; ch++ {
			if math.Abs(got.AtF64(n, ch)-want[n][ch]) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want %.10g", n, ch, got.AtF64(n, ch), want[n][ch])
			}
		}
	}
}

// §V2 (defining reduction): G=1 is exactly LayerNorm over all channels — checked
// against the existing nn.LayerNorm.
func TestGroupNormG1EqualsLayerNorm(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 71))
	const batch, c = 3, 6
	x := randMat(rng, batch, c)
	gn := nn.NewGroupNorm(tensor.F64, 1, c)
	ln := nn.NewLayerNorm(tensor.F64, c)
	for i := range c { // give both the same non-trivial affine
		v, w := rng.NormFloat64(), rng.NormFloat64()
		gn.Gamma.SetF64(v, i)
		ln.Gamma.SetF64(v, i)
		gn.Beta.SetF64(w, i)
		ln.Beta.SetF64(w, i)
	}
	g, err := gn.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	l, err := ln.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < batch; n++ {
		for ch := 0; ch < c; ch++ {
			if math.Abs(g.AtF64(n, ch)-l.AtF64(n, ch)) > 1e-9 {
				t.Errorf("G=1 GroupNorm[%d,%d]=%g != LayerNorm %g", n, ch, g.AtF64(n, ch), l.AtF64(n, ch))
			}
		}
	}
}

// §V2: with the identity affine each group of the output is standardized (mean ≈ 0,
// biased variance ≈ 1) per sample.
func TestGroupNormStandardizes(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 72))
	const batch, c, groups = 2, 9, 3
	gs := c / groups
	x := randMat(rng, batch, c)
	for i := range x.Numel() { // widen so variance ≫ eps
		co := tensor.Unravel(i, x.Shape())
		x.SetF64(x.AtF64(co...)*5, co...)
	}
	gn := nn.NewGroupNorm(tensor.F64, groups, c) // γ=1, β=0
	out, err := gn.Forward(backend.NewContext(), x)
	if err != nil {
		t.Fatal(err)
	}
	for n := 0; n < batch; n++ {
		for g := 0; g < groups; g++ {
			var mean, v float64
			for k := 0; k < gs; k++ {
				mean += out.AtF64(n, g*gs+k)
			}
			mean /= float64(gs)
			for k := 0; k < gs; k++ {
				d := out.AtF64(n, g*gs+k) - mean
				v += d * d
			}
			v /= float64(gs)
			if math.Abs(mean) > 1e-9 {
				t.Errorf("sample %d group %d mean=%g, want ≈0", n, g, mean)
			}
			if math.Abs(v-1) > 1e-3 {
				t.Errorf("sample %d group %d var=%g, want ≈1", n, g, v)
			}
		}
	}
}

// §V2: differentiable w.r.t. x, γ and β.
func TestGroupNormGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 73))
	const batch, c, groups = 3, 6, 3
	x := randMat(rng, batch, c)
	gn := nn.NewGroupNorm(tensor.F64, groups, c)
	for i := range c {
		gn.Gamma.SetF64(0.5+rng.Float64(), i)
		gn.Beta.SetF64(rng.NormFloat64(), i)
	}
	coef := randMat(rng, batch, c)
	loss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := gn.Forward(ctx, x)
		if err != nil {
			return nil, err
		}
		p, err := backend.Execute(ctx, backend.OpMul, []*tensor.Tensor{y, coef}, nil)
		if err != nil {
			return nil, err
		}
		return sumAll(ctx, p[0])
	}
	tape := autograd.NewTape()
	l, err := loss(tape.Context())
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(l); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	check := func(name string, p *tensor.Tensor) {
		g := tape.Grad(p)
		if g == nil {
			t.Fatalf("%s: nil grad", name)
		}
		for idx := range p.Numel() {
			coord := tensor.Unravel(idx, p.Shape())
			o := p.AtF64(coord...)
			p.SetF64(o+h, coord...)
			lp, _ := loss(backend.NewContext())
			p.SetF64(o-h, coord...)
			lm, _ := loss(backend.NewContext())
			p.SetF64(o, coord...)
			if num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
				t.Errorf("%s grad%v: numeric %.8g vs analytic %.8g", name, coord, num, ana)
			}
		}
	}
	check("x", x)
	check("gamma", gn.Gamma)
	check("beta", gn.Beta)
}

func TestGroupNormErrors(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("groups not dividing C: want panic")
		}
	}()
	nn.NewGroupNorm(tensor.F64, 3, 8) // 3 ∤ 8
}

func TestGroupNormRankError(t *testing.T) {
	gn := nn.NewGroupNorm(tensor.F64, 2, 4)
	if _, err := gn.Forward(backend.NewContext(), tensor.New(tensor.F64, tensor.Shape{2, 4, 1})); err == nil {
		t.Error("rank-3 input: want error")
	}
}

func ExampleGroupNorm() {
	// one sample, 4 channels, 2 groups: each group (2 channels) is standardized to
	// ±1 around its own mean, then the identity affine leaves it unchanged.
	gn := nn.NewGroupNorm(tensor.F64, 2, 4)
	x := toT([][]float64{{1, 3, 10, 14}}) // group0 {1,3}, group1 {10,14}
	out, _ := gn.Forward(backend.NewContext(), x)
	fmt.Printf("%.2f %.2f %.2f %.2f\n", out.AtF64(0, 0), out.AtF64(0, 1), out.AtF64(0, 2), out.AtF64(0, 3))
	// Output:
	// -1.00 1.00 -1.00 1.00
}
