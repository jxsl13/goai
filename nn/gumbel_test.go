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

func temperedSoftmaxRow(logits []float64, tau float64) []float64 {
	mx := math.Inf(-1)
	for _, v := range logits {
		if v/tau > mx {
			mx = v / tau
		}
	}
	var sum float64
	out := make([]float64, len(logits))
	for i, v := range logits {
		out[i] = math.Exp(v/tau - mx)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// §V16 tier-1 (paper CONFIRMED, arXiv:1611.01144 Eq.1-2): with zero Gumbel noise the
// relaxation is exactly the temperature-scaled softmax over logits.
func TestGumbelSoftmaxZeroNoiseIsTemperedSoftmax(t *testing.T) {
	logits := toT([][]float64{{2, 1, 0.5, -1}, {0, 3, -2, 1}})
	zero := tensor.New(tensor.F64, logits.Shape())
	for _, tau := range []float64{0.5, 1, 4} {
		y, err := nn.GumbelSoftmax(backend.NewContext(), logits, zero, tau)
		if err != nil {
			t.Fatal(err)
		}
		for b := 0; b < 2; b++ {
			want := temperedSoftmaxRow(toRows(logits)[b], tau)
			for j := range want {
				if math.Abs(y.AtF64(b, j)-want[j]) > 1e-9 {
					t.Errorf("τ=%g y[%d,%d]=%g want %g", tau, b, j, y.AtF64(b, j), want[j])
				}
			}
		}
	}
}

// §V2: every row is a valid distribution (nonneg, sums to 1).
func TestGumbelSoftmaxIsDistribution(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 30))
	logits := randMat(rng, 4, 5)
	noise := nn.SampleGumbelNoise(7, logits.Shape())
	y, err := nn.GumbelSoftmax(backend.NewContext(), logits, noise, 0.7)
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < 4; b++ {
		var sum float64
		for j := 0; j < 5; j++ {
			if v := y.AtF64(b, j); v < -1e-12 {
				t.Errorf("y[%d,%d]=%g negative", b, j, v)
			} else {
				sum += v
			}
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("row %d sums to %g, want 1", b, sum)
		}
	}
}

// §V16 tier-1 (τ→0 limit): a small temperature drives the sample to the one-hot
// Gumbel-Max argmax over (logit + gumbel).
func TestGumbelSoftmaxLowTempApproachesOneHot(t *testing.T) {
	logits := toT([][]float64{{0.5, 2.0, 1.0}})
	noise := nn.SampleGumbelNoise(3, logits.Shape())
	y, err := nn.GumbelSoftmax(backend.NewContext(), logits, noise, 0.01)
	if err != nil {
		t.Fatal(err)
	}
	// argmax of (logit + gumbel)
	want, bv := 0, math.Inf(-1)
	for j := 0; j < 3; j++ {
		if s := logits.AtF64(0, j) + noise.AtF64(0, j); s > bv {
			want, bv = j, s
		}
	}
	if y.AtF64(0, want) < 0.999 {
		t.Errorf("τ→0: y[%d]=%g not ≈1 (want one-hot at argmax)", want, y.AtF64(0, want))
	}
}

// §V16 tier-1 (Gumbel noise): g = −log(−log(u)) — deterministic per seed, finite,
// and the sample mean approaches the Euler-Mascheroni constant γ≈0.5772.
func TestSampleGumbelNoise(t *testing.T) {
	a := nn.SampleGumbelNoise(42, tensor.Shape{3, 3})
	b := nn.SampleGumbelNoise(42, tensor.Shape{3, 3})
	for i := range a.Numel() {
		c := tensor.Unravel(i, a.Shape())
		if a.AtF64(c...) != b.AtF64(c...) {
			t.Fatal("SampleGumbelNoise not deterministic for a fixed seed")
		}
		if math.IsInf(a.AtF64(c...), 0) || math.IsNaN(a.AtF64(c...)) {
			t.Fatal("SampleGumbelNoise produced non-finite value")
		}
	}
	big := nn.SampleGumbelNoise(1, tensor.Shape{20000})
	var mean float64
	for i := 0; i < 20000; i++ {
		mean += big.AtF64(i)
	}
	mean /= 20000
	if math.Abs(mean-0.5772) > 0.05 {
		t.Errorf("Gumbel sample mean=%g, want ≈γ=0.5772", mean)
	}
}

// §V2: the soft relaxation is differentiable w.r.t. the logits.
func TestGumbelSoftmaxGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 31))
	const batch, c = 3, 4
	logits := randMat(rng, batch, c)
	noise := nn.SampleGumbelNoise(9, logits.Shape())
	coef := randMat(rng, batch, c) // fixed linear read-out so the loss is non-trivial
	loss := func(ctx *backend.Context) (*tensor.Tensor, error) {
		y, err := nn.GumbelSoftmax(ctx, logits, noise, 0.8)
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
	g := tape.Grad(logits)
	const h = 1e-6
	for idx := range logits.Numel() {
		coord := tensor.Unravel(idx, logits.Shape())
		o := logits.AtF64(coord...)
		logits.SetF64(o+h, coord...)
		lp, _ := loss(backend.NewContext())
		logits.SetF64(o-h, coord...)
		lm, _ := loss(backend.NewContext())
		logits.SetF64(o, coord...)
		if num, ana := (lp.AtF64()-lm.AtF64())/(2*h), g.AtF64(coord...); math.Abs(num-ana) > 1e-4*math.Max(1, math.Abs(num)) {
			t.Errorf("grad%v: numeric %.8g vs analytic %.8g", coord, num, ana)
		}
	}
}

// §V16 tier-1 (Straight-Through, §2.2): the hard forward is one-hot at the
// Gumbel-Max argmax, and its gradient equals the soft relaxation's gradient (for a
// linear loss the two are exactly equal — the defining ST property).
func TestGumbelSoftmaxHardStraightThrough(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 32))
	const batch, c = 3, 4
	logits := randMat(rng, batch, c)
	noise := nn.SampleGumbelNoise(11, logits.Shape())
	coef := randMat(rng, batch, c)

	// forward is a one-hot at argmax(logit+gumbel).
	hard, err := nn.GumbelSoftmaxHard(backend.NewContext(), logits, noise, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	for b := 0; b < batch; b++ {
		var sum, mx float64
		want, bv := 0, math.Inf(-1)
		for j := 0; j < c; j++ {
			v := hard.AtF64(b, j)
			sum += v
			if v > mx {
				mx = v
			}
			if s := logits.AtF64(b, j) + noise.AtF64(b, j); s > bv {
				want, bv = j, s
			}
		}
		if math.Abs(sum-1) > 1e-12 || math.Abs(mx-1) > 1e-12 || hard.AtF64(b, want) != 1 {
			t.Errorf("row %d not one-hot at argmax %d: %v", b, want, sum)
		}
	}

	// ST gradient (hard) == soft gradient for the linear loss Σ coef·y.
	gradOf := func(hardPath bool) *tensor.Tensor {
		tape := autograd.NewTape()
		var y *tensor.Tensor
		var err error
		if hardPath {
			y, err = nn.GumbelSoftmaxHard(tape.Context(), logits, noise, 0.5)
		} else {
			y, err = nn.GumbelSoftmax(tape.Context(), logits, noise, 0.5)
		}
		if err != nil {
			t.Fatal(err)
		}
		p, _ := backend.Execute(tape.Context(), backend.OpMul, []*tensor.Tensor{y, coef}, nil)
		l, _ := sumAll(tape.Context(), p[0])
		if err := tape.Backward(l); err != nil {
			t.Fatal(err)
		}
		return tape.Grad(logits)
	}
	gHard, gSoft := gradOf(true), gradOf(false)
	for idx := range logits.Numel() {
		c := tensor.Unravel(idx, logits.Shape())
		if math.Abs(gHard.AtF64(c...)-gSoft.AtF64(c...)) > 1e-12 {
			t.Errorf("ST grad%v=%g != soft grad %g", c, gHard.AtF64(c...), gSoft.AtF64(c...))
		}
	}
}

func TestGumbelSoftmaxErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.GumbelSoftmax(backend.NewContext(), r(2, 3), r(2, 4), 1); err == nil {
		t.Error("shape mismatch: want error")
	}
	if _, err := nn.GumbelSoftmax(backend.NewContext(), r(2, 3), r(2, 3), 0); err == nil {
		t.Error("tau=0: want error")
	}
	if _, err := nn.GumbelSoftmax(backend.NewContext(), r(2, 3, 1), r(2, 3, 1), 1); err == nil {
		t.Error("rank-3: want error")
	}
}

func ExampleGumbelSoftmax() {
	// logit strongly favors class 1; with low temperature the relaxed sample is
	// nearly one-hot there.
	logits := toT([][]float64{{0, 5, 0}})
	zero := tensor.New(tensor.F64, logits.Shape()) // zero noise ⇒ deterministic
	y, _ := nn.GumbelSoftmax(backend.NewContext(), logits, zero, 0.5)
	fmt.Printf("%.2f\n", y.AtF64(0, 1))
	// Output:
	// 1.00
}
