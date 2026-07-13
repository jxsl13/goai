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

// gatedDeltaNetRef independently runs the gated delta rule with L2-normalized q,k:
// S̃=α_t·S_{t-1}; e=v−S̃k; S=S̃+β·e·kᵀ; o=S·q. §V16 tier-1 parity oracle.
func gatedDeltaNetRef(q, k, v [][]float64, alpha, beta []float64) [][]float64 {
	seq, dk, dv := len(q), len(q[0]), len(v[0])
	l2 := func(r []float64) []float64 {
		var ss float64
		for _, x := range r {
			ss += x * x
		}
		n := math.Sqrt(ss + 1e-12)
		o := make([]float64, len(r))
		for i, x := range r {
			o[i] = x / n
		}
		return o
	}
	S := make([][]float64, dv)
	for i := range S {
		S[i] = make([]float64, dk)
	}
	out := make([][]float64, seq)
	for t := 0; t < seq; t++ {
		kt, qt := l2(k[t]), l2(q[t])
		for i := 0; i < dv; i++ { // decay
			for j := 0; j < dk; j++ {
				S[i][j] *= alpha[t]
			}
		}
		pred := make([]float64, dv)
		for i := 0; i < dv; i++ {
			for j := 0; j < dk; j++ {
				pred[i] += S[i][j] * kt[j]
			}
		}
		for i := 0; i < dv; i++ {
			e := (v[t][i] - pred[i]) * beta[t]
			for j := 0; j < dk; j++ {
				S[i][j] += e * kt[j]
			}
		}
		o := make([]float64, dv)
		for i := 0; i < dv; i++ {
			for j := 0; j < dk; j++ {
				o[i] += S[i][j] * qt[j]
			}
		}
		out[t] = o
	}
	return out
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2412.06464): GatedDeltaNet matches the
// independent gated-delta recurrence over random q,k,v,α,β.
func TestGatedDeltaNetParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 20))
	const seq, dk, dv = 5, 3, 4
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	aVals, alpha := randBeta(rng, seq)
	bVals, beta := randBeta(rng, seq)
	got, err := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
	if err != nil {
		t.Fatal(err)
	}
	want := gatedDeltaNetRef(toRows(q), toRows(k), toRows(v), aVals, bVals)
	for i := 0; i < seq; i++ {
		for j := 0; j < dv; j++ {
			if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want %.10g", i, j, g, want[i][j])
			}
		}
	}
}

// §V2 (defining reduction): with α_t=1 (no decay) GatedDeltaNet is EXACTLY plain
// DeltaNet — checked against the independent nn.DeltaNet implementation.
func TestGatedDeltaNetAlphaOneEqualsDeltaNet(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 21))
	const seq, dk, dv = 5, 3, 4
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, beta := randBeta(rng, seq)
	ones := betaCol(func() []float64 {
		o := make([]float64, seq)
		for i := range o {
			o[i] = 1
		}
		return o
	}())
	gated, err := nn.GatedDeltaNet(backend.NewContext(), q, k, v, ones, beta)
	if err != nil {
		t.Fatal(err)
	}
	plain, err := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < seq; i++ {
		for j := 0; j < dv; j++ {
			if math.Abs(gated.AtF64(i, j)-plain.AtF64(i, j)) > 1e-12 {
				t.Errorf("α=1 out[%d,%d]=%g != DeltaNet %g", i, j, gated.AtF64(i, j), plain.AtF64(i, j))
			}
		}
	}
}

// §V2 (the distinctive decay): write a key at t=0 (β=1), then hold β=0 with α=0.5 —
// the retrieved memory decays geometrically o_t = 0.5ᵗ·o_0 (DeltaNet cannot forget).
func TestGatedDeltaNetGeometricDecay(t *testing.T) {
	key := [][]float64{{1, 0}, {1, 0}, {1, 0}}
	val := [][]float64{{4, 8}, {0, 0}, {0, 0}}
	out, err := nn.GatedDeltaNet(backend.NewContext(),
		toT(key), toT(key), toT(val), betaCol([]float64{1, 0.5, 0.5}), betaCol([]float64{1, 0, 0}))
	if err != nil {
		t.Fatal(err)
	}
	for j, base := range []float64{4, 8} {
		for tstep, want := range []float64{base, 0.5 * base, 0.25 * base} {
			if g := out.AtF64(tstep, j); math.Abs(g-want) > 1e-9 {
				t.Errorf("decay out[%d,%d]=%g want %g", tstep, j, g, want)
			}
		}
	}
}

// §V2: causal — a future value cannot affect an earlier output.
func TestGatedDeltaNetCausality(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 22))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, alpha := randBeta(rng, seq)
	_, beta := randBeta(rng, seq)
	base, _ := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
	v.SetF64(v.AtF64(seq-1, 0)+5, seq-1, 0)
	pert, _ := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
	for i := 0; i < seq-1; i++ {
		for j := 0; j < dv; j++ {
			if math.Abs(base.AtF64(i, j)-pert.AtF64(i, j)) > 1e-12 {
				t.Errorf("future value changed out[%d,%d]", i, j)
			}
		}
	}
}

// §V2: differentiable w.r.t. q, k, v, alpha and beta.
func TestGatedDeltaNetGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 23))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, alpha := randBeta(rng, seq)
	_, beta := randBeta(rng, seq)
	tape := autograd.NewTape()
	out, err := nn.GatedDeltaNet(tape.Context(), q, k, v, alpha, beta)
	if err != nil {
		t.Fatal(err)
	}
	loss, err := sumAll(tape.Context(), out)
	if err != nil {
		t.Fatal(err)
	}
	if err := tape.Backward(loss); err != nil {
		t.Fatal(err)
	}
	const h = 1e-6
	fwd := func() float64 {
		o, _ := nn.GatedDeltaNet(backend.NewContext(), q, k, v, alpha, beta)
		var s float64
		for i := range o.Numel() {
			s += o.AtF64(tensor.Unravel(i, o.Shape())...)
		}
		return s
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
	check("k", k)
	check("v", v)
	check("alpha", alpha)
	check("beta", beta)
}

func TestGatedDeltaNetErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.GatedDeltaNet(backend.NewContext(), r(2, 3, 1), r(2, 3), r(2, 3), r(2, 1), r(2, 1)); err == nil {
		t.Error("rank-3: want error")
	}
	if _, err := nn.GatedDeltaNet(backend.NewContext(), r(2, 3), r(3, 3), r(2, 3), r(2, 1), r(2, 1)); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := nn.GatedDeltaNet(backend.NewContext(), r(2, 3), r(2, 3), r(2, 3), r(2, 2), r(2, 1)); err == nil {
		t.Error("alpha not [seq,1]: want error")
	}
}

func ExampleGatedDeltaNet() {
	// write [4,8] under a key (β=1), then decay by α=0.5 for one step and read back.
	key := toT([][]float64{{1, 0}, {1, 0}})
	val := toT([][]float64{{4, 8}, {0, 0}})
	out, _ := nn.GatedDeltaNet(backend.NewContext(),
		key, key, val, betaCol([]float64{1, 0.5}), betaCol([]float64{1, 0}))
	fmt.Printf("%.2f %.2f\n", out.AtF64(1, 0), out.AtF64(1, 1)) // half of [4,8]
	// Output:
	// 2.00 4.00
}
