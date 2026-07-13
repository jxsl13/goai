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

// deltaNetRef independently runs the delta-rule recurrence with L2-normalized q,k.
// §V16 tier-1 parity oracle.
func deltaNetRef(q, k, v [][]float64, beta []float64) [][]float64 {
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

func betaCol(vals []float64) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{len(vals), 1})
	for i, v := range vals {
		t.SetF64(v, i, 0)
	}
	return t
}

func randBeta(rng *rand.Rand, seq int) ([]float64, *tensor.Tensor) {
	vals := make([]float64, seq)
	for i := range vals {
		vals[i] = 1 / (1 + math.Exp(-rng.NormFloat64())) // σ(·) ∈ (0,1)
	}
	return vals, betaCol(vals)
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2406.06484 §2.2/§3.1 Eq.3): DeltaNet matches
// the independent delta-rule recurrence over random q,k,v and per-token β.
func TestDeltaNetParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 14))
	const seq, dk, dv = 5, 3, 4
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	bvals, beta := randBeta(rng, seq)
	got, err := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	if err != nil {
		t.Fatal(err)
	}
	want := deltaNetRef(toRows(q), toRows(k), toRows(v), bvals)
	for i := 0; i < seq; i++ {
		for j := 0; j < dv; j++ {
			if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want %.10g", i, j, g, want[i][j])
			}
		}
	}
}

// §V2 (defining property, Eq.3): writing (k,v) with β=1 then reading the SAME key
// returns v exactly — and a second β=1 write to the same key OVERWRITES (v_b, not
// v_a+v_b as ungated linear attention would accumulate).
func TestDeltaNetExactOverwrite(t *testing.T) {
	// single-key retrieval: seq=1, q=k, β=1 ⇒ o_0 = v_0.
	k0 := [][]float64{{0.6, -0.8}} // any vector; the module L2-normalizes it
	v0 := [][]float64{{3, 7, -2}}
	o, err := nn.DeltaNet(backend.NewContext(), toT(k0), toT(k0), toT(v0), betaCol([]float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	for j, want := range v0[0] {
		if math.Abs(o.AtF64(0, j)-want) > 1e-9 {
			t.Errorf("retrieval o[0,%d]=%g want v0=%g", j, o.AtF64(0, j), want)
		}
	}
	// overwrite: write (k,v_a) then (k,v_b) with the same key, both β=1, then read.
	key := []float64{1, 0}
	va, vb := []float64{5, 5, 5}, []float64{-1, 2, 9}
	q := toT([][]float64{key, key, key})
	kk := toT([][]float64{key, key, key})
	vv := toT([][]float64{va, vb, {0, 0, 0}})
	out, err := nn.DeltaNet(backend.NewContext(), q, kk, vv, betaCol([]float64{1, 1, 0}))
	if err != nil {
		t.Fatal(err)
	}
	for j := range vb { // step 2 reads the key: delta rule ⇒ v_b (overwrite), NOT v_a+v_b
		if math.Abs(out.AtF64(2, j)-vb[j]) > 1e-9 {
			t.Errorf("overwrite o[2,%d]=%g want v_b=%g (accumulate would give %g)", j, out.AtF64(2, j), vb[j], va[j]+vb[j])
		}
	}
}

// §V2: β=0 makes no write, so with an empty initial memory every output is zero.
func TestDeltaNetZeroBetaNoUpdate(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 15))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	out, err := nn.DeltaNet(backend.NewContext(), q, k, v, betaCol(make([]float64, seq)))
	if err != nil {
		t.Fatal(err)
	}
	for i := range out.Numel() {
		if g := out.AtF64(tensor.Unravel(i, out.Shape())...); math.Abs(g) > 1e-12 {
			t.Errorf("β=0 output not zero: %g", g)
		}
	}
}

// §V2: causal — a future input cannot affect an earlier output.
func TestDeltaNetCausality(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 16))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, beta := randBeta(rng, seq)
	base, _ := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	v.SetF64(v.AtF64(seq-1, 0)+5, seq-1, 0) // perturb the last value
	pert, _ := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
	for i := 0; i < seq-1; i++ {
		for j := 0; j < dv; j++ {
			if math.Abs(base.AtF64(i, j)-pert.AtF64(i, j)) > 1e-12 {
				t.Errorf("future value changed out[%d,%d]", i, j)
			}
		}
	}
}

// §V2: differentiable w.r.t. q, k, v and β.
func TestDeltaNetGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 17))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	_, beta := randBeta(rng, seq)
	tape := autograd.NewTape()
	out, err := nn.DeltaNet(tape.Context(), q, k, v, beta)
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
		o, _ := nn.DeltaNet(backend.NewContext(), q, k, v, beta)
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
	check("beta", beta)
}

func TestDeltaNetErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.DeltaNet(backend.NewContext(), r(2, 3, 1), r(2, 3), r(2, 3), r(2, 1)); err == nil {
		t.Error("rank-3: want error")
	}
	if _, err := nn.DeltaNet(backend.NewContext(), r(2, 3), r(3, 3), r(2, 3), r(2, 1)); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := nn.DeltaNet(backend.NewContext(), r(2, 3), r(2, 4), r(2, 3), r(2, 1)); err == nil {
		t.Error("d_k mismatch: want error")
	}
	if _, err := nn.DeltaNet(backend.NewContext(), r(2, 3), r(2, 3), r(2, 3), r(2, 2)); err == nil {
		t.Error("beta not [seq,1]: want error")
	}
}

func ExampleDeltaNet() {
	// write value [10,20] under a key with β=1, then read the same key back:
	// the delta rule retrieves it exactly.
	key := toT([][]float64{{1, 0}})
	val := toT([][]float64{{10, 20}})
	out, _ := nn.DeltaNet(backend.NewContext(), key, key, val, betaCol([]float64{1}))
	fmt.Printf("%.2f %.2f\n", out.AtF64(0, 0), out.AtF64(0, 1))
	// Output:
	// 10.00 20.00
}
