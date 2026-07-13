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

// glaRef independently runs the GLA recurrence S_t=Diag(α_t)S_{t-1}+k_tᵀv_t,
// o_t=q_tS_t. §V16 tier-1 parity oracle.
func glaRef(q, k, v, gate [][]float64) [][]float64 {
	seq, dk, dv := len(q), len(q[0]), len(v[0])
	S := make([][]float64, dk)
	for i := range S {
		S[i] = make([]float64, dv)
	}
	out := make([][]float64, seq)
	for t := 0; t < seq; t++ {
		for i := 0; i < dk; i++ {
			for j := 0; j < dv; j++ {
				S[i][j] = gate[t][i]*S[i][j] + k[t][i]*v[t][j]
			}
		}
		o := make([]float64, dv)
		for j := 0; j < dv; j++ {
			for i := 0; i < dk; i++ {
				o[j] += q[t][i] * S[i][j]
			}
		}
		out[t] = o
	}
	return out
}

// linAttnRef computes ungated causal linear attention via the ATTENTION form
// o_t = Σ_{s≤t}(q_t·k_s)v_s — a distinct code path from the state recurrence.
func linAttnRef(q, k, v [][]float64) [][]float64 {
	seq, dk, dv := len(q), len(q[0]), len(v[0])
	out := make([][]float64, seq)
	for t := 0; t < seq; t++ {
		o := make([]float64, dv)
		for s := 0; s <= t; s++ {
			var qk float64
			for i := 0; i < dk; i++ {
				qk += q[t][i] * k[s][i]
			}
			for j := 0; j < dv; j++ {
				o[j] += qk * v[s][j]
			}
		}
		out[t] = o
	}
	return out
}

func randGate(rng *rand.Rand, r, c int) *tensor.Tensor {
	t := tensor.New(tensor.F64, tensor.Shape{r, c})
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			t.SetF64(1/(1+math.Exp(-rng.NormFloat64())), i, j) // σ(·) ∈ (0,1)
		}
	}
	return t
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2312.06635): GatedLinearAttention matches the
// independent recurrence over random q,k,v and a data-dependent gate.
func TestGLAParity(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	const seq, dk, dv = 5, 3, 4
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	gate := randGate(rng, seq, dk)
	got, err := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
	if err != nil {
		t.Fatal(err)
	}
	want := glaRef(toRows(q), toRows(k), toRows(v), toRows(gate))
	for i := 0; i < seq; i++ {
		for j := 0; j < dv; j++ {
			if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want %.10g", i, j, g, want[i][j])
			}
		}
	}
}

// §V2 (defining property): with the gate all-ones GLA reduces to ungated linear
// attention (matched against the attention-form oracle).
func TestGLAReducesToLinearAttention(t *testing.T) {
	rng := rand.New(rand.NewPCG(6, 10))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	ones := tensor.New(tensor.F64, tensor.Shape{seq, dk})
	for i := range ones.Numel() {
		ones.SetF64(1, tensor.Unravel(i, ones.Shape())...)
	}
	got, err := nn.GatedLinearAttention(backend.NewContext(), q, k, v, ones)
	if err != nil {
		t.Fatal(err)
	}
	want := linAttnRef(toRows(q), toRows(k), toRows(v))
	for i := 0; i < seq; i++ {
		for j := 0; j < dv; j++ {
			if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
				t.Errorf("gate=1 out[%d,%d]=%.10g want linear-attn %.10g", i, j, g, want[i][j])
			}
		}
	}
}

// §V2: the recurrence is causal — a future key cannot affect an earlier output.
func TestGLACausality(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	const seq, dk, dv = 4, 2, 3
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	gate := randGate(rng, seq, dk)
	base, _ := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
	k.SetF64(k.AtF64(seq-1, 0)+3, seq-1, 0) // perturb the last key
	pert, _ := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
	for i := 0; i < seq-1; i++ { // outputs before the last step must be unchanged
		for j := 0; j < dv; j++ {
			if math.Abs(base.AtF64(i, j)-pert.AtF64(i, j)) > 1e-12 {
				t.Errorf("future key changed out[%d,%d]: %g→%g", i, j, base.AtF64(i, j), pert.AtF64(i, j))
			}
		}
	}
}

// §V2: differentiable w.r.t. q, k, v and the gate.
func TestGLAGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(8, 12))
	const seq, dk, dv = 4, 3, 2
	q, k, v := randMat(rng, seq, dk), randMat(rng, seq, dk), randMat(rng, seq, dv)
	gate := randGate(rng, seq, dk)
	tape := autograd.NewTape()
	out, err := nn.GatedLinearAttention(tape.Context(), q, k, v, gate)
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
		o, _ := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
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
	check("gate", gate)
}

func TestGLAErrors(t *testing.T) {
	r := func(shape ...int) *tensor.Tensor { return tensor.New(tensor.F64, tensor.Shape(shape)) }
	if _, err := nn.GatedLinearAttention(backend.NewContext(), r(2, 3, 1), r(2, 3), r(2, 3), r(2, 3)); err == nil {
		t.Error("rank-3: want error")
	}
	if _, err := nn.GatedLinearAttention(backend.NewContext(), r(2, 3), r(3, 3), r(2, 3), r(2, 3)); err == nil {
		t.Error("seq mismatch: want error")
	}
	if _, err := nn.GatedLinearAttention(backend.NewContext(), r(2, 3), r(2, 4), r(2, 3), r(2, 3)); err == nil {
		t.Error("d_k mismatch: want error")
	}
}

func ExampleGatedLinearAttention() {
	// d_k=d_v=1, gate=1 ⇒ ungated linear attention: state accumulates keys·values.
	q := toT([][]float64{{1}, {1}})
	k := toT([][]float64{{1}, {1}})
	v := toT([][]float64{{2}, {3}})
	gate := toT([][]float64{{1}, {1}})
	out, _ := nn.GatedLinearAttention(backend.NewContext(), q, k, v, gate)
	fmt.Println(out.AtF64(0, 0), out.AtF64(1, 0))
	// Output:
	// 2 5
}
