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

// qknormAttnRef is an independent scaled-cosine attention (Henry et al. 2020):
// L2-normalize q,k rows (eps floor 1e-6), logits = g·(q̂·k̂), optional causal
// mask, row-softmax, weight·v. §V16 tier-1 parity oracle.
func qknormAttnRef(q, k, v [][]float64, g float64, causal bool) [][]float64 {
	l2 := func(row []float64) []float64 {
		var ss float64
		for _, x := range row {
			ss += x * x
		}
		n := math.Sqrt(ss + 1e-6)
		out := make([]float64, len(row))
		for i, x := range row {
			out[i] = x / n
		}
		return out
	}
	sq, sk, dk, dv := len(q), len(k), len(q[0]), len(v[0])
	qn := make([][]float64, sq)
	for i := range q {
		qn[i] = l2(q[i])
	}
	kn := make([][]float64, sk)
	for i := range k {
		kn[i] = l2(k[i])
	}
	off := sk - sq
	out := make([][]float64, sq)
	for i := range sq {
		logit := make([]float64, sk)
		for j := range sk {
			var d float64
			for t := 0; t < dk; t++ {
				d += qn[i][t] * kn[j][t]
			}
			logit[j] = g * d
			if causal && j > i+off {
				logit[j] = math.Inf(-1)
			}
		}
		mx := math.Inf(-1)
		for _, l := range logit {
			if l > mx {
				mx = l
			}
		}
		w := make([]float64, sk)
		var sum float64
		for j := range sk {
			w[j] = math.Exp(logit[j] - mx)
			sum += w[j]
		}
		o := make([]float64, dv)
		for j := range sk {
			w[j] /= sum
			for t := 0; t < dv; t++ {
				o[t] += w[j] * v[j][t]
			}
		}
		out[i] = o
	}
	return out
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2010.04245 Eq.2/4/5): QKNorm.Attention
// matches the independent scaled-cosine reference, both bidirectional and causal.
func TestQKNormAttentionParity(t *testing.T) {
	q := [][]float64{{1, 2, 0.5}, {-1, 0.3, 2}, {0.7, -0.7, 1}}
	k := [][]float64{{0.5, 1, -1}, {2, -0.5, 0.2}, {-1, 1, 1}}
	v := [][]float64{{1, 0}, {0, 1}, {0.5, 0.5}}
	for _, causal := range []bool{false, true} {
		qk := nn.NewQKNorm(tensor.F64, 1.7)
		got, err := qk.Attention(backend.NewContext(), toT(q), toT(k), toT(v), causal)
		if err != nil {
			t.Fatalf("causal=%v: %v", causal, err)
		}
		want := qknormAttnRef(q, k, v, 1.7, causal)
		for i := range want {
			for j := range want[i] {
				if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
					t.Errorf("causal=%v out[%d,%d]=%.10g want %.10g", causal, i, j, g, want[i][j])
				}
			}
		}
	}
}

// §V2: reading the attention weights via v=identity — rows are a valid softmax
// distribution (sum to 1, nonneg) and causal masking zeroes future positions.
func TestQKNormWeightsProperties(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 4))
	const s, d = 5, 4
	q, k := randMat(rng, s, d), randMat(rng, s, d)
	eye := tensor.New(tensor.F64, tensor.Shape{s, s}) // v=I ⇒ out == attention weights
	for i := range s {
		eye.SetF64(1, i, i)
	}
	qk := nn.NewQKNorm(tensor.F64, 2.3)
	w, err := qk.Attention(backend.NewContext(), q, k, eye, true)
	if err != nil {
		t.Fatal(err)
	}
	for i := range s {
		var sum float64
		for j := range s {
			wij := w.AtF64(i, j)
			if wij < -1e-12 {
				t.Errorf("weight[%d,%d]=%g negative", i, j, wij)
			}
			if j > i && math.Abs(wij) > 1e-12 {
				t.Errorf("causal leak weight[%d,%d]=%g", i, j, wij)
			}
			sum += wij
		}
		if math.Abs(sum-1) > 1e-9 {
			t.Errorf("row %d sums to %g, want 1", i, sum)
		}
	}
}

// §V16 tier-1: the paper's init g₀ = log₂(L²−L) (Eq.3).
func TestQKNormInitG(t *testing.T) {
	if got := nn.QKNormInitG(4); math.Abs(got-math.Log2(12)) > 1e-12 { // 4²−4=12
		t.Errorf("QKNormInitG(4)=%g want log2(12)=%g", got, math.Log2(12))
	}
}

// §V2: the attention is differentiable w.r.t. q, k, v and the learned scale G.
func TestQKNormGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(19, 5))
	const sq, sk, d = 3, 4, 3
	q, k, v := randMat(rng, sq, d), randMat(rng, sk, d), randMat(rng, sk, d)
	qk := nn.NewQKNorm(tensor.F64, 1.3)
	tape := autograd.NewTape()
	out, err := qk.Attention(tape.Context(), q, k, v, true)
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
		o, _ := qk.Attention(backend.NewContext(), q, k, v, true)
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
	check("G", qk.G)
}

func TestQKNormErrors(t *testing.T) {
	qk := nn.NewQKNorm(tensor.F64, 1)
	r2 := tensor.New(tensor.F64, tensor.Shape{2, 3})
	r3 := tensor.New(tensor.F64, tensor.Shape{2, 3, 1})
	if _, err := qk.Attention(backend.NewContext(), r3, r2, r2, false); err == nil {
		t.Error("rank-3 input: want error")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{2, 5})
	if _, err := qk.Attention(backend.NewContext(), r2, bad, bad, false); err == nil {
		t.Error("q/k head_dim mismatch: want error")
	}
}

// sumAll reduces every element of x to a scalar through ctx (differentiable).
func sumAll(ctx *backend.Context, x *tensor.Tensor) (*tensor.Tensor, error) {
	axes := make([]int, x.Ndim())
	for i := range axes {
		axes[i] = i
	}
	o, err := backend.Execute(ctx, backend.OpSum, []*tensor.Tensor{x}, backend.ReduceAttrs{Axes: axes})
	if err != nil {
		return nil, err
	}
	return o[0], nil
}

func ExampleQKNorm() {
	// scaled-cosine attention: two orthogonal keys, a query aligned with key 0.
	q := toT([][]float64{{1, 0}})
	k := toT([][]float64{{1, 0}, {0, 1}})
	v := toT([][]float64{{10, 0}, {0, 20}})
	qk := nn.NewQKNorm(tensor.F64, 5)
	out, _ := qk.Attention(backend.NewContext(), q, k, v, false)
	// query matches key 0 (cos=1) far more than key 1 (cos=0) ⇒ output ≈ v[0].
	fmt.Printf("%.2f %.2f\n", out.AtF64(0, 0), out.AtF64(0, 1))
	// Output:
	// 9.93 0.13
}
