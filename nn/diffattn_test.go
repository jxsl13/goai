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

// diffAttnRef independently computes Differential Attention (Ye et al. 2024,
// arXiv:2410.05258 Eq.1-2): two 1/√d-scaled softmax maps, subtracted with the
// reparameterized λ, applied to a shared V. §V16 tier-1 parity oracle.
func diffAttnRef(q1, k1, q2, k2, v [][]float64, lq1, lk1, lq2, lk2 []float64, lambdaInit float64, causal bool) [][]float64 {
	d := len(q1[0])
	scale := 1 / math.Sqrt(float64(d))
	softmaxMap := func(q, k [][]float64) [][]float64 {
		sq, sk := len(q), len(k)
		off := sk - sq
		out := make([][]float64, sq)
		for i := range sq {
			logit := make([]float64, sk)
			for j := range sk {
				var dp float64
				for t := range q[i] {
					dp += q[i][t] * k[j][t]
				}
				logit[j] = dp * scale
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
			var sum float64
			w := make([]float64, sk)
			for j := range sk {
				w[j] = math.Exp(logit[j] - mx)
				sum += w[j]
			}
			for j := range sk {
				w[j] /= sum
			}
			out[i] = w
		}
		return out
	}
	dot := func(a, b []float64) float64 {
		var s float64
		for i := range a {
			s += a[i] * b[i]
		}
		return s
	}
	a1, a2 := softmaxMap(q1, k1), softmaxMap(q2, k2)
	lambda := math.Exp(dot(lq1, lk1)) - math.Exp(dot(lq2, lk2)) + lambdaInit
	sq, sk, dv := len(q1), len(k1), len(v[0])
	out := make([][]float64, sq)
	for i := range sq {
		out[i] = make([]float64, dv)
		for j := range sk {
			w := a1[i][j] - lambda*a2[i][j]
			for t := 0; t < dv; t++ {
				out[i][t] += w * v[j][t]
			}
		}
	}
	return out
}

func setRow(t *tensor.Tensor, vals []float64) {
	for i, v := range vals {
		t.SetF64(v, 0, i)
	}
}

// §V16 tier-1 (paper CONFIRMED, arXiv:2410.05258 Eq.1-2, incl. unilm ref):
// DiffAttention.Attention matches the independent reference, bidirectional and
// causal, with non-trivial λ-vectors exercising the exp-dot reparameterization.
func TestDiffAttentionParity(t *testing.T) {
	q1 := [][]float64{{1, 2, 0.5}, {-1, 0.3, 2}, {0.7, -0.7, 1}}
	k1 := [][]float64{{0.5, 1, -1}, {2, -0.5, 0.2}, {-1, 1, 1}}
	q2 := [][]float64{{0.2, -1, 1}, {1, 1, 0}, {-0.5, 0.4, 0.9}}
	k2 := [][]float64{{1, 0, 0.5}, {-1, 2, 1}, {0.3, 0.3, -1}}
	v := [][]float64{{1, 0}, {0, 1}, {0.5, 0.5}}
	lq1, lk1 := []float64{0.1, -0.2, 0.3}, []float64{0.4, 0.1, -0.1}
	lq2, lk2 := []float64{-0.3, 0.2, 0.1}, []float64{0.2, -0.4, 0.5}
	for _, causal := range []bool{false, true} {
		da := nn.NewDiffAttention(tensor.F64, 3, 4) // layer 4
		setRow(da.LambdaQ1, lq1)
		setRow(da.LambdaK1, lk1)
		setRow(da.LambdaQ2, lq2)
		setRow(da.LambdaK2, lk2)
		got, err := da.Attention(backend.NewContext(), toT(q1), toT(k1), toT(q2), toT(k2), toT(v), causal)
		if err != nil {
			t.Fatalf("causal=%v: %v", causal, err)
		}
		want := diffAttnRef(q1, k1, q2, k2, v, lq1, lk1, lq2, lk2, da.LambdaInit, causal)
		for i := range want {
			for j := range want[i] {
				if g := got.AtF64(i, j); math.Abs(g-want[i][j]) > 1e-9 {
					t.Errorf("causal=%v out[%d,%d]=%.10g want %.10g", causal, i, j, g, want[i][j])
				}
			}
		}
	}
}

// §V2 (defining property): with identical query/key groups and zero λ-vectors
// (λ=λ_init), the two softmax maps cancel in common mode, so differential
// attention reduces to (1−λ_init)·ordinary-attention.
func TestDiffAttentionCommonModeCancellation(t *testing.T) {
	rng := rand.New(rand.NewPCG(23, 7))
	const s, d = 4, 3
	q, k, v := randMat(rng, s, d), randMat(rng, s, d), randMat(rng, s, d)
	da := nn.NewDiffAttention(tensor.F64, d, 3) // λ-vectors default to 0 ⇒ λ=λ_init
	got, err := da.Attention(backend.NewContext(), q, k, q, k, v, true)
	if err != nil {
		t.Fatal(err)
	}
	// ordinary causal attention via the reference (λ=0 ⇒ single softmax map).
	qm, km, vm := toRows(q), toRows(k), toRows(v)
	zero := make([]float64, d)
	ord := diffAttnRef(qm, km, qm, km, vm, zero, zero, zero, zero, 0, true)
	scale := da.SublayerScale() // 1 − λ_init
	for i := range s {
		for j := 0; j < d; j++ {
			want := scale * ord[i][j]
			if g := got.AtF64(i, j); math.Abs(g-want) > 1e-9 {
				t.Errorf("out[%d,%d]=%.10g want (1−λ_init)·attn=%.10g", i, j, g, want)
			}
		}
	}
}

// §V16 tier-1: λ_init = 0.8 − 0.6·exp(−0.3·(l−1)) (Eq.3), and SublayerScale=1−λ_init.
func TestDiffLambdaInit(t *testing.T) {
	if got := nn.DiffLambdaInit(1); math.Abs(got-0.2) > 1e-12 { // 0.8−0.6·1
		t.Errorf("DiffLambdaInit(1)=%g want 0.2", got)
	}
	if got := nn.DiffLambdaInit(1000); math.Abs(got-0.8) > 1e-6 { // deep layers → 0.8
		t.Errorf("DiffLambdaInit(1000)=%g want ~0.8", got)
	}
	da := nn.NewDiffAttention(tensor.F64, 2, 5)
	if math.Abs(da.SublayerScale()-(1-nn.DiffLambdaInit(5))) > 1e-12 {
		t.Errorf("SublayerScale=%g want 1−λ_init", da.SublayerScale())
	}
}

// §V2: differentiable w.r.t. q1,k1,q2,k2,v and all four λ-vectors.
func TestDiffAttentionGradcheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(29, 11))
	const sq, sk, d = 3, 4, 3
	q1, k1 := randMat(rng, sq, d), randMat(rng, sk, d)
	q2, k2 := randMat(rng, sq, d), randMat(rng, sk, d)
	v := randMat(rng, sk, d)
	da := nn.NewDiffAttention(tensor.F64, d, 4)
	// perturb λ-vectors off zero so the exp-dot gradients are non-degenerate.
	for _, p := range da.Params() {
		for j := 0; j < d; j++ {
			p.SetF64(0.1*float64(j+1), 0, j)
		}
	}
	tape := autograd.NewTape()
	out, err := da.Attention(tape.Context(), q1, k1, q2, k2, v, true)
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
		o, _ := da.Attention(backend.NewContext(), q1, k1, q2, k2, v, true)
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
	check("q1", q1)
	check("k1", k1)
	check("q2", q2)
	check("k2", k2)
	check("v", v)
	check("λq1", da.LambdaQ1)
	check("λk1", da.LambdaK1)
	check("λq2", da.LambdaQ2)
	check("λk2", da.LambdaK2)
}

func TestDiffAttentionErrors(t *testing.T) {
	da := nn.NewDiffAttention(tensor.F64, 3, 1)
	r2 := tensor.New(tensor.F64, tensor.Shape{2, 3})
	r3 := tensor.New(tensor.F64, tensor.Shape{2, 3, 1})
	if _, err := da.Attention(backend.NewContext(), r3, r2, r2, r2, r2, false); err == nil {
		t.Error("rank-3 input: want error")
	}
	bad := tensor.New(tensor.F64, tensor.Shape{2, 5})
	if _, err := da.Attention(backend.NewContext(), r2, bad, r2, r2, r2, false); err == nil {
		t.Error("head_dim mismatch: want error")
	}
}

// toRows reads a rank-2 tensor into [][]float64 for the reference helpers.
func toRows(t *tensor.Tensor) [][]float64 {
	r, c := t.Shape()[0], t.Shape()[1]
	out := make([][]float64, r)
	for i := range r {
		out[i] = make([]float64, c)
		for j := 0; j < c; j++ {
			out[i][j] = t.AtF64(i, j)
		}
	}
	return out
}

func ExampleDiffAttention() {
	// two key positions; query aligns with key 0 in BOTH groups, so the common-mode
	// attention cancels and the output is a (1−λ_init)-scaled copy of v[0].
	q1 := toT([][]float64{{4, 0}})
	k1 := toT([][]float64{{1, 0}, {0, 1}})
	v := toT([][]float64{{10, 0}, {0, 20}})
	da := nn.NewDiffAttention(tensor.F64, 2, 1) // λ_init=0.2 ⇒ scale 0.8
	out, _ := da.Attention(backend.NewContext(), q1, k1, q1, k1, v, false)
	fmt.Printf("%.2f %.2f\n", out.AtF64(0, 0), out.AtF64(0, 1))
	// Output:
	// 7.55 0.89
}
